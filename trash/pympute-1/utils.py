
import os
import numpy as np
import pylab as plt
import pickle
import pyreadr
import tempfile
import pandas as pd
import sklearn as sk
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score

import matplotlib.patches as mpatches
from sklearn.neighbors import KernelDensity

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def data_load(prefix='data/',i=0,part = '1st'):
    frame = {'data':[],'masked':[]}
    template = 'dffinal_repeat_select_widformat_t{}_{}{}.RData'
    for typ in ['','masked_']:
        fname = template.format(i,typ,part)
        result = pyreadr.read_r(prefix+fname)
        key = list(result.keys())[0]
        if typ=='':
            frame['data'] = result[key]
        else:
            frame['masked'] = result[key]
    return frame

def get_rescale(xx):
    normin = np.nanmin(xx,axis=0,keepdims=1)
    normax = np.nanmax(xx,axis=0,keepdims=1)
    return normin,normax
    
def set_rescale(xx,normin,normax):
    xx = (xx-normin)/(normax-normin)
#     xx = xx/normax
    return xx

def res_rescale(xx,normin,normax):
    xx = xx*(normax-normin)+normin
    return xx

def fill_random(df):
    dd = deepcopy(df)
    cols = dd.columns
    for col in cols:
        fmiss = dd[col].isna().values
        nmiss = np.sum(fmiss)
        smiss = dd[col].dropna().sample(nmiss,replace=1).values
        dd.loc[fmiss,col] = smiss
    return dd

# dd = fill_random(dd)
# dd

class Imputer:
    def __init__(self,data_frame,model,loss_f=None,fill_method='random',save_history=False):
        self.data_frame = data_frame
        self.disna = data_frame.isna()
        
        if type(model) is str:
            if model=='LR':
                from sklearn.linear_model import LinearRegression
                self.model_class = LinearRegression
            elif model=='SVR':
                from sklearn.svm import SVR
                self.model_class = SVR
            elif model=='SGD':
                from sklearn.linear_model import SGDRegressor
                self.model_class = SGDRegressor
            elif model=='Ridge':
                from sklearn.linear_model import Ridge
                self.model_class = Ridge
            elif model=='Lasso':
                from sklearn.linear_model import Lasso
                self.model_class = Lasso          
            elif model=='ElasticNet':
                from sklearn.linear_model import ElasticNet
                self.model_class = ElasticNet
            elif model=='KNN':
                from sklearn.neighbors import KNeighborsRegressor
                self.model_class = KNeighborsRegressor
            elif model=='GP':
                from sklearn.gaussian_process import GaussianProcessRegressor
                self.model_class = GaussianProcessRegressor
            elif model=='DT':
                from sklearn.tree import DecisionTreeRegressor
                self.model_class = DecisionTreeRegressor          
            elif model=='GB':
                from sklearn.ensemble import GradientBoostingRegressor
                self.model_class = GradientBoostingRegressor
            elif model=='Ada':
                from sklearn.ensemble import AdaBoostRegressor
                self.model_class = AdaBoostRegressor
            elif model=='RF':
                from sklearn.ensemble import RandomForestRegressor
                self.model_class = RandomForestRegressor
            elif model=='ET':
                from sklearn.tree import ExtraTreeRegressor
                self.model_class = ExtraTreeRegressor
            elif model=='ExT':
                from sklearn.ensemble import ExtraTreesRegressor
                self.model_class = ExtraTreesRegressor 
            elif model=='Bag':
                self.model_class = BaggingRegressor
            elif model=='XGB':
                self.model_class = XGBRegressor             
            else:
                models_dict = {'LR':'Linear Regression',
                               'SVR':'Support Vector Resreggor',
                               'SGD':' Stochastic Gradient Descent Regressor',
                               'Ridge':'Ridge Regressor',
                               'Lasso':'Lasso Regressor',
                               'ElasticNet':'ElasticNet Regressor',
                               'KNN':'KNeighbors Regressor',
                               'GP':'Gaussian Process Regressor',
                               'DT':'Decision Tree Regressor',
                               'ET':'Extra Tree Regressor',
                               'GB':'Gradient Boosting Regressor',
                               'Ada':'Ada Boost Regressor',
                               'RF':'Random Forest Regressor',
                               'ExT':'Extra Trees Regressor',
                               'Bag':'Bagging Regressor',
                               'XGB':'XGBoost Regressor' 
                              }
                print('implimented models are:')
                for k,v in models_dict.items():
                    print(55*'-')
                    print('| {:10s} | {:40s}|'.format(k,v))
                print(55*'-')
            
                assert 0,'The model is not recognized!'
        else:
            self.model_class = model
            pass #TODO check the needed methods
        
        self.cols = data_frame.columns
        self.ncol = len(self.cols)
        self.save_history = save_history
        self.history = 0
        if self.save_history:
            self.history = {i:[] for i in self.cols}
        
        if fill_method=='random':
            self.data_frame = fill_random(self.data_frame)
        elif fill_method=='mean':
            self.data_frame = self.data_frame.fillna(dd.mean())
        else:
            assert 0,'fill_mothod is not recognized!'
        
        if loss_f is None:
            from sklearn.metrics import mean_squared_error
            self.loss_f = mean_squared_error
        self.loss_frame = pd.DataFrame(columns=self.cols)
        self.models = {col:None for col in self.cols}

    def impute(self,n_it,inds=None,trsh=-np.inf,**kargs):
        if inds is None:
            inds = np.arange(self.ncol)
            np.random.shuffle(inds)
    
        ilf = self.loss_frame.shape[0]
        
        pbar = tqdm(total=n_it*len(inds))
        
#        for i in tqdm(range(n_it)):
#            for j in tqdm(range(len(inds)), leave=False):
        for i in range(n_it):
            for j in range(len(inds)):
#                perc = 100*(i*self.ncol+ji+1)/(n_it*self.ncol)
#                print('{:6.2f}%  <{:100s}>'.format( perc,int(perc)*'=' ),end='\r')
                pbar.update(1)

                col = self.cols[inds[j]]
                fisna = self.disna[col]
                if fisna.mean()==0:
                    self.loss_frame.loc[ilf+i,col] = 0
                    continue

                if i>2:
                    c_loss = self.loss_frame.iloc[-1][col]
                    if c_loss<trsh:
                        self.loss_frame.loc[ilf+i,col] = c_loss
                        continue

                x = self.data_frame.drop(columns=[col])
                y = self.data_frame[col]

                x_train = x[~fisna]
                y_train = y[~fisna]
                x_test = x[fisna]
                y_test = y[fisna]

#                 inp = x_train.shape[1]
#                 out = 1
                if self.models[col] is None:
                    model = self.model_class()
                    self.models[col] = model
                else:
                    model = self.models[col]

        #         print(col)
                model.fit(x_train,y_train,**kargs)
                pred = model.predict(x_test)
                if pred.ndim>1:
                    pred = pred[:,0]

                self.loss_frame.loc[ilf+i,col] = self.loss_f(y_test,pred)
                self.data_frame.loc[fisna,col] = pred
                if self.save_history:
                    self.history[col].append(pred)
        pbar.close()

    def plot_loss_frame(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,7))
        for lcol in self.loss_frame.columns:
            ax.plot(self.loss_frame[lcol])
        ax.set_yscale('log')
        return ax

    def compare(self,truth,com_f=None,ax=None,width=0.35,legend=True,with_random=True,save=None):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(30,10))
        lsses = []
        lsses_r = []
        for col in self.cols:
            fisna = self.disna[col]
            preds = self.data_frame.loc[fisna,col]
            nmiss = np.sum(fisna.values)
            smiss = self.data_frame.loc[~fisna,col].sample(nmiss,replace=1).values
            trth = truth.loc[fisna,col]
            if com_f is None:
                lss = self.loss_f(trth,preds)
                lssr = self.loss_f(trth,smiss)
            else:
                lss = com_f(trth,preds)
                lssr = com_f(trth,smiss)
            lsses.append(lss)
            lsses_r.append(lssr)
        x = np.arange(self.ncol)  # the label locations
        if with_random:
            ax.bar(x-width/2,lsses, width,color='b',label='imputed')
            ax.bar(x+width/2,lsses_r, width,color='r',label='radnom')
        else:
            ax.bar(x,lsses, width,color='b',label='imputed')
        if legend: ax.legend()
        
        if save is not None:
            plt.savefig(save+'.jpg',dpi=200)
        
        return ax


    def save(self,fname):
        self.loss_frame.to_csv(fname+'loss_frame.csv',index=0)
        save(fname+'model.pkl',self.models)
        if self.history:
            save(fname+'history.pkl',self.history)
        
    def load(self,fname):
        self.loss_frame = pd.read_csv(fname+'loss_frame.csv')
        self.models = load(fname+'model.pkl')
        if os.path.exists(fname+'history.pkl'):
            self.history = load(fname+'history.pkl')
            for col in self.cols:
                fisna = self.disna[col]
                self.data_frame.loc[fisna,col] = self.history[col][-1]

    def dist(self,col,truth=None,cl=25,bandwidth=0.02,ax=None,alpha=0.3,legend=True,save=None):
        fisna = self.disna[col]
        preds = self.data_frame.loc[fisna,col].values
        if truth is None:
            gd = self.data_frame.loc[~fisna,col]
            tlbl = 'existing data'
        else:
            gd = truth.loc[fisna,col]
            tlbl = 'ground truth'
        gd = gd.values
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,5))
        m1,l1,u1 = mykde(gd, cl=cl, color='b', alpha=alpha, kernel='gaussian', bandwidth=bandwidth, ax=ax)
        m2,l2,u2 = mykde(preds, cl=cl, color='r', alpha=alpha, kernel='gaussian', bandwidth=bandwidth, ax=ax)
        ax.set_ylim(bottom=0)
        red_patch = mpatches.Patch(color='red', label='imputed data')
        blue_patch = mpatches.Patch(color='blue', label=tlbl)
        if legend: ax.legend(handles=[red_patch, blue_patch])
        
        if save is not None:
            plt.savefig(save+'.jpg',dpi=200)
        
        return (m1,l1,u1),(m2,l2,u2)

    def dist_all(self,truth=None,cl=25,bandwidth=0.02,save=None):
#        ny = int(np.sqrt(self.ncol))
#        nx = (self.ncol//ny)+1
        
        nn = self.ncol
        ny = int(np.sqrt(nn))
        nx = (nn//ny)
        if nn/ny!=nn//ny:
            nx = nx+1
        
        fig,axs = plt.subplots(ny,nx,figsize=(nx*2.2,ny*2.2), dpi=100)
        ii = 0
        for column in range(nx):
            for row in range(ny):
                ax = axs[row,column]
                if ii==self.ncol: 
                    ax.axis("off")
                    continue
                col = self.cols[ii]
                self.dist(col,truth,cl=cl,bandwidth=bandwidth,ax=ax,legend=0)
                ax.set_title(col)
                ii = ii+1
        plt.tight_layout()
        if save is not None:
            plt.savefig(save+'.jpg',dpi=200)

    def metric_opt(self,metric_f,truth):
        lsses = []
        for col in self.cols:
            fisna = self.disna[col]
            preds = self.data_frame.loc[fisna,col]
            trth = truth.loc[fisna,col]
            lss = metric_f(trth,preds)
            lsses.append(lss)
        return np.array(lsses)

    def general_report(self,truth):

        def metr_ttest(tr_,im_): return ttest_ind(tr_.values,im_.values)[1]
        def mape_q1(tr_,im_): return mape_pranged(tr_,im_,0,25)
        def mape_q2(tr_,im_): return mape_pranged(tr_,im_,25,50)
        def mape_q3(tr_,im_): return mape_pranged(tr_,im_,50,75)
        def mape_q4(tr_,im_): return mape_pranged(tr_,im_,75,100)

        metr_dict = {'rmse':rmse,
                     'mape':mape,
                     'r2_score':r2_score,
                     'ttest':metr_ttest,
                     'mape_q1':mape_q1,
                     'mape_q2':mape_q2,
                     'mape_q3':mape_q3,
                     'mape_q4':mape_q4
                     }
        lsses = []
        for key in metr_dict.keys():
#            lsses.append(self.metric_opt(metr_dict[key],truth))
            lsses.append(metric_opt(self,metr_dict[key],truth))
        lsses = np.array(lsses)
        df = pd.DataFrame(data=lsses.T,columns=metr_dict.keys())
        df = df.apply(pd.to_numeric, errors='ignore')
        return df
        
    def metric_opt(self,metric_f,truth):
        lsses = pd.DataFrame(columns=['feature','value'])
        ii = 0
        for col in self.cols:
            fisna = self.disna[col]
            preds = self.data_frame.loc[fisna,col].values
            trth = truth.loc[fisna,col].values
            ndp = len(preds)
            for i in range(ndp):
                lsses.loc[ii,'feature'] = col
                lsses.loc[ii,'value'] = metric_f(preds[i],trth[i])
                ii = ii+1
            
        lsses = lsses.apply(pd.to_numeric, errors='ignore')
        return lsses  

    def model_reset(self,col=None):
        if col is None:
            for col in self.cols: self.models[col] = self.model_class()
        else:
            self.models[col] = self.model_class()

def metric_opt(self,metric_f,truth):
    lsses = []
    for col in self.cols:
        fisna = self.disna[col]
        preds = self.data_frame.loc[fisna,col]
        trth = truth.loc[fisna,col]
        lss = metric_f(trth,preds)
        lsses.append(lss)
    return np.array(lsses)

def mykde(x,
          cl = None,
          color = 'b',
          alpha = 0.3,
          kernel = 'gaussian',
          bandwidth = 0.2,
          ax = None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,5))
    x = x.reshape(-1, 1)
    xx = np.linspace(x.min(),x.max(), 100)[:, None]

    # kdep = sns.kdeplot(gd, color='b',n_levels=[0.2,0.8], shade=True, ax=ax)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x)
    kdep = kde.score_samples(xx)

    def estimate(x):
        return np.exp(kde.score_samples(np.array([x])[:, None]))

    ax.plot(xx,np.exp(kdep),color=color,lw=0.8)
    if cl is not None:
        dcl = 100-cl
        lower = np.percentile(x[:,0],dcl/2)
        upper = np.percentile(x[:,0],100-dcl/2)
        xx = xx[:,0]
        ax.fill_between(xx,xx-xx,np.exp(kdep),alpha=alpha,color=color)
        ax.plot(2*[lower],[0,estimate(lower)[0]],color=color)
        ax.plot(2*[upper],[0,estimate(upper)[0]],color=color)
        
        return np.percentile(x[:,0],50),lower,upper

def mape(tr_ ,im_):
    return np.mean((np.abs(tr_ - im_)) / (np.abs(tr_) + 0.01)) * 100

def rmse(tr_ ,im_):
    return np.sqrt( np.mean((tr_ - im_)**2) )

def mape_pranged(tr_,im_,p1,p2):
    p1,p2 = np.percentile(tr_,p1),np.percentile(tr_,p2)
    filt = (p1<tr_) & (tr_<p2)
    return mape(tr_[filt],im_[filt])

#def make_keras_picklable():
#    def __getstate__(self):
#        model_str = ""
#        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
#            save_model(self, fd.name, overwrite=True)
#            model_str = fd.read()
#        d = {'model_str': model_str}
#        return d

#    def __setstate__(self, state):
#        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
#            fd.write(state['model_str'])
#            fd.flush()
#            model = load_model(fd.name)
#        self.__dict__ = model.__dict__


#    cls = Model
#    cls.__getstate__ = __getstate__
#    cls.__setstate__ = __setstate__

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def dense_model(inp,out,nlayer,batchnorm=False,dropout=False,loss='mean_squared_error',optimizer='adam',kernel_initializer='normal'):
    dims = np.linspace(inp,out,nlayer).astype(int)[1:-1]
    input_img = keras.Input(shape=inp)
    x = input_img
    for dim in dims:
        x = layers.Dense(dim, kernel_initializer=kernel_initializer, activation='relu')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if dropout!=0:
            x = layers.Dropout(dropout)(x)
    # create model
    if dropout!=0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(out, kernel_initializer='normal')(x)
    
    model = keras.Model(input_img, x)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except:
        return False
    return True

def save(fname,data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

