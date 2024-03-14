
import os
import numpy as np
import pylab as plt
import pickle
#import pyreadr
import tempfile
import pandas as pd
import sklearn as sk
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score

import matplotlib.patches as mpatches
from sklearn.neighbors import KernelDensity

#def data_load(prefix='data/',i=0,part = '1st'):
#    frame = {'data':[],'masked':[]}
#    template = 'dffinal_repeat_select_widformat_t{}_{}{}.RData'
#    for typ in ['','masked_']:
#        fname = template.format(i,typ,part)
#        result = pyreadr.read_r(prefix+fname)
#        key = list(result.keys())[0]
#        if typ=='':
#            frame['data'] = result[key]
#        else:
#            frame['masked'] = result[key]
#    return frame

def get_range(xx):
    xmin = xx.min()#np.nanmin(xx,axis=0,keepdims=1)
    xmax = xx.max()#np.nanmax(xx,axis=0,keepdims=1)
    return xmin,xmax
    
def set_range(df,xmin=0,xmax=1,excepts=[]):
#     xx = (xx-xmin)/(xmax-xmin)
    for col in df.columns:
        if col in excepts: continue
        if df[col].std()==0: continue
        df[col] = (df[col]-xmin[col])/(xmax[col]-xmin[col])
    return df

def reset_range(df,xmin=0,xmax=1,excepts=[]):
#     xx = (xx-xmin)/(xmax-xmin)
    for col in df.columns:
        if col in excepts: continue
        if df[col].std()==0: continue
        dr = xmax[col]-xmin[col]
        df[col] = df[col]*dr+xmin[col]
    return df

def get_mean_std(xx):
    xmean = np.nanmean(xx,axis=0,keepdims=1)
    xstd = np.nanstd(xx,axis=0,keepdims=1)
    return xmean,xstd

def set_mean_std(xx,xmean=0,xstd=1):
    normean0 = np.nanmean(xx,axis=0,keepdims=1)
    norstd0 = np.nanstd(xx,axis=0,keepdims=1)
    xx = (xx-normean0)/norstd0
    xx = xmean+xstd*xx
    return xx

# def res_rescale(xx,normin,normax):
#     xx = xx*(normax-normin)+normin
#     return xx

def fill_random(df,batch_input):
    dd = deepcopy(df)
    cols = dd.columns
    for col in cols:
        fmiss = dd[col].isna().values
        nmiss = np.sum(fmiss)
        if batch_input and nmiss==0:
            continue
        smiss = dd[col].dropna().sample(nmiss,replace=1).values
        dd.loc[fmiss,col] = smiss
    return dd

def gfill_random(df,batch_input):
    dd = deepcopy(df)
    cols = dd.columns
    for col in cols:
        fmiss = dd[col].isna().values
        nmiss = cp.sum(fmiss)
        if batch_input and nmiss==0:
            continue
        print(col,nmiss)
        smiss = dd[col].dropna().sample(nmiss,replace=1).values
        dd.loc[fmiss,col] = smiss
    return dd


def get_model(model,gpu=False):
    if not gpu:
        if model=='LR-r':
            from sklearn.linear_model import LinearRegression
            model_class = LinearRegression
        elif model=='SVM-r':
            from sklearn.svm import SVR
            model_class = SVR
        elif model=='SGD-r':
            from sklearn.linear_model import SGDRegressor
            model_class = SGDRegressor
        elif model=='Ridge-r':
            from sklearn.linear_model import Ridge
            model_class = Ridge
        elif model=='Lasso-r':
            from sklearn.linear_model import Lasso
            model_class = Lasso          
        elif model=='ElasticNet-r':
            from sklearn.linear_model import ElasticNet
            model_class = ElasticNet
        elif model=='BRidge-r':
            from sklearn.linear_model import BayesianRidge
            model_class = BayesianRidge            
        elif model=='KNN-r':
            from sklearn.neighbors import KNeighborsRegressor
            model_class = KNeighborsRegressor
        elif model=='GP-r':
            from sklearn.gaussian_process import GaussianProcessRegressor
            model_class = GaussianProcessRegressor
        elif model=='DT-r':
            from sklearn.tree import DecisionTreeRegressor
            model_class = DecisionTreeRegressor          
        elif model=='GB-r':
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
        elif model=='Ada-r':
            from sklearn.ensemble import AdaBoostRegressor
            model_class = AdaBoostRegressor
        elif model=='RF-r':
            from sklearn.ensemble import RandomForestRegressor
            model_class = RandomForestRegressor
        elif model=='ET-r':
            from sklearn.tree import ExtraTreeRegressor
            model_class = ExtraTreeRegressor
        elif model=='ExT-r':
            from sklearn.ensemble import ExtraTreesRegressor
            model_class = ExtraTreesRegressor 
        elif model=='Bag-r':
            from sklearn.ensemble import BaggingRegressor
            model_class = BaggingRegressor
        elif model=='XGB-r':
            from xgboost import XGBRegressor
            model_class = XGBRegressor
        elif model=='LR-c':
            from sklearn.linear_model import LogisticRegression
            model_class = LogisticRegression
        elif model=='SVM-c':
            from sklearn.svm import SVC
            model_class = SVC
        elif model=='SGD-c':
            from sklearn.linear_model import SGDClassifier
            model_class = SGDClassifier
        elif model=='Ridge-c':
            from sklearn.linear_model import RidgeClassifier
            model_class = RidgeClassifier
        elif model=='KNN-c':
            from sklearn.neighbors import KNeighborsClassifier
            model_class = KNeighborsClassifier
        elif model=='GP-c':
            from sklearn.gaussian_process import GaussianProcessClassifier
            model_class = GaussianProcessClassifier
        elif model=='DT-c':
            from sklearn.tree import DecisionTreeClassifier
            model_class = DecisionTreeClassifier          
        elif model=='GB-c':
            from sklearn.ensemble import GradientBoostingClassifier
            model_class = GradientBoostingClassifier
        elif model=='Ada-c':
            from sklearn.ensemble import AdaBoostClassifier
            model_class = AdaBoostClassifier
        elif model=='RF-c':
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
        elif model=='ET-c':
            from sklearn.tree import ExtraTreeClassifier
            model_class = ExtraTreeClassifier
        elif model=='ExT-c':
            from sklearn.ensemble import ExtraTreesClassifier
            model_class = ExtraTreesClassifier 
        elif model=='Bag-c':
            from sklearn.ensemble import BaggingClassifier
            model_class = BaggingClassifier
        elif model=='XGB-c':
            from xgboost import XGBClassifier
            model_class = XGBClassifier           

        else:
            print('implimented regression models are:')
            for k,v in cpu_regressors_list().items():
                print(68*'-')
                print('| {:12s} | {:50s}|'.format(k,v))
            print(68*'-')
            print()
            print('implimented classification models are:')
            for k,v in cpu_classifiers_list().items():
                print(68*'-')
                print('| {:12s} | {:50s}|'.format(k,v))
            print(68*'=')

            assert 0, f'The model {model} is not recognized!'
    else:
        if model=='LR-r':
            from cuml.linear_model import LinearRegression
            model_class = LinearRegression
        elif model=='SVM-r':
            from cuml.svm import SVR
            model_class = SVR
        elif model=='SGD-r':
            from cuml.linear_model import MBSGDRegressor
            model_class = MBSGDRegressor
        elif model=='Ridge-r':
            from cuml.linear_model import Ridge
            model_class = Ridge
        elif model=='Lasso-r':
            from cuml.linear_model import Lasso
            model_class = Lasso          
        elif model=='ElasticNet-r':
            from cuml.linear_model import ElasticNet
            model_class = ElasticNet
        elif model=='KNN-r':
            from cuml.neighbors import KNeighborsRegressor
            model_class = KNeighborsRegressor
        elif model=='RF-r':
            from cuml.ensemble import RandomForestRegressor
            model_class = RandomForestRegressor
        elif model=='XGB-r':
            from xgboost import XGBRegressor
            def XGBRegressor_p(**kargs):
                return XGBRegressor(tree_method='gpu_hist',**kargs)
            model_class = XGBRegressor_p
        elif model=='LR-c':
            from cuml.linear_model import LogisticRegression
            model_class = LogisticRegression
        elif model=='SVM-c':
            from cuml.svm import SVC
            model_class = SVC
        elif model=='SGD-c':
            from cuml.linear_model import MBSGDClassifier
            model_class = MBSGDClassifier
        elif model=='KNN-c':
            from cuml.neighbors import KNeighborsClassifier
            model_class = KNeighborsClassifier
        elif model=='RF-c':
            from cuml.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
        elif model=='XGB-c':
            from xgboost import XGBClassifier
            def XGBClassifier_p(**kargs):
                return XGBClassifier(tree_method='gpu_hist',**kargs)
            model_class = XGBClassifier_p

        else:
            print('implimented regression models are:')
            for k,v in gpu_regressors_list().items():
                print(68*'-')
                print('| {:12s} | {:50s}|'.format(k,v))
            print(68*'-')
            print()
            print('implimented classification models are:')
            for k,v in gpu_classifiers_list().items():
                print(68*'-')
                print('| {:12s} | {:50s}|'.format(k,v))
            print(68*'=')

            assert 0, f'The model {model} is not recognized!'
    return model_class

def cpu_regressors_list():
    return {'LR-r':'Linear Regression',
            'SVM-r':'Support Vector Resreggor',
            'SGD-r':'Stochastic Gradient Descent Regressor',
            'Ridge-r':'Ridge Regressor',
            'Lasso-r':'Lasso Regressor',
            'ElasticNet-r':'ElasticNet Regressor',
            'BRidge-r':'Bayesian Ridge',
            'KNN-r':'KNeighbors Regressor',
            'GP-r':'Gaussian Process Regressor',
            'DT-r':'Decision Tree Regressor',
            'ET-r':'Extra Tree Regressor',
            'GB-r':'Gradient Boosting Regressor',
            'Ada-r':'Ada Boost Regressor',
            'RF-r':'Random Forest Regressor',
            'ExT-r':'Extra Trees Regressor',
            'Bag-r':'Bagging Regressor',
            'XGB-r':'XGBoost Regressor'
           }

def cpu_classifiers_list():
    return {'LR-c':'Logistic Regression',
            'SVM-c':'Support Vector Classifier',
            'SGD-c':'Stochastic Gradient Descent Classifier',
            'Ridge-c':'Ridge Classifier',
            'KNN-c':'KNeighbors Classifier',
            'GP-c':'Gaussian Process Classifier',
            'DT-c':'Decision Tree Classifier',
            'GB-c':'Gradient Boosting Classifier',
            'Ada-c':'Ada Boost Classifier',
            'RF-c':'Random Forest Classifier',
            'ET-c':'Extra Tree Classifier',
            'ExT-c':'Extra Trees Classifier',
            'Bag-c':'Bagging Classifier',
            'XGB-c':'XGBoost Classifier'
            }

def gpu_regressors_list():
    return {'LR-r':'Linear Regression',
            'SVM-r':'Support Vector Resreggor',
            'SGD-r':'Stochastic Gradient Descent Regressor',
            'Ridge-r':'Ridge Regressor',
            'Lasso-r':'Lasso Regressor',
            'ElasticNet-r':'ElasticNet Regressor',
            'KNN-r':'KNeighbors Regressor',
            'RF-r':'Random Forest Regressor',
            'XGB-r':'XGBoost Regressor'
           }

def gpu_classifiers_list():
    return {'LR-c':'Logistic Regression',
            'SVM-c':'Support Vector Classifier',
            'SGD-c':'Stochastic Gradient Descent Classifier',
            'KNN-c':'KNeighbors Classifier',
            'RF-c':'Random Forest Classifier',
            'XGB-c':'XGBoost Classifier' 
           }

class Imputer:
    def __init__(self,data_frame,model,loss_f=None,fill_method='random',save_history=False,batch_input=False,st=None):
        self.data_frame0 = data_frame.copy(deep=True)
        self.data_frame = data_frame
        self.disna = data_frame.isna()
        
        if type(model) is str:
            self.model_class = get_model(model,gpu=False)
        elif type(model) is dict:
            self.model_class = {col:get_model(m,gpu=False)() for col,m in model.items()}
        else:
            self.model_class = model
            pass #TODO check the needed methods
        
        self.cols = data_frame.columns
        self.ncol = len(self.cols)
        if type(model) is dict:
            self.imp_cols = list(model.keys())
        else:
            self.imp_cols = self.data_frame.columns[self.data_frame.isnull().any()]
#            self.imp_cols = self.cols
        self.imp_ncol = len(self.imp_cols)
        self.save_history = save_history
        self.batch_input = batch_input
        self.history = 0
        self.st = st
        if self.save_history:
            self.history = {i:[] for i in self.cols}
        
        if fill_method=='random':
            self.data_frame[self.imp_cols] = fill_random(self.data_frame[self.imp_cols],self.batch_input)
        elif fill_method=='mean':
            self.data_frame[self.imp_cols] = self.data_frame[self.imp_cols].fillna(dd.mean())
        else:
            assert 0,'fill_mothod is not recognized!'
            
        assert self.data_frame.isna().mean().mean()==0, 'Something went wrong! Data frame containes missing values after filling requested columns!'
        
        if loss_f is None:
            from sklearn.metrics import mean_squared_error
            self.loss_f = mean_squared_error

        self.loss_frame = pd.DataFrame(columns=self.imp_cols)
        if type(self.model_class) is dict:
            self.models = self.model_class
        else:
            self.models = {col:None for col in self.imp_cols}

    def manual_model_init(self,**kargs):
        if self.models[col] is None:
            model = self.model_class(**kargs)
            self.models[col] = model

    def explore(self,n_try=5):
        df = self.data_frame0.copy(deep=True)
        self.models = explore(df,device='cpu',n_try=n_try,st=self.st)

    def impute(self,n_it,inds=None,normalize=True,trsh=-np.inf,**kargs):
        if inds is None:
            inds = np.arange(self.imp_ncol)
            np.random.shuffle(inds)
    
        ilf = self.loss_frame.shape[0]
        
        if normalize:
            normin,normax = get_range(self.data_frame)
            self.data_frame = set_range(self.data_frame,normin,normax)            
        
        nprog = n_it*len(inds)
        pbar = tqdm(total=nprog, position=0, leave=True)
        if self.st:
            progress_bar = self.st.progress(0)
            status_text = self.st.empty()
            iprog = 0
            progress_bar.progress(iprog)
            status_text.text(f'Imputation... {iprog:4.2f}% complete.')
        
        for i in range(n_it):
            for j in range(len(inds)):
                pbar.update(1)
                if self.st:
                    iprog = iprog+1
                    progress_bar.progress(iprog/nprog)
                    status_text.text(f'Imputation... {100*iprog/nprog:4.2f}% complete.')

                col = self.imp_cols[inds[j]]
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
                if self.batch_input and len(y_test)==0:
                    clses.append(0)
                    continue
                if len(y_test)==0: continue

                if self.models[col] is None:
                    model = self.model_class()
                    self.models[col] = model
                else:
                    model = self.models[col]

                model.fit(x_train,y_train,**kargs)
                pred = model.predict(x_test)
                if pred.ndim>1:
                    pred = pred[:,0]

                self.loss_frame.loc[ilf+i,col] = self.loss_f(y_test,pred)
                self.data_frame.loc[fisna,col] = pred
                if self.save_history:
                    self.history[col].append(pred)
        
        if normalize: self.data_frame = reset_range(self.data_frame,normin,normax)
        pbar.close()
        if self.st:
            progress_bar.progress(100)
            progress_bar.empty()
            status_text.empty()

    def plot_loss_frame(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(15,7))
        for lcol in self.loss_frame.columns:
            ax.plot(self.loss_frame[lcol])
        ax.set_xlim(0,self.loss_frame.shape[0]-1)
        ax.set_xlabel('iteration #', fontsize=20)
        ax.set_ylabel('loss function', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        # ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set_yscale('log')
        plt.tight_layout()
        return ax

    def compare(self,truth,com_f=None,ax=None,width=0.35,legend=True,with_random=True,save=None):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(30,10))
        lsses = []
        lsses_r = []
        for col in self.imp_cols:
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
        x = np.arange(self.imp_ncol)  # the label locations
        if with_random:
            ax.bar(x-width/2,lsses, width,color='b',label='imputed')
            ax.bar(x+width/2,lsses_r, width,color='r',label='radnom')
        else:
            ax.bar(x,lsses, width,color='b',label='imputed')
        if legend: ax.legend()
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('comparison function', fontsize=20)
        ax.set_xticks(np.arange(self.imp_ncol))
        ax.set_xticklabels(self.imp_cols,rotation=90)
        ax.set_xlim(-1,self.imp_ncol)
        # ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        # ax.legend(fontsize=20)
        plt.tight_layout()
        
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

    def dist(self,col,truth=None,cl=25,bandwidth=None,ax=None,alpha=0.3,legend=True,save=None):
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
        
        return fig,ax,(m1,l1,u1),(m2,l2,u2)

    def dist_all(self,truth=None,cl=25,bandwidth=None,save=None):
        
        nn = self.imp_ncol
        ny = int(np.sqrt(nn))
        nx = (nn//ny)
        if nn/ny!=nn//ny:
            nx = nx+1
        
        fig,axs = plt.subplots(ny,nx,figsize=(nx*2.2,ny*2.2), dpi=100)
        ii = 0
        for column in range(nx):
            for row in range(ny):
                ax = axs[row,column]
                if ii==self.imp_ncol: 
                    ax.axis("off")
                    continue
                col = self.imp_cols[ii]
                self.dist(col,truth,cl=cl,bandwidth=bandwidth,ax=ax,legend=0)
                ax.set_title(col)
                ii = ii+1
        plt.tight_layout()
        if save is not None:
            plt.savefig(save+'.jpg',dpi=200)
        return fig, axs
        
    def metric_opt(self,metric_f,truth):
        lsses = []
        for col in self.imp_cols:
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
        for col in self.imp_cols:
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
            for col in self.imp_cols: self.models[col] = self.model_class()
        else:
            self.models[col] = self.model_class()

try:
    import cudf
    import cupy as cp
    import tensorflow as tf

    class GImputer(Imputer):
        def __init__(self,data_frame,model,loss_f=None,fill_method='random',save_history=False,batch_input=False,st=None):
            assert tf.test.is_built_with_cuda(),'No installed GPU is found!'
            print('Available physical devices are: ',tf.config.list_physical_devices())

            self.data_frame = data_frame
            self.data_frame = self.data_frame.apply(lambda x: x.astype(np.float32))
            self.disna = self.data_frame.isna()
            self.disna = cudf.from_pandas(self.disna)
            self.data_frame = cudf.from_pandas(self.data_frame)

            if type(model) is str:
                self.model_class = get_model(model,gpu=True)
            elif type(model) is dict:
                self.model_class = {col:get_model(m,gpu=True)() for col,m in model.items()}
            else:
                self.model_class = model
                pass #TODO check the needed methods

            self.cols = list(data_frame.columns)
            self.ncol = len(self.cols)
            if type(model) is dict:
                self.imp_cols = list(model.keys())
            else:
                self.imp_cols = data_frame.columns[data_frame.isnull().any()]
#                self.imp_cols = self.cols
            self.imp_ncol = len(self.imp_cols)
            self.save_history = save_history
            self.batch_input = batch_input
            self.history = 0
            self.st = st
            if self.save_history:
                self.history = {i:[] for i in self.imp_cols}

            if fill_method=='random':
                self.data_frame[self.imp_cols] = gfill_random(self.data_frame[self.imp_cols],self.batch_input)
            elif fill_method=='mean':
                self.data_frame[self.imp_cols] = self.data_frame[self.imp_cols].fillna(dd.mean())
            else:
                assert 0,'fill_mothod is not recognized!'

            assert self.data_frame.isna().mean().mean()==0, 'Something went wrong! Data frame containes missing values after filling requested columns!'
            if loss_f is None:
                from cuml.metrics import mean_squared_error
                self.loss_f = mean_squared_error

            self.loss_frame = cudf.DataFrame(columns=self.imp_cols)
            if type(self.model_class) is dict:
                self.models = self.model_class
            else:
                self.models = {col:None for col in self.imp_cols}

        def impute(self,n_it,inds=None,trsh=-np.inf,**kargs):
            if inds is None:
                inds = np.arange(self.imp_ncol)
                np.random.shuffle(inds)

            self.to_gpu()

            ilf = self.loss_frame.shape[0]

            nprog = n_it*len(inds)
            pbar = tqdm(total=nprog, position=0, leave=True)
            if self.st:
                progress_bar = self.st.progress(0)
                status_text = self.st.empty()
                iprog = 0
                progress_bar.progress(iprog)
                status_text.text(f'Imputation... {iprog:4.2f}% complete.')        

            for i in range(n_it):
                clses = []
                for j in range(len(inds)):
                    pbar.update(1)
                    if self.st:
                        iprog = iprog+1
                        progress_bar.progress(iprog/nprog)
                        status_text.text(f'Imputation... {100*iprog/nprog:4.2f}% complete.')

                    col = self.imp_cols[inds[j]]
                    fisna = self.disna[col]
                    if fisna.mean()==0:
    #                     self.loss_frame.loc[ilf+i,col] = 0
                        newrow = cudf.DataFrame(index=[ilf+i],columns=[col],data=[0])
                        self.loss_frame = self.loss_frame.append(newrow)

                        continue

                    if i>2:
                        c_loss = self.loss_frame.iloc[-1][col]
                        if c_loss<trsh:
    #                         self.loss_frame.loc[ilf+i,col] = c_loss
                            newrow = cudf.DataFrame(index=[ilf+i],columns=[col],data=[c_loss])
                            self.loss_frame = self.loss_frame.append(newrow)
                            continue

                    x = self.data_frame.drop(columns=[col])
                    y = self.data_frame[col]

                    x_train = x[~fisna]
                    y_train = y[~fisna]
                    x_test = x[fisna]
                    y_test = y[fisna]
                    if self.batch_input and len(y_test)==0:
                        clses.append(0)
                        continue
                    if len(y_test)==0: continue
                        
                    if self.models[col] is None:
                        model = self.model_class()
                        self.models[col] = model
                    else:
                        model = self.models[col]

                    model.fit(x_train,y_train,**kargs)
                    pred = model.predict(x_test)
                    if pred.ndim>1:
                        pred = pred[:,0]

                    c_loss = self.loss_f(y_test,pred)
                    clses.append(c_loss)
                    if type(pred) is cudf.Series:
                        pred = pred.to_array()
                    self.data_frame.loc[fisna,col] = pred
                    nn = self.data_frame[col].isna().sum()
                    assert nn==0,'{}   {}   {}   {}'.format(col,fisna.sum(),pred.shape,nn)

                    if self.save_history:
                        self.history[col].append(pred)

    #             self.loss_frame.loc[ilf+i,col] = self.loss_f(y_test,pred)
                clses = cp.stack(clses)
                clses = clses.reshape(1,-1)
                newrow = cudf.DataFrame(index=[ilf+i],columns=self.imp_cols,data=clses) #self.cols[inds]
                self.loss_frame = self.loss_frame.append(newrow)
            pbar.close()
            if self.st:
                progress_bar.progress(100)
                progress_bar.empty()
                status_text.empty()
            self.to_cpu()

        def to_cpu(self):
            if type(self.loss_frame) is cudf.DataFrame:
                self.loss_frame = self.loss_frame.to_pandas()
            if type(self.data_frame) is cudf.DataFrame:
                self.data_frame = self.data_frame.to_pandas()
            if type(self.disna) is cudf.DataFrame:
                self.disna = self.disna.to_pandas()

        def to_gpu(self):
            if type(self.loss_frame) is pd.DataFrame:
                self.loss_frame = cudf.from_pandas(self.loss_frame)
            if type(self.data_frame) is pd.DataFrame:
                self.data_frame = cudf.from_pandas(self.data_frame)
            if type(self.disna) is pd.DataFrame:
                self.disna = cudf.from_pandas(self.disna)     
except:
    print('GPU functionality is not available!')

def error_rate(x1,x2,eps=None):
    if not eps: eps = np.exp(-100*np.abs(x1))
    err = 100*np.abs(x1-x2)/(x1+eps)
    return np.mean(err)

def explore(df,device='cpu',n_try=5,st=None):

    if device=='cpu':
        modelset = np.union1d(
                    [i.replace('-r','') for i in cpu_regressors_list().keys()],
                    [i.replace('-c','') for i in cpu_classifiers_list()]
                    )
    elif device=='gpu':
        modelset = np.union1d(
                    [i.replace('-r','') for i in gpu_regressors_list().keys()],
                    [i.replace('-c','') for i in gpu_classifiers_list()]
                    )
    else:
        assert 0,'Device is not recognized!'

    isreg = df.nunique()>10
    nd = df.shape[0]
    cols = list(df.columns)
    assert nd>50 , 'The data is too small!'
    nsample = min(np.clip(nd/10,50,500).astype(int),nd)
    nho = nsample//10

    n_iterate = 10
    n_try = 5

    dfcomp = pd.DataFrame(columns=['model','try']+cols)
    idf = 0

    if st:
        progress_bar = st.progress(0)
        status_text = st.empty()
        iprog = 0
        nprog = n_try*len(modelset)
        progress_bar.progress(iprog)
        status_text.text(f'Finding the best models... {iprog:4.2f}% complete.')
        
    for i_try in range(n_try):

        dfs = df.sample(nsample)
        normin,normax = get_range(df)
        masked_ho,hold_outs = do_holdout(dfs,nho)
        dfs = set_range(dfs,normin,normax)
        missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
        for mdl in modelset:
            if st:
                iprog = iprog+1
                progress_bar.progress(iprog/nprog)
                status_text.text(f'Finding the best models... {100*iprog/nprog:4.2f}% complete.')
            masked_hop = masked_ho.copy(deep=True)
            models = {}
            for col in missing_columns:
                if isreg.loc[col]:
                    ext = '-r'
                else:
                    ext = '-c'
                models[col] = mdl+ext
            if device=='cpu':
                imp = Imputer(masked_hop,models,loss_f=None,fill_method='random',save_history=True)
                imp.impute(n_iterate,inds=None)
            else:
                imp = GImputer(masked_hop,models,loss_f=None,fill_method='random',save_history=True)
                imp.impute(n_iterate,inds=None)
            # try:
            #     imp = Imputer(masked_hop,models,loss_f=None,fill_method='random',save_history=True,st=st)
            #     imp.impute(n_iterate,inds=None)
            # except Exception as e:
            #     print(f'Something went wrong with {mdl}. {e}')

            if device=='gpu':
                imp.to_cpu()
            imputed = imp.data_frame

            # dfs = reset_range(dfs,normin,normax)
            imputed = reset_range(imputed,normin,normax)

            res = compare_holdout(imputed,hold_outs,error_rate)
            dfcomp.loc[idf,'model'] = mdl+ext
            dfcomp.loc[idf,'try'] = i_try
            for col in missing_columns:
                dfcomp.loc[idf,col] = res[col]
            idf = idf+1

#        print(dfcomp.loc[idf])
    print(dfcomp.set_index(['model','try']).mean(level=0).apply(pd.to_numeric, errors='ignore').dtypes)
    best_models = dfcomp.set_index(['model','try']).mean(level=0).apply(pd.to_numeric, errors='ignore').idxmin().to_dict()
    if st: 
        progress_bar.progress(100)
        progress_bar.empty()
        status_text.empty()
    return best_models

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
          bandwidth = None,
          ax = None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,5))
    x = x.reshape(-1, 1)
    xmin,xmax = x.min(),x.max()
    xx = np.linspace(xmin,xmax, 100)[:, None]
    if bandwidth is None:
        bandwidth = 0.02*(xmax-xmin)
    
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

def do_holdout(df0,n_hold):
    df = df0+0
    cols = df.columns[df.isnull().any()]
    hold_outs = {}
    for col in cols:
        filt = ~df[col].isna()
        samples = filt[filt].sample(n_hold)
        inds = list(samples.index)
        hold_outs[col] = df.loc[inds,col]
        df.loc[inds,col] = np.nan
    return df,hold_outs

def compare_holdout(df,hold_outs,loss_func):
    cols = list(hold_outs.keys())
    error = {}
    for col in cols:
        inds = hold_outs[col].index
        vals1 = df.loc[inds,col].values
        vals2 = hold_outs[col].values
        error[col] = loss_func(vals1,vals2)
    return error

def compare_using_holdout(df1,df2,hold_outs,loss_func):
    cols = list(hold_outs.keys())
    error = {}
    for col in cols:
        inds = hold_outs[col].index
        vals1 = df1.loc[inds,col].values #hold_outs[col].values
        vals2 = df2.loc[inds,col].values
        error[col] = loss_func(vals1,vals2)
    return error

def undo_holdout(df,hold_outs):
    cols = list(hold_outs.keys())
    for col in cols:
        inds = hold_outs[col].index
        vals = hold_outs[col].values
        df.loc[inds,col] = vals
    return df

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



