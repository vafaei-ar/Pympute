import os
import time
import base64
import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st
from sklearn.neighbors import KernelDensity
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

import pympute
dirname = os.path.dirname(pympute.__file__)

session_state = st.session_state

st.sidebar.title('Data imputation tool.')
st.sidebar.image(os.path.join(dirname,'media/logo.png'), use_column_width=True)


#mode = st.radio(
#"Please choose the procedure:",
#('Train a model', 'Predict with a model'))


## import libraries
#import tkinter as tk
#from tkinter import filedialog

## Set up tkinter
#root = tk.Tk()
#root.withdraw()

## Make folder picker dialog appear on top of other windows
#root.wm_attributes('-topmost', 1)

## Folder picker button
#st.title('Folder Picker')
#st.write('Please select a folder:')
#clicked = st.button('Folder Picker')
#dirname = None
#if clicked:
#    dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))



#st.subheader('You selected comedy.')
#if st.button('Upload file'):  
#    import easygui
#    print(easygui.fileopenbox())

#    import tkinter as tk
#    from tkinter.filedialog import askopenfilename
#    root = tk.Tk()  
#    root.withdraw()  
#    filename = askopenfilename(parent=root)
#    root.destroy()
#    print(filename)
#    
#    st.write(file_path)


#st.title("Hello world!")


def color_cells(s):
    if np.isnan(s):
        return 'color:{0}; font-weight:bold'.format('green')
    else:
        return ''

def get_table_download_link_csv(df,fname):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="imputed.csv" target="_blank">Download</a>'#.format(fname)
    return href

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


from pympute import *
from pathlib import Path

all_models = list(cpu_regressors_list().keys())+\
             list(cpu_classifiers_list().keys())+\
             list(gpu_regressors_list().keys())+\
             list(gpu_classifiers_list().keys())


uploaded_file = st.file_uploader("Please choose a csv file.")
done = False
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if not hasattr(session_state,'first_time'):
        session_state.cols = df.columns.tolist()
        session_state.models = {i:'LR-r' for i in session_state.cols}
    session_state.first_time = False
    
    chart_data = df.isnull().mean().to_frame('null')
    chart_data['data'] = 1-chart_data['null']
    missed = np.isnan(df)
#        filt = np.mean(missed,axis=1)!=1
    st.bar_chart(chart_data)

#    regn = st.selectbox(
#        'please choose the Ml estimator?',
#        ('None', 'BayesianRidge', 'ExtraTreesRegressor'))
#        
#    max_iter = st.number_input('Choose maximum iteration number', value=10)

    
    
    
    for ikey,col in enumerate(session_state.cols):
#        col1, col2 = st.columns(2)
#        col1.write(f'{col:15.15}:')
        session_state.models[col] = st.selectbox(
            label = col,
            options = all_models,
            index = all_models.index(session_state.models[col]),
            key = f'MyKey{ikey}',
            label_visibility = 'hidden'
        )
        st.markdown('---')

    if st.button('Impute'): 

        if regn=='None':
            pass
        elif regn=='BayesianRidge':
            from sklearn.linear_model import BayesianRidge
            estimator = BayesianRidge()
            imputer = IterativeImputer(random_state=0,
                                       missing_values=np.nan,
                                       estimator=estimator,
                                       max_iter=max_iter,verbose=0)
            
            with st.spinner('Wait for it...'):
                imputed_data = imputer.fit_transform(df)
                done = True
           
    #                y_true = good_data[missed]
    #                y_pred = imputed_data[missed]
            
        elif regn=='ExtraTreesRegressor':
            from sklearn.ensemble import ExtraTreesRegressor
            estimator = ExtraTreesRegressor(n_estimators=10, random_state=0,n_jobs=-1)
            imputer = IterativeImputer(random_state=0,
                                       missing_values=np.nan,
                                       estimator=estimator,
                                       max_iter=max_iter,verbose=0)
            
            with st.spinner('Wait for it...'):
                imputed_data = imputer.fit_transform(df)
                done = True
            
        else:
            print('???')
            
            
        if done:
            dfi = df+0
            dfi.loc[:,:] = imputed_data
            st.success('Done!')
            col1, col2 = st.columns(2)
            col1.write('original data')
            col1.write(df.style.highlight_null(null_color='red'))
            col2.write('imputed data')
            col2.write(dfi.style.apply(lambda x: df.applymap(color_cells), axis=None))
            col2.markdown(get_table_download_link_csv(dfi,'imputed'), unsafe_allow_html=True)

if done:
    col = st.selectbox(
        'You can choose and see the original vs. imputed data points distrinutions:',
         ['Select one column']+list(dfi.columns))


    if col!='Select one column':
        ad = dfi[col].values
        ma = np.isnan(df[col].values)
        gd = ad[~ma]
        im = ad[ma]
        
        if len(im)==0:
            st.write('This column was already complete!')
        else:
            cl = 95
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            m,l,u = mykde(gd, cl=95, color='b', alpha=0.3, kernel='gaussian', bandwidth=0.2, ax=ax)
            aw1 = u-l
            m,l,u = mykde(im, cl=95, color='r', alpha=0.3, kernel='gaussian', bandwidth=0.2, ax=ax)
            aw2 = u-l
            
            ax.set_ylim(bottom=0)
            
            red_patch = mpatches.Patch(color='red', label='imputed data')
            blue_patch = mpatches.Patch(color='blue', label='existing data')

            ax.legend(handles=[red_patch, blue_patch])
            
            st.pyplot(fig)
            
#            st.write('R is {:4.2f}'.format(r2_score())
            st.write('AW1 is {:4.2f} and AW2 is {:4.2f}'.format(aw1,aw2))




        
#        st.image(file_path)


#    train()
#elif mode == 'Predict with a model':
#    st.subheader("You didn't select comedy.")
#    predict()



#import cv2
#from skimage.transform import resize
#from tf_keras_vis.utils import normalize
#from tf_keras_vis.gradcam import GradcamPlusPlus
#from tf_keras_vis.scorecam import ScoreCAM
#from utils import *
#import models


#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)

#for i in range(1, 101):
#    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
##    status_text.text("%i%% Complete" % i)
#    chart.add_rows(new_rows)
##    progress_bar.progress(i)
#    last_rows = new_rows
#    time.sleep(0.05)

##progress_bar.empty()

## Streamlit widgets automatically run the script from top to bottom. Since
## this button is not connected to any other logic, it just causes a plain
## rerun.
#st.button("Rerun!")

#my_bar = st.progress(0)
#for percent_complete in range(100):
#    time.sleep(0.1)
#    my_bar.progress(percent_complete + 1)
#    st.write('{}%'.format((percent_complete+1)/100))



























