
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def api(bad_data,core):

    if not type(core) is int:
        imputer = IterativeImputer(random_state=0,
                                   missing_values=np.nan,
                                   estimator=core,
                                   max_iter=50,verbose=1)
        imputed_data = imputer.fit_transform(bad_data)
        return imputed_data
    
    else:
        k = core
    
    if k==0:
        from sklearn.impute import SimpleImputer
        strategy = 'mean'
        si = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputed_data = si.fit_transform(bad_data)
    elif k==1:
        from sklearn.impute import SimpleImputer
        strategy = 'median'
        si = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputed_data = si.fit_transform(bad_data)
    elif k==2:
        from sklearn.linear_model import BayesianRidge
        impute_estimator0 = BayesianRidge()
    elif k==3:
        from sklearn.tree import DecisionTreeRegressor
        impute_estimator0 = DecisionTreeRegressor(max_features='sqrt', random_state=0)
    elif k==4:
        from sklearn.ensemble import RandomForestRegressor
        impute_estimator0 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    elif k==5:
        from sklearn.neural_network import MLPRegressor
        impute_estimator0 = MLPRegressor()
    elif k==6:
        from sklearn.ensemble import ExtraTreesRegressor
        impute_estimator0 = ExtraTreesRegressor(n_estimators=10, random_state=0,n_jobs=-1)
    elif k==7:
        from sklearn.neighbors import KNeighborsRegressor
        impute_estimator0 = KNeighborsRegressor(n_neighbors=15)
    elif k==8:
        from xgboost import XGBModel
        impute_estimator0 = XGBModel(n_jobs=-1)
    else:
        assert 0, 'Unknown core!'

    if k>1:
        impute_estimator = clone(impute_estimator0)
        imputer = IterativeImputer(random_state=0,
                                   missing_values=np.nan,
                                   estimator=impute_estimator,
                                   max_iter=50,verbose=1)
        imputed_data = imputer.fit_transform(bad_data)

#    imputed_data[~missed] = good_data[~missed]
#    y_true = good_data[missed]
#    y_pred = imputed_data[missed]

    return imputed_data





























