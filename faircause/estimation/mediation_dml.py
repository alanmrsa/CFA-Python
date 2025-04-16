from copy import deepcopy
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ..utils.helpers import msd_two, inh_str


# helpers



def tune_ranger_params(Y, X, regr=False, **kwargs):
    if not regr: 
        mns_grid = [10, 20, 50, 100]
        model_class = RandomForestClassifier
        scoring = 'accuracy'
    else: 
        mns_grid = [5, 10, 25, 50]
        model_class = RandomForestRegressor
        scoring = 'neg_mean_squared_error'
    
    best_score = float('-inf')
    best_model = None
    best_mns = None
    
    # Try each min_samples_leaf value
    for mns in mns_grid:
        model = model_class(min_samples_leaf=mns, 
                            scoring=scoring,
                            oob_score=True,  # Enable out-of-bag scoring
                            **kwargs)
        
        # Fit the model
        model.fit(X, Y)
        
        # Get OOB score (note: higher is better for sklearn's oob_score)
        current_score = model.oob_score_
        
        # Update best model if current is better
        if current_score > best_score:
            best_score = current_score
            best_model = model
            best_mns = mns
    
    return (best_model, best_mns)

def fit_model(Y, X, model='ranger', tune_params = False, mns = None, **kwargs):
    #Z set empty -> x equals Null -> return just mean
    if X is None or (isinstance(X, pd.DataFrame) and X.empty):
      return np.mean(Y)

    if Y.nunique(dropna=True) == 2: 
        #categorical, prob case
        if model == 'ranger':
            if tune_params and mns==None: 
                model = tune_ranger_params(Y, X, regr=False, **kwargs)
            else: 
                model = RandomForestClassifier(min_samples_leaf=mns, **kwargs)
                model.fit(X, Y)
            return model
        elif model == 'linear':
            return sm.Logit(Y, X).fit()
    else: 
        #regression case
        if model == 'ranger':
            if tune_params and mns==None: 
                model = tune_ranger_params(Y, X, regr=True, **kwargs)
            else: 
                model = RandomForestRegressor(min_samples_leaf=mns, **kwargs)
                model.fit(X, Y)
            return model
        elif model == 'linear':
            return sm.OLS(Y, X).fit()

def pred(m, X, model='ranger', regr=False): 
    if pd.api.types.is_numeric_dtype(m): 
        return np.asarray([m]*len(X))
    elif (model=='linear'):
        return m.predict(X)
    elif (model=='ranger'): 
        if not regr:
            probs = m.predict_proba(X)
            # Check if positive class is first or second column
            if m.classes_[1] == 1:
                return probs[:, 1]  # Second column has probability of class 1
            else:
                return probs[:, 0]  # First column has probability of class 1
        else:
            return m.predict(X)
    else: 
        raise TypeError("invalid model")

def doubly_robust_med(X, Y, Z, W, K = 5, model='ranger', tune_params=False, eps_trim = 0.01, params= None, **kwargs):
    if pd.api.types.is_categorical_dtype(X) or pd.api.types.is_object_dtype(X):
        X = pd.get_dummies(X)
    if pd.api.types.is_categorical_dtype(Y) or pd.api.types.is_object_dtype(Y):
        Y = pd.get_dummies(Y)
    folds = KFold(n_splits=K, shuffle=True)
    y0 =[]
    y1=[]
    y0w1= []
    y1w0 = []
    px_z = []
    px_zw = []
    all_idxs = np.arange(len(X))
    regr = (Y.nunique(dropna=True) == 2)
    if params is None:
        params = {
            'mns_pxz': None,
            'mns_yz': None, 
            'mns_pxzw': None,
            'mns_yzw': None,
            'mns_eyzw': None
        }

    for tr, ts in folds.split(all_idxs):
        #TE

        #regress X on Z
        #def fit_model(Y, X, model='ranger', tune_params = False, mns = None, **kwargs):
        px_z_tr = fit_model(Z.loc[tr], X.loc[tr], model =model, tune_params=tune_params, mns=params['mns_pxz'], **kwargs)
        if isinstance(px_z_tr, tuple): 
            params['mns_pxz'] = px_z_tr[1]
            px_z_tr = px_z_tr[0]
        
        #regress Y on Z for each level
        y_z0_tr = fit_model(Y.loc[tr][X.loc[tr] == 0], Z.loc[tr][X.loc[tr] == 0], model =model, tune_params=tune_params, mns=params['mns_yz'], **kwargs)
        if isinstance(y_z0_tr, tuple): 
            params['mns_yz'] = y_z0_tr[1]
            y_z0_tr = y_z0_tr[0]

        y_z1_tr = fit_model(Y.loc[tr][X.loc[tr] == 1], Z.loc[tr][X.loc[tr] == 1], model =model, tune_params=tune_params, mns=params['mns_yz'], **kwargs)
        if isinstance(y_z1_tr, tuple): 
            y_z1_tr = y_z1_tr[0]

        px_z_ts = pred(px_z_tr, Z.loc[ts], model=model, regr=regr)
        px_z.append(px_z_ts)

        y_z0_ts = pred(y_z0_tr, Z.loc[ts], model=model, regr=regr)
        y_z1_ts = pred(y_z1_tr, Z.loc[ts], model=model, regr=regr)

        y0.append((Y.loc[ts] - y_z0_ts) * (X.loc[ts] == 0) / (1-px_z_ts) + y_z0_ts)
        y1.append((Y.loc[ts] - y_z1_ts) * (X.loc[ts] == 1) / (px_z_ts) + y_z1_ts)

        #regress X on Z and W

        px_zw_tr = fit_model(X.loc[tr], (np.column_stack((Z, W)))[ts])
        if isinstance(px_zw_tr, tuple): 
            params['mns_pxzw'] = px_zw_tr[1]
            px_zw_tr = px_zw_tr[0]
        
        #predict regression on parget partition
        px_zw_ts = pred(px_zw_tr, np.column_stack((Z, W))[ts], model=model, regr=regr)
        px_zw.append(px_zw_ts)

        #split complement intwo two equal parts
        # part 1 learn mean
        mu_indices = np.random.choice(tr, size=len(tr) // 2, replace=False)
        mu = np.zeros_like(tr, dtype=bool)
        mu[mu_indices] = True
        ns = tr & ~mu

        #regress Y ~Z + W for each level of X
        y_zw0_mu = fit_model(y[mu & (X == 0)], np.column_stack((Z[mu & X == 0], W[mu & X == 0])), model=model, mns=params['mns_yzw'], tune_params=tune_params, **kwargs)
        if isinstance(y_zw0_mu, tuple): 
            params['mns_yzw'] = y_zw0_mu[1]
            y_zw0_mu = y_zw0_mu[0]

        y_zw1_mu = fit_model(y[mu & (X == 1)], np.column_stack((Z[mu & X == 1], W[mu & X == 1])), model=model, mns=params['mns_yzw'], tune_params=tune_params, **kwargs)
        if isinstance(y_zw1_mu, tuple): 
            y_zw1_mu = y_zw1_mu[0]

        #part 2 learn nested mean
        #predict on nested partition using mu
        y_zw0_ns = pred(y_zw0_mu, np.column_stack((Z[ns], W[ns])), model=model, regr=regr)
        y_zw1_ns = pred(y_zw1_mu, np.column_stack((Z[ns], W[ns])), model=model, regr=regr)

        ey_zw1_0_ns = fit_model(y_zw1_ns[X[ns]==0], Z[ns & X == 0], model=model, mns=params['mns_eyzw'], tune_params=tune_params, **kwargs)
        if isinstance(ey_zw1_0_ns, tuple): 
            params['mns_eyzw'] = ey_zw1_0_ns[1]
            ey_zw1_0_ns = ey_zw1_0_ns[0]

        ey_zw0_1_ns = fit_model(y_zw0_ns[X[ns]==1], Z[ns & X == 1], model=model, mns=params['mns_eyzw'], tune_params=tune_params, **kwargs)
        if isinstance(ey_zw0_1_ns, tuple): 
            ey_zw0_1_ns = ey_zw0_1_ns[0]

        #part 3 compute the mean/nested mean on target partition
        y_zw0_ts = pred(y_zw0_mu, np.column_stack((Z.loc[ts], W.loc[ts])), model=model, regr=regr)
        ey_zw0_1_ts = pred(ey_zw0_1_ns, Z[ts], model=model, regr=regr)
        y_zw1_ts = pred(y_zw1_mu, np.column_stack((Z.loc[ts], W.loc[ts])), model=model, regr=regr)
        ey_zw1_0_ts = pred(ey_zw1_0_ns, Z.loc[ts], model=model, regr=regr)

        #part4 compute the formula 
        y0w1.append(
            (px_zw_ts) * (X.loc[ts] == 0) / ((1 - px_zw_ts) * px_z_ts) * (Y.loc[ts] - y_zw0_ts) + (X.loc[ts] == 1) /  (px_z_ts) * (y_zw0_ts - ey_zw0_1_ts) + ey_zw0_1_ts
        )
        y1w0.append(
            (1 - px_zw_ts) * (X.loc[ts] == 1) / ((px_zw_ts) * (1- px_z_ts)) * (Y.loc[ts] - y_zw1_ts) + (X.loc[ts] == 0) / (1 - px_z_ts) * (y_zw1_ts - ey_zw1_0_ts) + ey_zw1_0_ts
        )

    #trim the extreme probs
    y0 = np.asarray(y0)
    y1= np.asarray(y1)
    y0w1= np.asarray(y0w1)
    y1w0 = np.asarray(y1w0)
    px_z = np.asarray(px_z)
    px_zw = np.asarray(px_zw)
    extrm_pxz = (px_z < eps_trim) | (1 - px_z < eps_trim)
    extrm_pxzw = (px_zw < eps_trim) | (1 - px_zw < eps_trim)
    extrm_idx = extrm_pxz | extrm_pxzw
    pw =  {'px_z' : px_z, 'px_zw' : px_zw}
    y0[extrm_idx] = np.nan
    y1[extrm_idx] = np.nan
    y0w1[extrm_idx] = np.nan
    y1w0[extrm_idx] = np.nan

    if (np.mean(extrm_idx) > 0.02): 
        print(round(100 * np.mean(extrm_idx), 2), "\% of extreme P(x|z) or p(x|z,w) prob\n Reported results are for the overlap pop. Consider investigating overlap issues")
    
    return {
        'y0': y0, 
        'y1': y1, 
        'y0w1':y0w1, 
        'y1w0':y1w0, 
        'params':params,
        'pw':pw
    }

def ci_mdml(data, X, Z, W, Y, x0, x1, model, rep, nboot, tune_params, params): 
    if rep > 1: 
        boot_samp = np.random.choice(data.index, size=len(data), replace=True) 
    else: 
        data.index.tolist()
    boot_data = data.loc[boot_samp]
    
    boots = []
    for _ in range(nboot):
        ind = np.random.choice(len(boot_data), size=len(boot_data), replace=True)
        
        idx0 = boot_data.iloc[ind][X] == x0
        
        # Create a dictionary of indices
        boot_dict = {
            'all': ind,  # All sampled indices
            'id0': ind[idx0],  # Indices where condition is True
            'id1': ind[~idx0]  # Indices where condition is False
        }
        
        boots.append(boot_dict)

    y = pd.to_numeric(boot_data[Y])
    tv = msd_two(Y, "id1", -y, "id0", "tv", boots)

    if len([Z, W]) > 0: 
        est_med = doubly_robust_med(
            boot_data[X], 
            boot_data[Z], 
            boot_data[W], 
            boot_data[Y], 
            model=model, 
            tune_params=tune_params, 
            params=params
        )

        params = est_med['params']
        yx0 = est_med['y0']
        yx1 = est_med['y1']
        yx1wx0 = est_med['y1w0']
        if rep == 1: 
            pw = est_med['pw']
        else: 
            None
    else: 
        pw = None
    
    if len(Z) == 0: 
        te = inf_str(tv, "te")
        ett = inh_str(tv, "ett")
        expse_x1 = inh_str(tv, "expse_x1", set0=True)
        expse_x0 = inh_str(tv, "expse_x0", set0=True)
        ctfse = inh_str(tv, "ctfse", set0=True)
    else: 
        te = msd_two(yx1, "all", -yx0, "all", "te", boots)
        ett = msd_two(yx1, "id0", -y, "id0", "ett", boots)
    
        ctfse = msd_two(yx1, "id0", -y, "id1", "ctfse", boots)
        expse_x1 = msd_two(y, "id1", -yx1, "all", "expse_x1", boots)
        expse_x0 = msd_two(y, "id0", -yx0, "all", "expse_x0", boots)

    if len(W) == 0: 
        nde = inh_str(te, "nde")
        ctfde = inh_str(ett, "ctfde")
        ctfie = inh_str(ett, "ctfie", set0=True)
        nie = inh_str(te, "nie", set0=True)

    else: 
        nde = msd_two(yx1wx0, "all", -yx0, "all", "nde", boots)
        ctfde = msd_two(yx1wx0, "id0", -y, "id0", "ctfde", boots)

        nie = msd_two(yx1wx0, "all", -yx1, "all", "nie", boots)
        ctfie = msd_two(yx1wx0, "id0", -yx1, "id0", "ctfie", boots)
    res = pd.concat([tv, te, expse_x1, expse_x0, ett, ctfse, nde, nie, ctfde, ctfie], axis=0)

    res['rep'] = rep

    res.attrs['params'] = params
    res.attrs['pw'] = pw
    return {
    'results': res,
    'params': params,
    'pw': pw
    }