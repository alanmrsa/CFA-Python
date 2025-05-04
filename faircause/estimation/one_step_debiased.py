import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from typing import Dict, List, Union, Literal, Optional


def preproc_data(data, X, Z, W, Y):
    data = data.copy()
    
    SFM = {'X': X, 'Z': Z or [], 'W': W or [], 'Y': Y}
    
    for cmp in ['X', 'Z', 'W', 'Y']:
        vars_to_process = []
        
        # Handle case when component is a string (X and Y)
        if isinstance(SFM[cmp], str) and SFM[cmp]:
            vars_to_process = [SFM[cmp]]
        # Handle case when component is a list (Z and W)
        elif isinstance(SFM[cmp], list):
            vars_to_process = SFM[cmp]
        
        for var in vars_to_process:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(data[var]):
                continue
                
            # Handle logical values
            if pd.api.types.is_bool_dtype(data[var]):
                data[var] = data[var].astype(int)
                continue
                
            # Convert strings to categorical
            if pd.api.types.is_string_dtype(data[var]):
                data[var] = data[var].astype('category')
                
            # Handle categorical variables
            if pd.api.types.is_categorical_dtype(data[var]):
                # Binary categorical
                if len(data[var].cat.categories) == 2:
                    data[var] = (data[var] == data[var].cat.categories[0]).astype(int)
                else:
                    # Multi-level categorical - one-hot encode
                    encoder = OneHotEncoder(drop='first', sparse_output=False)
                    enc_mat = encoder.fit_transform(data[[var]])
                    
                    # Create column names
                    col_names = [f"{var}{i+1}" for i in range(enc_mat.shape[1])]
                    
                    # Create a DataFrame with encoded columns
                    enc_df = pd.DataFrame(enc_mat, columns=col_names, index=data.index)
                    
                    # Remove original variable from SFM
                    if cmp == 'X' or cmp == 'Y':
                        SFM[cmp] = col_names[0]  
                    else:
                        SFM[cmp] = [c for c in SFM[cmp] if c != var] + col_names
                    
                    # Drop original column and add encoded columns
                    data = data.drop(var, axis=1)
                    data = pd.concat([data, enc_df], axis=1)
    
    return (data, SFM)


def cv_xgb(df, y, weights=None, **kwargs):
    if isinstance(df, pd.DataFrame):
        df = df.values
    
    y = np.asarray(y)
    is_binary = (len(np.unique(y)) == 2)
    
    params = {
        'objective': 'binary' if is_binary else 'regression',
        'metric': 'binary_logloss' if is_binary else 'rmse',
        'verbose': -1,
        'force_col_wise': True
    }
    
    params.update(kwargs)
    
    train_data = lgb.Dataset(df, label=y, weight=weights)
    
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000,
        nfold=5,
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        return_cvbooster=True, 
        stratified=False
    )

    best_iteration = len(cv_results['valid rmse-mean']) if not is_binary else len(cv_results['valid binary_logloss-mean'])

    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_iteration
    )
    
    return {'model': model, 'is_binary': is_binary}
    
def pred_xgb(xgb_model, df_test, intervention=None, X="X"):
    # Handle dict wrapper from cv_xgb
    if isinstance(xgb_model, dict) and 'model' in xgb_model:
        model = xgb_model['model']
    else:
        model = xgb_model
    
    # Create a copy to avoid modifying the original
    if isinstance(df_test, pd.DataFrame):
        df_test = df_test.copy()
        
        # Set intervention if specified
        if intervention is not None:
            df_test[X] = intervention
            
        df_test = df_test.values
    else:
        raise ValueError("Intervention not supported for numpy arrays")

    return model.predict(df_test)

def measure_spec(spec=None):
    if spec is None:
        spec = ['xspec']
    
    meas = {}
    
    # Include exposure-specific effects
    if 'xspec' in spec:
        xspec = {
            'tv': {
                'sgn': [1, -1],
                'spc': [[1, 1, 1], [0, 0, 0]],
                'ia': "tv"
            },
            'ctfde': {
                'sgn': [1, -1],
                'spc': [[0, 0, 1], [0, 0, 0]],
                'ia': "ctfde"
            },
            'ctfie': {
                'sgn': [1, -1],
                'spc': [[0, 0, 1], [0, 1, 1]],
                'ia': "ctfie"
            },
            'ctfse': {
                'sgn': [1, -1],
                'spc': [[0, 1, 1], [1, 1, 1]],
                'ia': "ctfse"
            },
            'ett': {
                'sgn': [1, -1],
                'spc': [[0, 1, 1], [0, 0, 0]],
                'ia': "ett"
            }
        }
        meas.update(xspec)
    
    return meas

def cross_fit(data, X, Z, W, Y, nested_mean, log_risk, K=10, **kwargs):
    """
    Perform cross-fitting to estimate various conditional expectations using sklearn's KFold.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    X : str
        Protected attribute name
    Z : list
        List of mediator names
    W : list
        List of confounder names
    Y : str
        Outcome variable name
    nested_mean : str
        Method for nested mean estimation: 'refit' or 'wregr'
    log_risk : bool
        Whether to use log-risk scale
    **kwargs :
        Additional parameters for XGBoost
    
    Returns:
    --------
    dict
        Dictionary with cross-fitted estimates
    """
    # Special case: both Z and W are empty
    if (not Z or len(Z) == 0) and (not W or len(W) == 0):
        px = np.repeat(data[X].mean(), len(data))
        return {
            'px_z': [1 - px, px],
            'px_zw': [1 - px, px],
            'y_xzw': [None, None],
            'y_xz': [None, None],
            'ey_nest': [None, None]
        }
    
    n = len(data)
    
    # Initialize arrays to be filled
    y_xzw = [np.full(n, np.nan), np.full(n, np.nan)]
    y_xz = [np.full(n, np.nan), np.full(n, np.nan)]
    px_z = [np.full(n, np.nan), np.full(n, np.nan)]
    px_zw = [np.full(n, np.nan), np.full(n, np.nan)]
    ey_nest = [np.full(n, np.nan), np.full(n, np.nan)]
    
    # Extract x and y data
    y = data[Y].values
    x = data[X].values
    
    # Create KFold object for splitting data
    kf = KFold(n_splits=K, shuffle=True)
    
    # Create a mapping from indices to fold numbers
    fold_indices = np.zeros(n, dtype=int)
    for i, (_, test_idx) in enumerate(kf.split(np.arange(n))):
        fold_indices[test_idx] = i + 1
    
    # Cross-fit across all folds
    for i in range(1, K+1):
        # Split into test, development, and validation sets
        tst_idx = (fold_indices == i)
        
        # Create development and validation sets from remaining folds
        remaining_folds = [j for j in range(1, K+1) if j != i]
        np.random.shuffle(remaining_folds)
        
        dev_folds = remaining_folds[:6]  # Use 60% of remaining folds for development
        val_folds = remaining_folds[6:]  # Use 40% of remaining folds for validation
        
        dev_idx = np.isin(fold_indices, dev_folds)
        val_idx = np.isin(fold_indices, val_folds)
        
        # Prepare datasets
        Z_cols = Z if Z else []
        W_cols = W if W else []
        
        # Train models if Z is not empty
        if Z and len(Z) > 0:
            # Model for P(X|Z)
            mod_x_z = cv_xgb(data.loc[dev_idx, Z], data.loc[dev_idx, X], **kwargs)
            
            # Model for E[Y|X,Z]
            mod_y_xz = cv_xgb(data.loc[dev_idx, [X] + Z], data.loc[dev_idx, Y], **kwargs)
        
        # Train models if W is not empty
        if W and len(W) > 0:
            # Model for P(X|Z,W)
            mod_x_zw = cv_xgb(data.loc[dev_idx, Z_cols + W_cols], data.loc[dev_idx, X], **kwargs)
            
            # Model for E[Y|X,Z,W]
            mod_y_xzw = cv_xgb(data.loc[dev_idx, [X] + Z_cols + W_cols], data.loc[dev_idx, Y], **kwargs)
        else:
            # Inherit from Z if W is empty
            mod_x_zw = mod_x_z if 'mod_x_z' in locals() else None
            mod_y_xzw = mod_y_xz if 'mod_y_xz' in locals() else None
        
        # Get predictions on validation set (needed for nested means)
        if W and len(W) > 0:
            px_zw_val = pred_xgb(mod_x_zw, data.loc[val_idx, Z_cols + W_cols])
            px_zw_val = [1 - px_zw_val, px_zw_val]
        
        if Z and len(Z) > 0:
            px_z_val = pred_xgb(mod_x_z, data.loc[val_idx, Z])
            px_z_val = [1 - px_z_val, px_z_val]
        
        # Get E[Y|X=0,Z,W] and E[Y|X=1,Z,W] on validation set
        y_xzw_val = [
            pred_xgb(mod_y_xzw, data.loc[val_idx, [X] + Z_cols + W_cols], intervention=0, X=X),
            pred_xgb(mod_y_xzw, data.loc[val_idx, [X] + Z_cols + W_cols], intervention=1, X=X)
        ]
        
        # Get predictions on test set
        px_zw_tst = pred_xgb(mod_x_zw, data.loc[tst_idx, Z_cols + W_cols])
        px_zw[0][tst_idx] = 1 - px_zw_tst
        px_zw[1][tst_idx] = px_zw_tst
    
        if Z and len(Z) > 0:
            px_z_tst = pred_xgb(mod_x_z, data.loc[tst_idx, Z])
            px_z[0][tst_idx] = 1 - px_z_tst
            px_z[1][tst_idx] = px_z_tst
        else:
            px_z[0][tst_idx] = 1 - x.mean()
            px_z[1][tst_idx] = x.mean()
        
        # Get E[Y|X=0,Z,W] and E[Y|X=1,Z,W] on test set
        y_xzw[0][tst_idx] = pred_xgb(mod_y_xzw, data.loc[tst_idx, [X] + Z_cols + W_cols], intervention=0, X=X)
        y_xzw[1][tst_idx] = pred_xgb(mod_y_xzw, data.loc[tst_idx, [X] + Z_cols + W_cols], intervention=1, X=X)
        
        # Get E[Y|X=0,Z] and E[Y|X=1,Z] on test set
        if Z and len(Z) > 0:
            features = [X] + Z
            y_xz[0][tst_idx] = pred_xgb(mod_y_xz, data.loc[tst_idx, features], intervention=0, X=X)
            y_xz[1][tst_idx] = pred_xgb(mod_y_xz, data.loc[tst_idx, features], intervention=1, X=X)
        
        # Nested means are not needed if either Z or W are empty
        if (not Z or len(Z) == 0) or (not W or len(W) == 0):
            continue
        

        # Compute nested means
        for xw in [0, 1]:
            xy = 1 - xw
            
            if nested_mean == "wregr":
                if log_risk:
                    raise ValueError("Weighted regression not available for log-risk scale.")
                
                # Compute weights
                weights = np.where(
                    x[val_idx] == xy,
                    px_z_val[xy][val_idx] / px_z_val[xw][val_idx] * 
                    px_zw_val[xw][val_idx] / px_zw_val[xy][val_idx],
                    px_z_val[xy][val_idx] / px_z_val[xw][val_idx]
                )
                
                # Train nested mean model
                mod_nested = cv_xgb(data.loc[val_idx, Z], data.loc[val_idx, Y], weights=weights, **kwargs)
                ey_nest[xy][tst_idx] = pred_xgb(mod_nested, data.loc[tst_idx, Z])
                
            elif nested_mean == "refit":
                # Get counterfactual predictions
                features = [X] + Z_cols + W_cols
                y_tilde = pred_xgb(mod_y_xzw, data.loc[val_idx, features], intervention=xy, X=X)
                
                if log_risk:
                    y_tilde = np.log(y_tilde)
                
                # Train nested mean model
                mod_nested = cv_xgb(data.loc[val_idx, [X] + Z], y_tilde, **kwargs)
                
                # Get nested mean predictions on test set
                ey_nest[xy][tst_idx] = pred_xgb(mod_nested, data.loc[tst_idx, [X] + Z], intervention=xw, X=X)
    
    return {
        'y_xzw': y_xzw,
        'y_xz': y_xz,
        'px_z': px_z,
        'px_zw': px_zw,
        'ey_nest': ey_nest
    }

def pso_diff(cfit, data, X, Z, W, Y, **kwargs):
    """
    Compute pseudo-outcomes for difference scale.
    
    Parameters:
    -----------
    cfit : dict
        Cross-fitting results
    data : pandas.DataFrame
        Input data
    X : str
        Protected attribute name
    Z : list
        List of mediator names
    W : list
        List of confounder names
    Y : str
        Outcome variable name
    **kwargs : 
        Additional parameters
    
    Returns:
    --------
    pso : list
        Nested list of pseudo-outcomes
    """
    n = len(data)
    
    # Unpack cross-fitting results
    y_xzw = cfit['y_xzw']
    y_xz = cfit['y_xz']
    px_z = cfit['px_z']
    px_zw = cfit['px_zw']
    ey_nest = cfit['ey_nest']
    
    # Get data
    y = data[Y].values
    x = data[X].values
    
    # Initialize pseudo-outcome structure (nested list)
    pso = [[[[None, None], [None, None]], [[None, None], [None, None]]], 
           [[[None, None], [None, None]], [[None, None], [None, None]]]]
    
    # Fill with empty arrays
    for xz in [0, 1]:
        for xw in [0, 1]:
            for xy in [0, 1]:
                pso[xz][xw][xy] = np.full(n, np.nan)
    
    # Get E[Y | x] potential outcomes
    for xz in [0, 1]:
        pso[xz][xz][xz] = (x == xz) / np.mean(x == xz) * y
    
    # Case I: Z, W empty
    if (not Z or len(Z) == 0) and (not W or len(W) == 0):
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    pso[xz][xw][xy] = (x == xy) / np.mean(x == xy) * y
    
    # Case II: Z empty, W non-empty
    elif not Z or len(Z) == 0:
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    if xz == xw and xz == xy:
                        continue
                    
                    pso[xz][xw][xy] = (
                        (x == xy) / np.mean(x == xw) * px_zw[xw] / px_zw[xw] *
                        (y - y_xzw[xy]) +
                        (x == xw) / np.mean(x == xw) * y_xzw[xy]
                    )
    
    # Case III: W empty, Z non-empty
    elif not W or len(W) == 0:
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    if xz == xw and xz == xy:
                        continue
                    
                    pso[xz][xw][xy] = (
                        (x == xy) / np.mean(x == xz) * px_zw[xz] / px_zw[xz] *
                        (y - y_xzw[xy]) +
                        (x == xz) / np.mean(x == xz) * y_xzw[xy]
                    )
    
    # Case IV: Z, W non-empty
    else:
        # Special case for TE-like outcomes
        for xz in [0, 1]:
            xw = xy = 1 - xz
            pso[xz][xw][xy] = (
                # Term I
                (x == xw) / np.mean(x == xz) * px_z[xz] / px_z[xw] *
                (y - y_xz[xw]) +
                # Term II
                (x == xz) / np.mean(x == xz) * y_xz[xw]
            )
        
        # General case
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    if xy == xw:
                        continue
                    
                    pso[xz][xw][xy] = (
                        # Term T1
                        (x == xy) * (y - y_xzw[xy]) * px_zw[xw] / px_zw[xy] *
                        px_z[xz] / px_z[xw] * 1 / np.mean(x == xz) +
                        # Term T2
                        (x == xw) / np.mean(x == xz) * px_z[xz] / px_z[xw] *
                        (y_xzw[xy] - ey_nest[xy]) +
                        # Term T3
                        (x == xz) / np.mean(x == xz) * ey_nest[xy]
                    )
    
    return pso

def pso_logr(cfit, data, X, Z, W, Y, **kwargs):
    """
    Compute pseudo-outcomes for log-risk scale.
    
    Parameters:
    -----------
    cfit : dict
        Cross-fitting results
    data : pandas.DataFrame
        Input data
    X : str
        Protected attribute name
    Z : list
        List of mediator names
    W : list
        List of confounder names
    Y : str
        Outcome variable name
    **kwargs : 
        Additional parameters
    
    Returns:
    --------
    pso : list
        Nested list of pseudo-outcomes
    """
    n = len(data)
    
    # Unpack cross-fitting results
    y_xzw = cfit['y_xzw']
    y_xz = cfit.get('y_xz', [None, None])  # May be None if Z is empty
    px_z = cfit['px_z']
    px_zw = cfit['px_zw']
    ey_nest = cfit.get('ey_nest', [None, None])  # May be None if Z or W is empty
    
    # Get data
    y = data[Y].values
    x = data[X].values
    
    # Initialize pseudo-outcome structure (nested list)
    pso = [[[[None, None], [None, None]], [[None, None], [None, None]]], 
           [[[None, None], [None, None]], [[None, None], [None, None]]]]
    
    # Fill with empty arrays
    for xz in [0, 1]:
        for xw in [0, 1]:
            for xy in [0, 1]:
                pso[xz][xw][xy] = np.full(n, np.nan)
    
    # Case I: Z, W empty
    if (not Z or len(Z) == 0) and (not W or len(W) == 0):
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    # Compute mean of y where x equals xy
                    mean_y_x = np.mean(y[x == xy])
                    pso[xz][xw][xy] = np.log(mean_y_x) + (x == xy) / np.mean(x == xy) * (y / mean_y_x - 1)
    
    # Case II: Z empty, W non-empty
    elif not Z or len(Z) == 0:
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    # Avoid division by zero - add small constant
                    epsilon = 1e-10
                    pso[xz][xw][xy] = (
                        (x == xy) / np.mean(x == xw) * px_zw[xw] / (px_zw[xw] + epsilon) *
                        (y - y_xzw[xy]) / (y_xzw[xy] + epsilon) +
                        (x == xw) / np.mean(x == xw) * np.log(y_xzw[xy] + epsilon)
                    )
    
    # Case III: W empty, Z non-empty
    elif not W or len(W) == 0:
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    # Avoid division by zero - add small constant
                    epsilon = 1e-10
                    pso[xz][xw][xy] = (
                        (x == xy) / np.mean(x == xz) * px_zw[xz] / (px_zw[xz] + epsilon) *
                        (y - y_xzw[xy]) / (y_xzw[xy] + epsilon) +
                        (x == xz) / np.mean(x == xz) * np.log(y_xzw[xy] + epsilon)
                    )
    
    # Case IV: Z, W non-empty
    else:
        for xz in [0, 1]:
            for xw in [0, 1]:
                for xy in [0, 1]:
                    # Avoid division by zero - add small constant
                    epsilon = 1e-10
                    
                    pso[xz][xw][xy] = (
                        # Term T1
                        (x == xy) / np.mean(x == xz) *
                        (y - y_xzw[xy]) / (y_xzw[xy] + epsilon) *
                        px_zw[xw] / (px_zw[xy] + epsilon) *
                        px_z[xz] / (px_z[xw] + epsilon) +
                        # Term T2
                        (x == xw) / np.mean(x == xz) * px_z[xz] / (px_z[xw] + epsilon) *
                        (np.log(y_xzw[xy] + epsilon) - ey_nest[xy]) +
                        # Term T3
                        (x == xz) / np.mean(x == xz) * ey_nest[xy]
                    )
    
    return pso


def one_step_debias(data: pd.DataFrame, 
                    X: str, 
                    Z: List[str], 
                    W: List[str], 
                    Y: str, 
                    nested_mean: Literal["refit", "wregr"] = "refit",
                    log_risk: bool = False, 
                    eps_trim: float = 0, 
                    **kwargs) -> pd.DataFrame:
    """
    Perform one-step debiasing for causal fairness analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    X : str
        Protected attribute name
    Z : list
        List of mediator names
    W : list
        List of confounder names
    Y : str
        Outcome variable name
    nested_mean : {"refit", "wregr"}
        Method for nested mean estimation
    log_risk : bool
        Whether to use log-risk scale
    eps_trim : float
        Threshold for trimming extreme propensity weights
    **kwargs : 
        Additional parameters for XGBoost
    
    Returns:
    --------
    pandas.DataFrame
        Results with causal fairness measures and standard errors
    """
    # Validate nested_mean argument
    if nested_mean not in ["refit", "wregr"]:
        raise ValueError("nested_mean must be either 'refit' or 'wregr'")
    
    # Perform cross-fitting
    cfit = cross_fit(data, X, Z, W, Y, nested_mean, log_risk, **kwargs)
    
    # Compute pseudo-outcomes based on scale
    if log_risk:
        pso = pso_logr(cfit, data, X, Z, W, Y, **kwargs)
    else:
        pso = pso_diff(cfit, data, X, Z, W, Y, **kwargs)
    
    # Handle extreme propensity weights
    px_z_0 = cfit['px_z'][0]
    px_zw_0 = cfit['px_zw'][0]
    
    extrm_pxz = (px_z_0 < eps_trim) | (1 - px_z_0 < eps_trim)
    extrm_pxzw = (px_zw_0 < eps_trim) | (1 - px_zw_0 < eps_trim)
    extrm_idx = extrm_pxz | extrm_pxzw
    
    # Report if large number of propensity weights below threshold
    if np.mean(extrm_idx) > 0.02:
        print(f"{round(100 * np.mean(extrm_idx), 2)}% of extreme P(x | z) or P(x | z, w) probabilities at threshold = {eps_trim}.\n"
              f"Reported results are for the overlap population. Consider investigating overlap issues.")
    
    # Trim population to exclude extreme weights
    for xz in [0, 1]:
        for xw in [0, 1]:
            for xy in [0, 1]:
                pso[xz][xw][xy][extrm_idx] = np.nan
    
    # Get specification of measures to be reported
    ias = measure_spec()  # Returns default ['xspec'] measures
    
    # Calculate results for each measure
    results = []
    scale = "difference" if not log_risk else "log-risk"
    
    for measure_name, measure_info in ias.items():
        pseudo_out = np.zeros_like(data[Y].values, dtype=float)
        
        for j, (sign, spec) in enumerate(zip(measure_info['sgn'], measure_info['spc'])):
            xz = spec[0]
            xw = spec[1]
            xy = spec[2]
            pseudo_out += sign * pso[xz][xw][xy]
        
        # Calculate mean and standard deviation
        psi_osd = np.nanmean(pseudo_out)
        dev = np.sqrt(np.nanvar(pseudo_out, ddof=1) / np.sum(~np.isnan(pseudo_out)))
        
        results.append({
            'measure': measure_info['ia'],
            'value': psi_osd,
            'sd': dev,
            'scale': scale
        })
    
    # Convert results to DataFrame
    res = pd.DataFrame(results)
    
    # Store propensity weights as attribute (using a dictionary in pandas
    pw = {'px_z': cfit['px_z'], 'px_zw': cfit['px_zw']}
    res.attrs['pw'] = pw
    
    return res