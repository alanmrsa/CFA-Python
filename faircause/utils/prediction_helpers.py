import numpy as np
from typing import List, Union, Literal
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

def compute_auc(out, pred):
    """
    Calculate the Area Under the Curve (AUC) from binary outcomes and predicted probabilities.
    """

    out = np.array(out)
    pred = np.array(pred)
    
    # Check basic conditions
    if not all(x in [0, 1] for x in out):
        return 0
    if any(p < 0 or p > 1 for p in pred):
        return 0
    
    # Count positive and negative cases
    n_pos = np.sum(out == 1)
    n_neg = np.sum(out == 0)
    
    # Handle edge cases - need both positive and negative samples for ROC curve
    if n_pos == 0 or n_neg == 0:
        return 0
        
    # Use sklearn's roc_auc_score which is efficient and well-tested
    try:
        return roc_auc_score(out, pred)
    except Exception:
        # Return 0 if calculation fails for any reason
        return 0

def acc_measure(y, p, loss="bce"):
    """
    Calculate various accuracy measures between actual values y and predictions p.
    """
    y = np.array(y)
    p = np.array(p)
    
    if loss == "bce":
        # Binary cross-entropy (limiting p to avoid log(0))
        p_safe = np.clip(p, 1e-15, 1 - 1e-15)
        ret = -np.mean(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))
    elif loss == "acc":
        # Accuracy
        ret = np.mean(np.round(p) == y)
    elif loss == "auc":
        # Area Under the Curve
        ret = compute_auc(y, p)
    elif loss == "mse":
        # Mean Squared Error
        ret = np.mean((np.array(y) - np.array(p)) ** 2)
    else:
        raise ValueError("Invalid loss type")
    
    return ret

def acc_measure_boot(y, p, loss):
    """
    Calculate the standard deviation of an accuracy measure using bootstrapping.
    """
    results = []
    for _ in range(100):
        # Sample with replacement
        boot_idx = np.random.choice(len(y), size=len(y), replace=True)
        y_boot = [y[i] for i in boot_idx]
        p_boot = [p[i] for i in boot_idx]
        results.append(acc_measure(y_boot, p_boot, loss))
    
    return np.std(results)

def lambda_performance(meas, y, p, lmbd):
    """
    Calculate various performance metrics and combine with existing measurements.
    """
    # Filter rows with specified measures
    filtered_meas = meas[meas['measure'].isin(['nde', 'nie', 'expse_x1', 'expse_x0'])]
    
    # Calculate performance metrics
    performance_data = {
        'bce': acc_measure(y, p, "bce"),
        'bce_sd': acc_measure_boot(y, p, "bce"),
        'acc': acc_measure(y, p, "acc"),
        'acc_sd': acc_measure_boot(y, p, "acc"),
        'auc': acc_measure(y, p, "auc"),
        'auc_sd': acc_measure_boot(y, p, "auc"),
        'mse': acc_measure(y, p, "mse"),
        'mse_sd': acc_measure_boot(y, p, "mse"),
        'lmbd': lmbd
    }
    
    # Create a DataFrame with the same number of rows as filtered_meas
    performance_df = pd.DataFrame({k: [v] * len(filtered_meas) for k, v in performance_data.items()})
    
    # Combine the DataFrames
    result = pd.concat([filtered_meas.reset_index(drop=True), 
                        performance_df.reset_index(drop=True)], axis=1)
    
    return result