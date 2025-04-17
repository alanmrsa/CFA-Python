import numpy as np
import pandas as pd

def msd_two(x1, t1, x2, t2, meas, boots): 
    ms = []
    for i, ids in enumerate(boots): 
        mean_x1 = np.nanmean(x1[ids[t1]])
        mean_x2 = np.nanmean(x2[ids[t2]])
        ms.append(mean_x1 + mean_x2)

    result_df = pd.DataFrame({
        'value': ms, 
        'boot': range(1, len(boots) + 1), 
        'measure': meas
    })
    return result_df



def inh_str(x, meas, set0=False, setna=False): 
    x['measure'] = meas
    if set0:
        x['value'] = 0
    if setna:
        x['value'] = float('nan') 
    return x