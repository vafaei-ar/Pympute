
import numpy as np
from copy import deepcopy


def missing_simulation(data,ref_cols,target_cols,lower,upper,inner,miss1,miss2,miss3):
    masked = deepcopy(data)
    ndata = masked.shape[0]
    ntar = len(target_cols)
    refs = data[ref_cols]
    refs = set_mean_std(refs)
    refs = refs.sum(axis=1)

    lower = np.percentile(refs,lower)
    upper = np.percentile(refs,upper)

    if inner:
        filt = (lower<refs) & (refs<upper)
    else:
        filt = (lower>refs) | (refs>upper)

    # refs.loc[filt].index
    inds = np.argwhere(filt.values)
    inds = inds.reshape(-1)
    ninds = len(inds)
    rind = np.arange(ndata)
    np.random.shuffle(rind)
    n_miss = int(miss1*ninds)
    rind = rind[:n_miss]

    mask = np.random.uniform(0,1,(n_miss,ntar))
    mask = (miss2<mask).astype(float)
    mask[mask==0] = np.nan
    masked.loc[rind,target_cols] *= mask

    mask = np.random.uniform(0,1,masked.shape)
    mask = (miss3<mask).astype(float)
    mask[mask==0] = np.nan
    masked *= mask
    return masked