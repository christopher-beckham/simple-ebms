from scipy import stats
import numpy as np

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def rejection_sample_dataset(data, density_thresh, accept_p):
    mvn = stats.multivariate_normal(np.mean(data, axis=0), 1)
    ind = mvn.pdf(data) < density_thresh # 0.1
    for j in range(len(ind)):
        if ind[j] == False and np.random.uniform() < accept_p:
            ind[j] = True
    return ind
