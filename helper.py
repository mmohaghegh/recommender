import numpy as np

def get_uniques_from_dict(array):
    keys = []
    vals = []
    for _dict in array:
        keys.append(list(_dict.keys()))
        vals.append(list(_dict.values()))
    unique_keys = np.unique(np.concatenate(keys))
    unique_vals = np.unique(np.concatenate(vals))
    return unique_keys, unique_vals

def to_indicator(array, uniques):
    indicator_feat = np.zeros_like(uniques, dtype=bool)
    for arr in array:
        indicator_feat = np.logical_or(uniques==arr, indicator_feat)
    return indicator_feat

def expand_to_list(var, length):
    if not hasattr(var, 'len'):
        return [var]*length
    elif type(var) == str:
        return [var]*length
    else:
        return var