import numpy as np
import scipy


class TrainObject(object):
    def __init__(self, X, y):
        assert len(X) == len(y)
        mean = np.mean(X,axis=1,keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        X = (X - mean) / std
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)


def get_bnci_data(data_path='bnci_raw_npy', bnci_fs=250.0, resample_fs=125.0):
    X_bnci, y_bnci = [], []

    for subj in list(range(1, 10)):
        X_subj = []
        y_subj = []
        for sess in ['E', 'T']:
            bnci_data = np.load(f'{data_path}/subj_{subj}_session_{sess}.npz')
            X, y = bnci_data['X'], bnci_data['y']

            s = X.shape[2] / bnci_fs
            resample_s = int(s * resample_fs)
            X_resampled = scipy.signal.resample(X.T, resample_s).T

            X_subj.append(X_resampled)
            y_subj.extend(y.tolist())
            
        X_subj = np.vstack(X_subj)
        y_subj = np.array(y_subj)
        
        X_bnci.append(X_subj)
        y_bnci.append(y_subj)
    
    X_bnci = np.array(X_bnci)
    y_bnci = np.array(y_bnci)

    return X_bnci, y_bnci


def get_batches(n_trials, random_state, shuffle, n_batches=None, batch_size=None):
    assert batch_size is not None or n_batches is not None
    
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    
    assert n_batches_with_extra_trial < n_batches
    
    all_inds = np.array(range(n_trials))
    
    if shuffle:
        random_state.shuffle(all_inds)
    
    i_start_trial = 0
    i_stop_trial = 0
    
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    
    assert i_start_trial == n_trials
    return batches
