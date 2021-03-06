from __future__ import print_function
from astropy.io import ascii

import numpy as np
import pickle
from sklearn import clone
from itertools import islice
from sklearn.cross_validation import train_test_split


def parse(data_file, feature_file, response_var, parametric_var=None, debug_limit=None):
    with open(feature_file) as f:
        features = f.readlines()
    features = map(str.strip, features)

    data = ascii.read(data_file,
                      fill_values=[("", "-99"), ("null", "-99"), ("--", "-99"), ("999999", "-99")],
                      data_end=debug_limit)

    X = np.array([data[f] for f in features])
    X = np.transpose(X)
    Y = np.array(data[response_var])

    if parametric_var:
        P = np.array(data[parametric_var])
        P = np.transpose(P)
        return X, Y, P
    else:
        return X, Y


def labeled_subset(X, Y, P=None, labeled=True):
    unlabeled_subset = Y == -99
    labeled_subset = np.array([not x for x in unlabeled_subset])
    if labeled:
        subset = labeled_subset
    else:
        subset = unlabeled_subset

    X_out = X[subset]
    Y_out = Y[subset]
    if P:
        P_out = P[subset]
        return X_out, Y_out, P_out
    else:
        return X_out, Y_out


def csv_add(filename, array, add=True):
    if add:
        f = open(filename, 'a')
    else:
        f = open(filename, 'w')

    for j in range(0, len(array) - 1):
        print(array[j], file=f, end=',')
    print(array[-1], file=f)
    f.close()


def subsample(data_list, n, axis=0):
    index_array = np.arange(data_list[0].shape[axis])
    np.random.shuffle(index_array)
    new_list = []
    for X in data_list:
        sample = False
        if len(X.shape) >= axis + 1 and X.shape[axis] > n:
            sample = True
        elif len(X.shape) < axis + 1:
            print('Dim error')

        if sample:
            if len(X.shape) == 2 and axis == 0:
                X = X[index_array[0:n], :]
            if len(X.shape) == 2 and axis == 1:
                X = X[:, index_array[0:n]]
            if len(X.shape) == 1:
                X = X[index_array[0:n]]
        new_list.append(X)
    print('Data Subsampled')
    return new_list


def clf_iterator(X_train, Y_train, X_test, clf_list):
    res_list_test = []
    res_list_train = []
    for clf in clf_list:
        clf.fit(X_train, Y_train)
        res_list_train.append(clf.predict(X_train))
        res_list_test.append(clf.predict(X_test))
    return res_list_train, res_list_test


def best_pipelines_by_algo(grid, algo_tag = 'regressor__algorithm', bigger_is_better=True):
    results = []

    if bigger_is_better:
        best_mean = -np.inf
        def is_better(x, y): return x > y
    else:
        best_mean = np.inf
        def is_better(x, y): return x < y

    best_params_by_algo = {}
    for params, mean, std in grid.grid_scores_:
        # print(params)
        algo_type = type(params[algo_tag])
        if algo_type not in best_params_by_algo:
            best_params_by_algo[algo_type] = (mean, params)
        else:
            best_mean = best_params_by_algo[algo_type][0]
            if is_better(mean, best_mean):
                best_params_by_algo[algo_type] = (mean, params)

    for algo, (mean, params) in best_params_by_algo.iteritems():
        new_estimator = clone(grid.estimator)
        new_estimator.set_params(**params)
        results.append(new_estimator)

    return results


def default_pipelines_by_algo(estimator, meta_parameter_space, needed_params=None):
    results = []

    for param_space in meta_parameter_space:
        algo_dict = {'imputer__base_algorithm': param_space['imputer__base_algorithm'][0],
                     'scaler__base_algorithm': param_space['scaler__base_algorithm'][0],
                     'dim_reducer__base_algorithm': param_space['dim_reducer__base_algorithm'][0],
                     'regressor__base_algorithm': param_space['regressor__base_algorithm'][0]}

        if needed_params:
            algo_dict.update(needed_params)
        if algo_dict not in results:
            new_estimator = clone(estimator)
            new_estimator.set_params(**algo_dict)
            results.append(new_estimator)
    return results

#TODO make this slicker
def append_prefix_to_dict(dictionary, prefix):
    old_keys = []
    new_keys = []
    for k in dictionary:
        old_keys.append(k)
        new_keys.append(prefix + k)
    for old_key, new_key in zip(old_keys, new_keys):
        dictionary[new_key] = dictionary.pop(old_key)
    return dictionary


def dict_merge(d1, d2):
    d3 = d1.copy()
    d3.update(d2)
    return d3


def add_entry_to_dicts(entry, dicts):
    if isinstance(dicts, dict):
        dicts = [dicts]
    for dictionary in dicts:
        key, value = entry
        dictionary[key] = value

    return dicts


def append_prefix_to_dicts(dicts, prefix):
    if isinstance(dicts,dict):
        dicts = [dicts]
    out = [append_prefix_to_dict(dictionary, prefix) for dictionary in dicts]
    return out


def list_of_dict_direct_product(ld1, ld2):
    out = []
    for dict1 in ld1:
        for dict2 in ld2:
            new_dict = dict1.copy()
            new_dict.update(dict2)
            out.append(new_dict)
    return out


def combine_parameter_spaces(list_of_parameter_spaces):
    return reduce(lambda dl1, dl2: list_of_dict_direct_product(dl1, dl2), list_of_parameter_spaces)


def fit_list(reg_list, X_train, Y_train, X_test, Y_test):
    Y_actual_train_l = []
    Y_actual_test_l = []
    Y_pred_train_l = []
    Y_pred_test_l = []

    for reg in reg_list:
        reg.fit(X_train, Y_train)
        Y_actual_train_l.append(Y_train)
        Y_actual_test_l.append(Y_test)
        Y_pred_train_l.append(reg.predict(X_train))
        Y_pred_test_l.append(reg.predict(X_test))

    return (Y_actual_train_l, Y_actual_test_l, Y_pred_train_l, Y_pred_test_l)


def bootstrap_fit(reg, X, Y, test_size=.4, n_iters=5):
    Y_actual_train = np.array([])
    Y_actual_test = np.array([])
    Y_pred_train = np.array([])
    Y_pred_test = np.array([])

    for i in range(n_iters):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        # print(X_train.shape,Y_train.shape,reg,i)
        reg.fit(X_train, Y_train)
        try:
            Y_pred_train = np.concatenate((Y_pred_train, reg.predict(X_train)))
            Y_pred_test = np.concatenate((Y_pred_test, reg.predict(X_test)))
        except:
            Y_pred_train = np.concatenate((Y_pred_train, reg.predict(X_train)[:, 0]))
            Y_pred_test = np.concatenate((Y_pred_test, reg.predict(X_test)[:, 0]))

        Y_actual_train = np.concatenate((Y_actual_train, Y_train))
        Y_actual_test = np.concatenate((Y_actual_test, Y_test))
    return (Y_actual_train, Y_actual_test, Y_pred_train, Y_pred_test)


def bootstrap_fit_list(reg_list, X, Y, test_size=.4, n_iters=5):
    Y_actual_train_l = []
    Y_actual_test_l = []
    Y_pred_test_l = []
    Y_pred_train_l = []

    for reg in reg_list:
        (Y_actual_train, Y_actual_test, Y_pred_train, Y_pred_test) = bootstrap_fit(reg, X, Y, test_size, n_iters)
        Y_actual_train_l.append(Y_actual_train)
        Y_actual_test_l.append(Y_actual_test)
        Y_pred_train_l.append(Y_pred_train)
        Y_pred_test_l.append(Y_pred_test)

    return (Y_actual_train_l, Y_actual_test_l, Y_pred_train_l, Y_pred_test_l)


def pickler(object, file_name):
    with open(file_name, "w") as f:
        pickle.dump(object, f)


def depickler(file_name):
    with open(file_name, "r") as f:
        return pickle.load(f)




