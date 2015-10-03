from __future__ import print_function
import sys
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor

import utils as utils
import algo_parameters as algo_param
import param as param
from MetaEstimator import make_meta_pipeline

reload(utils)
reload(algo_param)
reload(param)


# create scorer
sigNMAD = make_scorer(utils.sigmaNMAD, greater_is_better=False)

# TODO Add unlabeled subset functionality

#####################   PERFORM GRID SEARCH    ########################
if param.optimize_params:
    # parse data

    all_X, all_Y = utils.parse(param.data_file, param.feature_file, param.response_var)
    X, Y = utils.labeled_subset(all_X, all_Y)
    X, Y = utils.subsample((X, Y), param.labeled_subsample)
    (X_train, X_test, Y_train, Y_test) = utils.train_test_split(X, Y, test_size=param.test_size)

    # pickle data for use in other files
    saved_data = (X_train, X_test, Y_train, Y_test)
    utils.pickler(saved_data, param.optimization_data_pickle)

    pipeline, parameter_space = make_meta_pipeline([
        ('imputer', param.imputer_params),
        ('scaler', param.scaler_params),
        ('dim_reducer', param.dim_reducer_params),
        ('regressor', param.regressor_params)
    ])

    print("Opening logfiles")
    sys.stdout.flush()
    with open(param.verbose_search_log, "w+") as log_verbose, open(param.best_search_log, "w+") as log_best:

        # perform grid search
        print('Performing Grid Search with', pipeline)
        sys.stdout.flush()
        print('')
        # grid = param.search_method(pipeline, parameter_space, cv=param.cv_folds, error_score=np.NaN, scoring=sigNMAD)
        # grid = algo_param.GridSearchCV(pipeline, parameter_space, cv=param.cv_folds, error_score=np.NaN, scoring=sigNMAD)
        grid = param.search_method(pipeline, parameter_space, cv=param.cv_folds, error_score=np.NaN)
        grid.fit(X_train, Y_train)

        print("Best parameters set found on development set:")
        print(grid.best_params_, file=log_best)
        print(grid.best_params_)
        print('', file=log_best)
        print('')

        for params, mean_score, scores in grid.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (
                mean_score, scores.std() / 2, params), file=log_verbose)

    utils.pickler(grid, param.verbose_search_pickle)
    print('Grid search pickled')

####################  PERFORM ENSEMBLE METHOD  ###########################

loaded_ensemble_data = False
if param.use_ensemble_pickle:
    try:
        results, names = utils.depickler(param.ensemble_pickle)
        (Y_actual_train_l, Y_actual_test_l, Y_pred_test_l, Y_pred_train_l) = results
        loaded_ensemble_data = True
        print('Loaded Ensemble Results')

    except IOError:
        print("No Ensemble Results found, proceeding to calculate results.")

loaded_optimization_data = False
if not param.use_ensemble_pickle or not loaded_ensemble_data:

    pipe_list = []
    if param.using_grid_search_data:
        try:
            saved_data = utils.depickler(param.optimization_data_pickle)
            (X_train, X_test, Y_train, Y_test) = saved_data
            loaded_optimization_data = True

            grid = utils.depickler(param.verbose_search_pickle)
            pipe_list = utils.best_pipelines_by_algo(grid)

            print('Loaded optimization data')
        except IOError:
            print('Could not load optimization data')

    if not param.using_grid_search_data or not loaded_optimization_data:
        print('Not using optimization data: Re-parsing File')
        # parse data
        X, Y, X_unlabeled = utils.parse(param.data_file, param.feature_file, param.response_var)
        X, Y = utils.subsample((X, Y), param.labeled_subsample)
        X_train, X_test, Y_train, Y_test = utils.train_test_split(X, Y, test_size=param.test_size)

        pipeline, parameter_space = make_meta_pipeline([
            ('imputer', param.imputer_params),
            ('scaler', param.scaler_params),
            ('dim_reducer', param.dim_reducer_params),
            ('regressor', param.regressor_params)
        ])

        pipe_list = utils.default_pipelines_by_algo(pipeline, parameter_space)

    for pipe in pipe_list:
        pipe.steps[3] = ('regressor',
                         BaggingRegressor(pipe.steps[3][1],
                                          n_estimators=param.n_estimators,
                                          oob_score=True))

    # If there are no fears of contamination
    # bootstrap for better visualization of errors
    if not loaded_optimization_data:
        (Y_actual_train_l, Y_actual_test_l, Y_pred_train_l, Y_pred_test_l) = \
            utils.bootstrap_fit_list(pipe_list, X, Y, test_size=param.test_size, n_iters=param.n_iterations)
    else:
        (Y_actual_train_l, Y_actual_test_l, Y_pred_train_l, Y_pred_test_l) = \
            utils.fit_list(pipe_list, X_train, Y_train, X_test, Y_test)


    results = (Y_actual_train_l, Y_actual_test_l, Y_pred_test_l, Y_pred_train_l)
    names = [algo_param.pipeline_to_name(pipe) for pipe in pipe_list]

    utils.pickler((results, names), param.ensemble_pickle)
    print('Results Pickled')


print('Plotting Results')
plt.clf()
plt.close()
plt.figure(figsize=(23, 10))

for i in range(len(names)):
    plt.subplot(2, len(names), i + 1)
    utils.astro_plot(Y_actual_train_l[i], Y_pred_train_l[i], names[i] + ' train')
    plt.subplot(2, len(names), len(names) + i + 1)
    utils.astro_plot(Y_actual_test_l[i], Y_pred_test_l[i],names[i] + ' test')
plt.savefig(param.ensemble_plots)
plt.show()
