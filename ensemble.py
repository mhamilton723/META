
from __future__ import print_function
import matplotlib.pyplot as plt
import MetaEstimator as my
import algo_parameters as reglist
import ml_param as param
reload(my); reload(reglist); reload(param)

try:
    results=my.depickler(param.ensemble_pickle_file)
    print('Loaded Ensemble Results')
    (long_train,long_res_train,long_test,long_res_test)=results
except IOError:
    print("No Ensemble Results found, proceeding to calculate results.")

    loaded_data=False
    try:
        saved_data = my.depickler(param.split_data_file)
        (X_train, X_test, Y_train, Y_test) = saved_data
        print('Loaded previous data')
        loaded_data=True
    except IOError:
        print('CONTAMINATION WARNING:')
        print('Could not load previous data, reparsing data')

        #parse data
        data=my.parser(param.data_file,param.feature_file,param.response_var)
        data=my.subsample(data,param.subsample)
        X=data[0]; Y=data[1]

    reg_list=[]
    try:
        grid_list=my.depickler(param.opt_pickle_file)
        print('Loaded previous grid search results')
        for grid in grid_list:
            reg=grid.best_estimator_
            reg.steps[3]=('reg', my.BaggingRegressor(reg.steps[3][1], n_estimators=param.n_estimators,oob_score=True))
            reg_list.append(reg)
    except IOError:
        print('Cound not load previous grid search results')
        print('Bagging default regressors')

        for reg,param_space in zip(reglist.reg_list, reglist.param_space_list):
            pipe=my.Pipeline([
                ('inputer',reglist.imputer),
                ('scaler',reglist.scaler),
                ('dim_reduce',reglist.dim_reduce),
                ('reg',reg)])
            reg_list.append(pipe)

   #If there are no fears of contamination
   #bootstrap for better visualization of errors
    if not loaded_data:
        (Y_actual_train_l,Y_actual_test_l,Y_pred_train_l,Y_pred_test_l) = \
        my.bootstrap_fit_list(reg_list,X,Y,test_size=param.test_size,n_iters=param.n_iterations)
    else:
        (Y_actual_train_l,Y_actual_test_l,Y_pred_train_l,Y_pred_test_l) = \
        my.fit_list(reg_list,X_train,Y_train,X_test,Y_test)

    results = (Y_actual_train_l,Y_actual_test_l,Y_pred_test_l,Y_pred_train_l)
    my.pickler(results,param.ensemble_pickle_file)
    print('Results Pickled')

print('Plotting Results')
plt.clf()
plt.close()
plt.figure(figsize=(23,10))
size=len(reg_list)
for i in range(0,size):
    plt.subplot(2,size,i+1)
    my.astro_plot(Y_actual_train_l[i],Y_pred_train_l[i],reglist.names_list[i]+' train')
    plt.subplot(2,size,size+i+1)
    my.astro_plot(Y_actual_test_l[i],Y_pred_test_l[i],reglist.names_list[i]+' test')
plt.savefig(param.opt_ensemble_res_file)
plt.show()
