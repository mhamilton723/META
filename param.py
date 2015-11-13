import algo_parameters as algo_param
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from astro_utils import sigmaNMAD

########################### DATASET PARAMETERS ###########################

###### Dataset to use, can be a mix of labelled and unlabelled data
data_file = "data/raw_data/ananna_astro/AO13_Galex_Wise_2mass_point_ukidss_final_match_28.txt"
#data_file="data/raw_data/hamilton_astro/S82A_zphot_fix_sdss_galex_ukidss_wise_AB_ext.csv"
# data_file="data/S82A_sdss_galex_ukidss_wise_AB_ext.txt"
# data_file="raw_data/sdssfix_galex_ukidss_wise_aper_AB_ext.csv"

###### Parameters for specifying files feature names
feature_file = "data/raw_data/ananna_astro/astro_features_2.txt"  # Specify a file containing the variable names in the dataset
#feature_file = "data/raw_data/hamilton_astro/astro_features_1.txt"
#response_var = 'z_spec'  # Specify the response variable name
response_var = 'redshift'  # Specify the response variable name
parametric_col = 'None'  # Specify a column of precomputed parametric data
scorer = make_scorer(sigmaNMAD, greater_is_better=False)  # TODO incorporate scoring into main.py

###### Sub-sampling parameters, use to limit computation time for rapid development
labeled_subsample = 10000  # Size of labelled training and testing set
unlabeled_subsample = 10000  # Size of unlabelled training
debug_limit = 100 # number of lines of file to read in, useful for debugging speed

number_of_cores = 1

###########################################################################################################
#  Note: to adjust the settings for the regressors do this directly in the "algo_parameters.py" file      #
#        This file contains grid search ranges, and regressors                                            #
###########################################################################################################

########################### CONSTRUCTION OF PIPELINE ###########################

####IMPUTING
impute_data = True
optimize_impute = False
imputer_params = algo_param.IM_params

####SCALING 
scale_data = True
optimize_scale = False
scaler_params = algo_param.SC_params

####DIMENSIONALITY REDUCTION / UNSUPERVISED LEARNING
reduce_dim = True
optimize_dim_reduce = True
dim_reducer_params = algo_param.PCA_params2
#dim_reducer_params = algo_param.AE_params

####SUPERVISED REGRESSION 
optimize_regression = True
regressor_params = algo_param.RF_params# + algo_param.KR_params + algo_param.SV_params


########################### OPTIMIZING META PARAMETERS ###########################

#### Search Parameters
optimize_params = False  # Boolean to specify whether to optimize the marked parameters above
search_method = GridSearchCV  # Algorithm to perform meta-parameter optimization
test_size = .3  # the size of the test set relative to the training set
cv_folds = 2  # Number of times to try each setting of the parameters

#### Search output files
verbose_search_log = 'results/verbose_search_log.txt'  # log file to holds all the results of the optimization
best_search_log = "results/best_search_log.txt"  # log file that holds the best results of the optimization
verbose_search_pickle = "results/verbose_search_pickle.pkl"  # log file to holds all the results of the optimization
optimization_data_pickle = 'data/intermediate_data/optimization_data_pickle.pkl'  # Data used in calculating Grid search parameters, use this to avoid information leakage in fitting


########################### BAGGING MODELS ###############################

#######################################################################################
#  Note: If using grid search and bagging please set: using_grid_search_data = True   #
#######################################################################################

using_grid_search_data = False

n_estimators = 10  # The number of estimators used in a bagging regression. 1 corresponds to no bagging.
n_iterations = 1  # Used for estimating uncertainties. Beware of information leakage between datasets

ensemble_pickle = 'results/ensemble_pickle.pkl'  # File to save pickled results


########################### PLOTTING RESULTS ###############################

use_ensemble_pickle = False
ensemble_plots = 'results/ensemble_plots.png'  # file to save plots

print('Metaparameters Read')
