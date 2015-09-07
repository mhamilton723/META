####### DATASET PARAMETERS
subsample=100
#data_file="data/S82A_sdss_galex_ukidss_wise_AB_ext.txt"
data_file="data/sdssfix_galex_ukidss_wise_aper_AB_ext.csv"
#data_file="data/S82A_zphot_sdss_galex_ukidss_wise_AB_ext.txt"

####### FEATURE PARAMETERS
feature_file="features.txt"
response_var='z_spec'
parametric_col='None'

####### FILE PARAMETERS
opt_verbose_file='results/optimization_verbose_2.txt'
opt_best_file="results/optimization_best_2.txt"

opt_pickle_file="results/optimization_verbose_2.pkl"
ensemble_pickle_file='results/ensemble_results_2.pkl'

nonopt_res_file='results/non_opt_results.pdf'
opt_ensemble_res_file='results/opt_ensemble_results.png'

####### GRID SEARCH PARAMETERS
test_size=.2
cv_opt=2
cv_plot=1

####### BAGGING PARAMETERS
n_estimators = 3
n_iterations = 1

print('Metaparameters Read')