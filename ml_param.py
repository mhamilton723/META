####### DATASET PARAMETERS
subsample=400
#data_file="data/S82A_sdss_galex_ukidss_wise_AB_ext.txt"
data_file="raw_data/sdssfix_galex_ukidss_wise_aper_AB_ext.csv"
#data_file="data/S82A_zphot_sdss_galex_ukidss_wise_AB_ext.txt"

####### FEATURE PARAMETERS
feature_file="features.txt"
response_var='z_spec'
parametric_col='None'

####### FILE PARAMETERS
test_size = .3
split_data_file='data/split_data.pkl'


opt_verbose_file='results/opt_verbose.txt'
opt_best_file="results/opt_best.txt"
opt_pickle_file="results/opt_verbose.pkl"
ensemble_pickle_file='results/ens_results.pkl'

nonopt_res_file='results/nonopt_results.pdf'
opt_ensemble_res_file='results/opt_ensemble_results.png'

####### GRID SEARCH PARAMETERS
cv_folds=5

####### BAGGING PARAMETERS
n_estimators = 3
n_iterations = 1

print('Metaparameters Read')