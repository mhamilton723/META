from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression, BayesianRidge, RANSACRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from NeuralNet import NeuralNet
from sklearn.preprocessing import Imputer, StandardScaler
from sknn.ae import AutoEncoder, Layer

IM_params = {'base_algorithm': [Imputer()], 'missing_values': [-99], 'strategy': ['mean']}

SC_params = {'base_algorithm': [StandardScaler()]}

# dim_reduce_params={'n_components': range(1,n_comps+1), 'whiten':[True,False]}
# dim_reduce_params={'n_components': [1,2,5,10,15], 'whiten':[True,False]}
PCA_params = {'base_algorithm': [PCA()], 'copy': [True], 'n_components': [10, 15], 'whiten': [True]}

PCA_params2 = [{'base_algorithm': [PCA()], 'copy': [True], 'n_components': [10, 15, 21], 'whiten': [True]}]

AE_layers = [Layer('Sigmoid', units=32), Layer('Sigmoid', units=16), Layer('Sigmoid', units=8)]
AE_params = [{ 'base_algorithm': [AutoEncoder(AE_layers)] }]

KR_params = [
    {'base_algorithm': [KernelRidge()], 'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'alpha': [.1, 1, 10]},
    {'base_algorithm': [KernelRidge()], 'kernel': ['poly'], 'degree': [2, 3, 4], 'coef0': [0, 1]},
    {'base_algorithm': [KernelRidge()], 'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'alpha': [.1, 1, 10],
     'coef0': [0, 1]},
    {'base_algorithm': [KernelRidge()], 'kernel': ['linear'], 'alpha': [.1, 1, 10]}]

SV_params = [
    {'base_algorithm': [SVR()], 'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]},
    {'base_algorithm': [SVR()], 'kernel': ['poly'], 'degree': [2, 3, 4], 'coef0': [0, 1]},
    {'base_algorithm': [SVR()], 'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000],
     'coef0': [0, 1]},
    {'base_algorithm': [SVR()], 'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

RF_params = [
    {'base_algorithm': [RandomForestRegressor()],
     'n_estimators': [10, 50, 100, 200, 400],
     'max_depth': [None, 2, 3, 4, 5]}]

GB_params = [
    {'base_algorithm': [GradientBoostingRegressor()],
     'loss': ['ls', 'lad', 'huber'],
     'n_estimators': [10, 50, 100, 200, 400],
     'max_depth': [2, 3, 4, 5, 6]}]

AB_params = [
    {'base_algorithm': [AdaBoostRegressor()],
     'n_estimators': [10, 50, 100, 200, 400],
     'learning_rate': [.01, .1, 1, 10],
     'loss': ['linear', 'square', 'exponential']}]

BR_params = [
    {'base_algorithm': [BayesianRidge()],
     'alpha_1': [1.e-7, 1.e-6, 1.e-5],
     'alpha_2': [1.e-7, 1.e-6, 1.e-5],
     'lambda_1': [1.e-7, 1.e-6, 1.e-5],
     'lambda_2': [1.e-7, 1.e-6, 1.e-5]}]

KN_params = [
    {'base_algorithm': [KNeighborsRegressor()],
     'n_neighbors': [1, 3, 5, 10],
     'weights': ['uniform', 'distance'],
     'algorithm': ['ball_tree', 'kd_tree'],
     'leaf_size': [5, 30, 40]}]

NN_params = [
    {'base_algorithm': [NeuralNet()],
     'iterations': [100, 200, 800],
     'layer_struct': [
         [10], [10, 10], [10, 10, 10], [10, 10, 10, 10],
         [20], [20, 20], [20, 20, 20], [20, 20, 20, 20],
         [50], [50, 50], [50, 50, 50], [50, 50, 50, 50],
         [41, 29, 20, 22, 18, 20, 20]]}]

NN_params_2 = [
    {'base_algorithm': [NeuralNet()],
     'layer_struct': [[41, 29, 20, 22, 18, 20, 20]]}]




#### USE THIS TO ASSOCIATE PARAMETER SPACES WITH ALGORITHIMS  ####
#### NOTE ONE CAN ONLY USE TYPES TO HASH, DO NOT INCLUDE ()   ####
algo2params = {
    Imputer: IM_params,
    StandardScaler: SC_params,
    PCA: PCA_params,
    GradientBoostingRegressor: GB_params,
    AdaBoostRegressor: AB_params,
    BayesianRidge: BR_params,
    KNeighborsRegressor: KN_params,
    KernelRidge: KR_params,
    SVR: SV_params,
    RandomForestRegressor: RF_params}

algo2names = {
    Imputer: 'IM',
    StandardScaler: 'SC',
    PCA: 'PC',
    GradientBoostingRegressor: 'GB',
    AdaBoostRegressor: 'AB',
    BayesianRidge: 'BR',
    KNeighborsRegressor: 'KN',
    KernelRidge: 'KR',
    SVR: 'SV',
    RandomForestRegressor: 'RF'}


def algo_to_param(instantiated_algo, dict=algo2params):
    return dict[type(instantiated_algo)]


def algo_to_name(instantiated_algo, dict=algo2names):
    return dict[type(instantiated_algo)]


def params_to_name(param_space):
    name_list = [algo_to_name(param_space['imputer__base_algorithm']) + '_',
                 algo_to_name(param_space['scaler__base_algorithm']) + '_',
                 algo_to_name(param_space['dim_reducer__base_algorithm']) + '_',
                 algo_to_name(param_space['regressor__base_algorithm']) + '_']
    name = ''.join(name_list)
    return name


def pipeline_to_name(pipeline):
    return params_to_name(pipeline.get_params())


def algo_list_to_params(algo_list):
    out = []
    for algo in algo_list:
        param_obj = algo_to_param(algo)
        if isinstance(param_obj, list):
            for param_dict in param_obj:
                param_dict['base_algorithm'] = [algo]
                out.append(param_dict)
        if isinstance(param_obj, dict):
            param_obj['base_algorithm'] = [algo]
            out.append(param_obj)
    return out
