import hamilton_ml_mod as my

imputer = my.Imputer(missing_values=-99, strategy='mean')
scaler = my.StandardScaler()
dim_reduce = my.PCA(copy=True)
#dim_reduce_params={'n_components': range(1,n_comps+1), 'whiten':[True,False]}
#dim_reduce_params={'n_components': [1,2,5,10,15], 'whiten':[True,False]}
dim_reduce_params={'n_components': [10,15], 'whiten':[True]}

KR_params = [
	{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'alpha': [.1, 1, 10]},
	{'kernel': ['poly'], 'degree':[2,3,4], 'coef0': [0,1]}, 
	{'kernel': ['sigmoid'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'alpha': [.1, 1, 10],'coef0':[0,1]},
	{'kernel': ['linear'], 'alpha': [.1, 1, 10]}]

SV_params = [
	{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'C': [1, 10, 100, 1000]},
	{'kernel': ['poly'], 'degree':[2,3,4], 'coef0': [0,1]},
	{'kernel': ['sigmoid'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'C': [1, 10, 100, 1000],'coef0':[0,1]},
	{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]	

RF_params = [
	{'n_estimators': [10,50,100,200,400],
	'max_depth':[None,2,3,4,5]}]

GB_params = [
	{'loss' : ['ls', 'lad', 'huber'],
	'n_estimators': [10,50,100,200,400],
	'max_depth':[2,3,4,5,6]}]

AB_params = [
	{'n_estimators' : [10,50,100,200,400],
	'learning_rate' : [.01,.1,1,10],
	'loss' : ['linear', 'square', 'exponential']}]

BR_params = [
	{'alpha_1' : [1.e-7,1.e-6,1.e-5],
	'alpha_2' : [1.e-7,1.e-6,1.e-5],
	'lambda_1' : [1.e-7,1.e-6,1.e-5],
	'lambda_2' : [1.e-7,1.e-6,1.e-5]}]

KN_params = [
	{'n_neighbors' : [1,3,5,10],
	'weights':['uniform','distance'],
	'algorithm' : ['ball_tree', 'kd_tree'],
	'leaf_size' : [5,30,40]}]

NN_params = [
	{'iterations':[100,200,800],
	'layer_struct':[
	[10],[10,10],[10,10,10],[10,10,10,10],
	[20],[20,20],[20,20,20],[20,20,20,20],
	[50],[50,50],[50,50,50],[50,50,50,50],
	[41, 29, 20, 22, 18,20,20]] }]

NN_params_2 = [
	{'layer_struct':[[41, 29, 20, 22, 18,20,20]] }]

reg_list = [
	my.KernelRidge(),
	my.SVR(),
	my.RandomForestRegressor()]
	
param_space_list=[KR_params,SV_params,RF_params]
names_list=['KR','SV','RF']

#reg_list=[
#	my.KernelRidge(),
#	my.SVR(),
#	my.RandomForestRegressor(),
#	my.GradientBoostingRegressor(),
#	my.AdaBoostRegressor(),
#	my.BayesianRidge(),
#	my.KNeighborsRegressor()]
	
#param_space_list=[KR_params,SV_params,RF_params,GB_params,AB_params,BR_params,KN_params]
#names_list=['KR','SV','RF','GB','AB','BR','KN']

