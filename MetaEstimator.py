from __future__ import print_function

import param as param
import algo_parameters as algo_param
import utils as utils

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline


class MetaEstimator(BaseEstimator):
    # __slots__ = "algorithm","parameters"

    # TODO make this more elegant
    def __init__(self, **kwargs):
        if 'algorithm' in kwargs:
            algo = kwargs.pop('algorithm')
            self.algorithm = algo
        else:
            self.algorithm = None

        self.parameters = {}
        if kwargs:
            self.parameters = utils.dict_merge(self.parameters, kwargs)

    def set_params(self, **params):
        if 'algorithm' in params:
            algo = params.pop('algorithm')
            self.algorithm = algo
        if not self.parameters:
            self.parameters = {}
        if params:
            self.parameters = self.parameters = utils.dict_merge(self.parameters, params)
        if self.algorithm and self.parameters:
            self._update_algorithm()
        return self

    def _update_algorithm(self):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        if self.algorithm and self.parameters:
            self.algorithm = self.algorithm.set_params(**self.parameters)
        elif not self.algorithm:
            raise ValueError("No algorithm found")
        elif not self.parameters:
            print("WARNING: No parameters found")

    def get_params(self, deep=True):
        out = {}
        if self.algorithm is not None:
            dict1 = self.algorithm.get_params(deep)
            out.update(dict1)
            out['algorithm'] = self.algorithm
        if self.parameters:
            dict2 = self.parameters
            out.update(dict2)
        return out


class MetaRegressor(MetaEstimator):
    # __slots__ = "fit_algorithm"

    # def __init__(self, **kwargs):
    #    super(MetaRegressor, self).__init__(**kwargs)

    def fit(self, X, Y=None):
        # self._update_algorithm()
        return self.algorithm.fit(X, Y)

    def score(self, X, Y, **kwargs):
        return self.algorithm.score(X, Y, **kwargs)

    def predict(self, X):
        return self.algorithm.predict(X)


class MetaTransformer(MetaEstimator):
    # __slots__ = "fit_algorithm"

    def fit(self, X, Y=None):
        # self._update_algorithm()
        return self.algorithm.fit(X)

    def transform(self, X, Y=None):
        # self._update_algorithm()
        return self.algorithm.transform(X)

    def fit_transform(self, X, Y=None):
        # self._update_algorithm()
        return self.algorithm.fit_transform(X)

    def inverse_transform(self, X, copy=None):
        # self._update_algorithm()
        return self.algorithm.inverse_transform(X, copy)



def make_meta_pipeline(labeled_param_spaces):
    components = []
    pipeline_param_spaces = []

    for label, param_spaces in labeled_param_spaces:

        if isinstance(param_spaces, dict):
            param_spaces = [param_spaces]

        #Check to make sure the components are not of mixed type
        is_regressor = False
        is_transformer = False
        for param_space in param_spaces:
            algo = param_space['algorithm'][0]
            if issubclass(type(algo), RegressorMixin):
                meta_object = MetaRegressor
                is_regressor = True
            elif issubclass(type(algo), TransformerMixin):
                meta_object = MetaTransformer
                is_transformer = True
            else:
                raise ValueError('Need to be a subclass of Regressors or Transformers')

        if is_regressor and is_transformer:
            raise ValueError('Cannot have mixed regressors and transformers in the same component')
        components.append((label, meta_object()))

        new_param_spaces = utils.append_prefix_to_dicts(param_spaces, label + '__')
        pipeline_param_spaces.append(new_param_spaces)


    output_parameter_space = utils.combine_parameter_spaces(pipeline_param_spaces)
    pipeline = Pipeline(components)

    return pipeline, output_parameter_space
