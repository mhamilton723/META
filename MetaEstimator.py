from __future__ import print_function

import param as param
import algo_parameters as algo_param
import utils as utils
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline


class MetaEstimator(BaseEstimator):
    # __slots__ = "algorithm","parameters"

    # TODO make this more elegant
    def __init__(self, **kwargs):
        if 'base_algorithm' in kwargs:
            algo = kwargs.pop('base_algorithm')
            self.base_algorithm = algo
        else:
            self.base_algorithm = None

        self.parameters = {}
        if kwargs:
            self.parameters = utils.dict_merge(self.parameters, kwargs)

    def __repr__(self):
        return "MetaEstimator: "+repr(self.base_algorithm)

    def set_params(self, **params):
        if 'base_algorithm' in params:
            algo = params.pop('base_algorithm')
            self.base_algorithm = algo
        if not self.parameters:
            self.parameters = {}
        if params:
            self.parameters = self.parameters = utils.dict_merge(self.parameters, params)
        if self.base_algorithm and self.parameters:
            self._update_algorithm()
        return self

    def _update_algorithm(self):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        if self.base_algorithm and self.parameters:
            self.base_algorithm = self.base_algorithm.set_params(**self.parameters)
        elif not self.base_algorithm:
            raise ValueError("No algorithm found")
        elif not self.parameters:
            print("WARNING: No parameters found")

    def get_params(self, deep=True):
        out = {}
        if self.base_algorithm is not None:
            dict1 = self.base_algorithm.get_params(deep)
            out.update(dict1)
            out['base_algorithm'] = self.base_algorithm
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
        return self.base_algorithm.fit(X, Y)

    def score(self, X, Y, **kwargs):
        return self.base_algorithm.score(X, Y, **kwargs)

    def predict(self, X):
        return self.base_algorithm.predict(X)


class MetaTransformer(MetaEstimator):
    # __slots__ = "fit_algorithm"

    def fit(self, X, Y=None):
        # self._update_algorithm()
        return self.base_algorithm.fit(X)

    def transform(self, X, Y=None):
        # self._update_algorithm()
        return self.base_algorithm.transform(X)

    def fit_transform(self, X, Y=None):
        # self._update_algorithm()
        return self.base_algorithm.fit_transform(X)

    def inverse_transform(self, X, copy=None):
        # self._update_algorithm()
        return self.base_algorithm.inverse_transform(X, copy)


class UnsupervisedPipeline(Pipeline):
    def __init__(self, steps, all_X=None, all_Y=None):
        super(UnsupervisedPipeline, self).__init__(steps)
        self.all_X = all_X
        self.all_Y = all_Y

    def fit(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """

        if self.all_X is not None:
            #print("Good")
            Xt, fit_params = self._pre_transform(self.all_X, self.all_Y, **fit_params)
            Xt, fit_params = self._transform(X)
        else:
            #print("Bad")
            Xt, fit_params = self._pre_transform(X, y, **fit_params)

        self.steps[-1][-1].fit(Xt, y, **fit_params)
        return self



    def _transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.iteritems():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return Xt, fit_params_steps[self.steps[-1][0]]


    def _reconstitute_y(self,X,Y,all_X):
        all_Y = []
        fill = np.nan
        count = 0
        for x, y in zip(X, Y):
            while np.any(all_X[count, :] != x):
                all_Y.append(fill)
                count += 1
                if count >= 2860:
                    print('here')
            all_Y.append(y)
            count += 1
            if count >= 2860:
                print('here')

        if count < len(all_X):
            all_Y += [fill]*(len(all_X)-count)
        return all_Y





def make_meta_pipeline(labeled_param_spaces,all_X = None, all_Y=None):
    components = []
    pipeline_param_spaces = []

    for label, param_spaces in labeled_param_spaces:

        if isinstance(param_spaces, dict):
            param_spaces = [param_spaces]

        #Check to make sure the components are not of mixed type
        is_regressor = False
        is_transformer = False
        for param_space in param_spaces:
            algo = param_space['base_algorithm'][0]
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
    pipeline = UnsupervisedPipeline(components, all_X,all_Y)

    return pipeline, output_parameter_space
