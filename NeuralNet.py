from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from sklearn.metrics import mean_squared_error

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, FullConnection, LinearLayer, SigmoidLayer, SoftmaxLayer, TanhLayer, \
    GaussianLayer


class NeuralNet(BaseEstimator, RegressorMixin):
    def __init__(self, layer_struct=[20, 20], iterations=200, function='Sigmoid'):
        self.layer_struct = layer_struct
        self.iterations = iterations
        self.function = function

    def fit(self, X, Y):
        self.n_features = X.shape[1]
        self.net = self._create_net(self.n_features, self.layer_struct, self.function)
        TrainDS = self.skl2pb(X, Y)
        return self._train_net(TrainDS, self.net, self.iterations )

    def score(self, X, Y, **kwargs):
        results = self.predict(X)
        score = mean_squared_error(results, Y)  #TODO add arbitrary scoring algorithm
        return score

    def predict(self, X):
        results = np.zeros(len(X))
        for k in range(0, len(X)):
            results[k] = self.net.activate(X[k])
        return results

    @staticmethod
    def _create_net(n_features, layer_struct, function='Sigmoid'):  # return(net)
        """ takes in a structure vector and creates a feed forward network"""

        # catch errors
        if min(layer_struct) < 1:
            raise ValueError('cannot create network with less than 1 layer')

        # initialize variables
        layers = []
        connections = []
        net = FeedForwardNetwork()

        # generate layers
        LI = LinearLayer(n_features)
        LO = LinearLayer(1)
        net.addInputModule(LI)
        net.addOutputModule(LO)

        string2layer_type = {'Sigmoid': SigmoidLayer,
                             'Softmax': SoftmaxLayer,
                             'Linear': LinearLayer,
                             'Gaussian': GaussianLayer,
                             'Tanh': TanhLayer}

        if string2layer_type[function]:
            layer_type = string2layer_type[function]
        else:
            print('Unknown function: Reverting to Sigmoid')
            layer_type = SigmoidLayer

        for i in range(0, len(layer_struct)):
            layers.append(layer_type(layer_struct[i]))
            net.addModule(layers[i])

        # connect layers
        CI = FullConnection(LI, layers[0])
        net.addConnection(CI)
        CO = FullConnection(layers[-1], LO)
        net.addConnection(CO)

        for i in range(0, len(layer_struct) - 1):
            connections.append(FullConnection(layers[i], layers[i + 1]))
            net.addConnection(connections[i])

        net.sortModules()
        return net


    # Train the network
    @staticmethod
    def _train_net(TrainDS, net, iterations):  # return()
        trainer = BackpropTrainer(net, TrainDS)
        # trainer.trainUntilConvergence(dataset=TrainDS, maxEpochs=20, verbose=True)
        for i in range(0, iterations):
            trainer.train()
        return net

    @staticmethod
    def skl2pb(X, Y):
        DS = SupervisedDataSet(X.shape[1], 1)
        for i in range(0, len(X)):
            DS.addSample(X[i], Y[i])
        return DS

