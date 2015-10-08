import os
import theano
import theano.tensor as T
from collections import OrderedDict
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, regularization
from lasagne.objectives import aggregate
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import objective
from lasagne.regularization import regularize_layer_params, l2, l1
import numpy as np

lambda_regularization = 1e-3


def get_or_compute_grads(loss_or_grads, params):
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def apply_prox(loss_or_grads, params, learning_rate, momentum=.5, reg_l1=.1):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable, borrow=True)
        x = momentum * velocity - learning_rate * grad
        updates[velocity] = x
        updates[param] = T.sgn(param+x)*T.maximum(abs(param+x)-reg_l1, 0)

    return updates


def objective_with_L1(layers,
                      loss_function,
                      target,
                      aggregate=aggregate,
                      deterministic=False,
                      get_output_kw=None):
    reg = regularize_layer_params([layers["hidden4"], layers["hidden5"]], l1)
    loss = objective(layers, loss_function, target, aggregate, deterministic, get_output_kw)
    return loss + reg * lambda_regularization


class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 54, 54),
        use_label_encoder=True,
        verbose=1,
        #objective=objective_with_L1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    # Conv layer 1
    conv1_num_filters=64,
    conv1_filter_size=(3, 3),
    pool1_pool_size=(2, 2),

    # Conv layer 2
    conv2_num_filters=128,
    conv2_filter_size=(2, 2),
    pool2_pool_size=(2, 2),

    # Conv layer 3
    conv3_num_filters=128,
    conv3_filter_size=(2, 2),
    pool3_pool_size=(4, 4),

    # Layer 4
    hidden4_num_units=500,
    hidden4_nonlinearity = nonlinearities.leaky_rectify,
    dropout4_p=0,

    # Layer 5
    hidden5_num_units=500,
    hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.5,

    # Output
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,

    # Training
    #update= apply_prox,
    update_learning_rate=0.1,
    update_momentum=0.9,
    #update_reg_l1=.001,
    max_epochs=150,

    # handlers
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_loss')]
)


class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        X = X[:, :, 5:-5, 5:-5]
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        print('Start fitting')
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
