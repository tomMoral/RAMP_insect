import os
from sklearn.base import BaseEstimator
import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from lasagne import layers, nonlinearities, init, regularization
from lasagne.objectives import aggregate
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.base import objective
from lasagne.regularization import regularize_layer_params, l2, l1
import numpy as np

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
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    # conv1_nonlinearity = nonlinearities.very_leaky_rectify,
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500,
    hidden4_nonlinearity=nonlinearities.rectify,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_num_units=500,
    output_num_units=18,
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    # update_momentum=0.9,
    update=adagrad,
    max_epochs=100,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)]
)

class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        X = X[:, :, 10:-10, 10:-10]
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def preprocess_train(self, X, y):
        y_train = list(self.preprocess_y(y))
        X_train = list(self.preprocess(X))
        N = len(y)
        i0 = np.random.choice(N, N/2)
        X_train.extend([X_train[i]+.1*np.random.normal(size=(44, 44)) for i in i0])
        y_train.extend([y_train[i] for i in i0])
        print(len(X_train), len(y_train))
        X_train = np.array(X_train).astype(np.float32)
        return X_train, np.array(y_train)

    def fit(self, X, y):
        X_train, y_train = self.preprocess_train(X, y)
        print('Start fitting')
        self.net.fit(X_train, y_train)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
