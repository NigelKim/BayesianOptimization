#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 21:29:28 2018

@author: Nigel
"""

""" gp.py
Bayesian optimisation of loss functions.
"""

import numpy as np
import pandas as pd
import sys
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from matplotlib import rc
from plotters import plot_iteration
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
from datetime import timedelta
from sklearn.model_selection import cross_val_score

import sklearn.gaussian_process as gp
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import gaussian_process
from sklearn import svm
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# =============================================================================
# ML 3 : MICE version features ------------------------------------------------
DATASET_PATH = "/Users/Nigel/Desktop/Wash U/2018 Junior Spring/Research-ML/code/routefinalimputed.csv"
data_features = ["congestion", "congestionSpeed", "warningcounts", "weather", "temperature", "windspeed"]
# -----------------------------------------------------------------------------

#DATASET_PATH = "/Users/Nigel/Desktop/Wash U/2018 Junior Spring/Research-ML/code/routefinalimputed.csv"

data = pd.read_csv(DATASET_PATH)
dataDay = data[data.day == 'Mon']  
train_x, test_x, train_y, test_y = train_test_split(dataDay[data_features],dataDay["routeoption"], train_size=0.2)

accuracy = 0
cvmean = 0
cvstd = 0
#cvs = []
y_score=[]
# =============================================================================

def sample_loss(params):
  C = params[0]
  gamma = params[1]
  epsilon = params[2]

  # Sample C and gamma on the log-uniform scale
  # model = SVC(C=(10) ** (C), gamma=(10) ** (gamma), random_state=12345)
#  model = SVR(gamma=(10)**(gamma), C=(10)**(C), epsilon=(10)**(epsilon))
  model = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=1.0)

#  model = LinearSVR(C=10**C, dual=True, epsilon=10**epsilon, fit_intercept=True,intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,random_state=0, tol=0.0001, verbose=0)

  pred=model.fit(train_x,train_y).predict(test_x)
#  pred = np.around(pred)
  
  return mean_squared_error(test_y, pred)
#  return cross_val_score(estimator=model,
#                         X=data,
#                         y=pred,
#                         # X=train_x,
#                         # y=train_y,
#                         scoring='roc_auc',
#                         cv=3).mean()

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]
    # n_params = 2

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            # print(params)
            # print(sample_loss(params))
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
#        kernel = gp.kernels.Matern()
        kernel = gp.kernels.RBF()
#        kernel = 'linear'
        # kernel = gp.kernels.RationalQuadratic()
#        kernel = gp.kernels.ExpSineSquared()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=False, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=False, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)
    return xp, yp

# =============================================================================
# MAIN CODE

# KernelSVR Without Optimization-----------------------------------------------
model = SVR(C=2, gamma=2, epsilon=1)
pred = model.fit(train_x,train_y).predict(test_x)
loss = mean_squared_error(test_y, pred)

# Make data.
lambdas = np.linspace(1, -4, 20)
gammas = np.linspace(1, -4, 20)
epsilons = np.linspace(1, -4, 20)

# We need the cartesian combination of these two vectors
param_grid = np.array([[C, gamma, epsilon] for gamma in gammas for C in lambdas for epsilon in epsilons])

real_loss = [sample_loss(params) for params in param_grid]

# The maximum is at:
param_grid[np.array(real_loss).argmax(), :]

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
C, G = np.meshgrid(lambdas, gammas)
Z = np.array(real_loss).reshape(C.shape)
surf = ax.plot_surface(C, G, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()



#rc('text', usetex=True)
#
#C, G, Ep = np.meshgrid(lambdas, gammas, epsilons)
#fig = plt.figure()
##ax = fig.add_subplot(111, projection="3d")
#plt.plot_surface(C, G, Ep, cmap="autumn_r")
#cp = plt.contour3(C, G, Ep, np.array(real_loss).reshape(C.shape))
#plt.colorbar(cp)
#plt.title('Filled contours plot of loss function $\mathcal{L}$($\gamma$, $C$)')
#plt.xlabel('$C$')
#plt.ylabel('$\gamma')
#plt.zlabel('$\epsilon')
#plt.savefig('/Users/Nigel/Desktop/Wash U/2018 Junior Spring/CSE 515T/project/real_loss_contour.png', bbox_inches='tight')
#plt.show()
# -----------------------------------------------------------------------------

# Bayesian Optimization to find optimal kernel hyperparameters for SVR---------

# gp_param should be either --- None or gparam
gparam = {'kernel':1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=2.5),'alpha':1e-5,'n_restarts_optimizer':10,'normalize_y':True}
x,y = bayesian_optimisation(n_iters=50, sample_loss=sample_loss, bounds=np.array([[-4,1],[-4,1],[-4,1]]), x0=None, n_pre_samples=3,
                      gp_params=gparam, random_search=100000, alpha=1e-5, epsilon=1e-7)
x_opt = x[y.argmin()]
y_opt = np.amin(y)
print("Optimal Loss from random search: {:.9f}\n".format(y_opt))
print("Optimal Hyperparameters(C,gamma,epsilon): {}\n".format(x_opt))

# KernelSVR With Optimization
model_opt = SVR(C=10**(x_opt[0]), gamma=10**(x_opt[1]), epsilon=10**(x_opt[2]))
pred_opt = model_opt.fit(train_x,train_y).predict(test_x)
loss_opt = mean_squared_error(test_y, pred_opt)

print("Loss (without BO): {:.9f}\n".format(loss))
print("Loss (with BO): {:.9f}\n".format(loss_opt))


rc('text', usetex=False)
plot_iteration(lambdas, x, y, first_iter=3, second_param_grid=gammas, optimum=[0.58333333, -2.15789474])
#plt.plot(X,y, label='True data')
##plt.plot(x_test[::end], pred[::end], 'co', label='SVR')
#plt.legend(loc='upper left');
#plt.show()