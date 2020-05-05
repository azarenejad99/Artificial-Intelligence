import numpy as np
import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from copy import deepcopy

def alphabetize(x,y):
    if x.get_name()>y.get_name():
        return 1
    return -1

def abs_mean(values):
    """Compute the mean of the absolute values a set of numbers.
    For computing the stopping condition for training neural nets"""
 
    return np.mean(np.abs(values))

def finite_difference(network):
    for w in network.weights:
        network.clear_cache()
        cur_val = network.performance.output()
        w.set_value(w.get_value() + 1e-8)
        network.clear_cache()
        new_value = network.performance.output()
        w.set_value(w.get_value() - 1e-8)
        finite_diff = (new_value - cur_val) / 1e-8
        print(f'{w.get_name():5s} finite:{finite_diff: 2.4f} real:{network.performance.dOutdX(w): 2.4f}', end='')
        if abs(network.performance.dOutdX(w) - finite_diff) < 1e-4:
            print("True")
        else:
            print("False")
    network.clear_cache()


def plot_decision_boundary(network, data, xmin=-10, xmax=10, ymin=-10, ymax=10):
    print("plot is")
    X = np.array([[item[0], item[1]] for item in data])
    y = np.array([item[2] for item in data])
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10*h, X[:, 0].max() + 10*h
    y_min, y_max = X[:, 1].min() - 10*h, X[:, 1].max() + 10*h

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    _chiz = np.c_[xx.ravel(), yy.ravel()]
    _new_data = []
    for i in range(len(_chiz)):
        _new_data.append((_chiz[i, 0], _chiz[i, 1]))
    z = []
    for datum in _new_data:
        for i in range(len(network.inputs)):
            network.inputs[i].set_value(datum[i])
        network.clear_cache()
        result = network.output.output()
        prediction = round(result)
        network.clear_cache()
        z.append(prediction)
    z = np.array(z)
    z = z.reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, z, cmap='Paired_r', alpha=0.25)
    plt.contour(xx, yy, z, colors='k', linewidths=0.02)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired_r', edgecolors='k')
    plt.show()
    input('enter to continue')

