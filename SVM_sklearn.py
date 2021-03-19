# File name: SVM_sklearn.py

import time
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def confusion_matrix_accuracy(cm):
    """
    confusion_matrix_accuracy method
    :param cm: {array-like}
    :return: {float}
    """
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return diagonal_sum / sum_of_all_elements


def support_vector_machine(X_train, X_test, y_train, y_test):
    """
    support_vector_machine method
    :param X_train: {array-like}
    :param X_test: {array-like}
    :param y_train: {array-like}
    :param y_test: {array-like}
    :return: SVM estimator
    """
    start = time.time()
    # Using SVC method
    """
        class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', 
                coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
                class_weight=None, verbose=False, max_iter=-1, 
                decision_function_shape='ovr', break_ties=False, random_state=None)

        C=1.0: Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty.
       
        kernel='rbf': Specifies the kernel type to be used in the algorithm. 
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
            If none is given, ‘rbf’ (Radial basis function kernel) will be used. If a callable
            is given it is used to pre-compute the kernel matrix from data matrices; 
            that matrix should be an array of shape (n_samples, n_samples).
            
        degree=3: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        
        gamma='scale': gamma{‘scale’, ‘auto’} or float, default=’scale’
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                    if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
                    if ‘auto’, uses 1 / n_features.
        
        random_state=None: random_state: int, RandomState instance or None, default=None
            Controls the pseudo random number generation for shuffling the data for probability estimates. 
            Ignored when probability is False. 
            Pass an int for reproducible output across multiple function calls.
        
        
        """
    svm_model = SVC(kernel='linear', random_state=0)
    # fit model no training data
    svm_model.fit(X_train, y_train)

    # Predict on the training sets (X_test and y_test)
    print('\n*** Predictions:')
    predictions = svm_model.predict(X_test)

    # Test accuracy using svc.score
    _score = svm_model.score(X_test, y_test)
    print("\tSVM Score Function:: {%.2f%%}" % (_score * 100.0))
    # Test accuracy using accuracy_score
    _accuracy_score = accuracy_score(y_test, predictions)
    print("\tAccuracy Score Function: {%.2f%%}" % (_accuracy_score * 100.0))
    # Test accuracy using confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    _confusion_matrix_accuracy = confusion_matrix_accuracy(cm)
    print("\tConfusion Matrix Accuracy Score: {%.2f%%}" % (_confusion_matrix_accuracy * 100.0))

    # Pull out the slope and intercept parameters from results
    intercept = svm_model.intercept_
    slope = svm_model.coef_[0]
    print("\n--- Intercept - β0 = ", intercept)
    print("--- Slope - β1 = ", slope)

    end = time.time()
    print('Execution Time: {%f}' % ((end - start) / 1000) + ' seconds.')

    return svm_model


def plot_svm(svm, X, y, step=0.1):
    """
    plot_svm method
    :param svm: SVM estimator
    :param X: {array-like}
    :param y: {array-like}
    :param step: {float}
    :return: None
    """
    from matplotlib.colors import ListedColormap

    plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)

    x_min, x_max = np.amin(X[:, 0]) - 1, np.amax(X[:, 0]) + 1
    y_min, y_max = np.amin(X[:, 1]) - 1, np.amax(X[:, 1]) + 1

    # # Create a mesh to plot:
    # np.arange(start, stop, step)
    xx = np.arange(x_min, x_max, step)
    yy = np.arange(y_min, y_max, step)

    # np.meshgrid return coordinate matrices from coordinate vectors.
    x1, x2 = np.meshgrid(xx, yy)

    z = svm.predict(np.c_[x1.ravel(), x2.ravel()])
    z = z.reshape(x1.shape)

    # plot contour
    markers = ['x', '.', 'o', 's']
    colors = ['green', 'red', 'blue', 'yellow']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    linestyles = ['dashed', 'solid', 'dashed', 'solid']
    plt.contourf(x1, x2, z,
                 alpha=0.3,
                 cmap=cmap,
                 linestyles=linestyles
                 )
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    # Plot X training points
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.9, c=cmap(idx),
                    marker=markers[idx], label='Class ' + str(cl))

    # Format the plot
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    plt.title('SVM', fontdict=font)
    plt.xlabel('petal length (cm) - X1', fontdict=font, loc='right')
    plt.ylabel('petal width (cm) - X2', fontdict=font, loc='top')

    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper right')
    #plt.show()


if __name__ == '__main__':
    print('\nSupport Vector Machine ' + '=' * 40)

    # Load data
    iris = datasets.load_iris()

    # Append a new column called flower
    flower_names = iris.target_names

    # Create X from columns: 3 and 4 of iris data
    X = iris.data[:, [2, 3]]  # X = iris.data[:, [2, 3]]
    # Create y from iris target
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)

    svm = support_vector_machine(X_train, X_test, y_train, y_test)
    
    # Plot the svm contour region
    plot_svm(svm, X, y, step=0.2)
    
    # Plot the confusion matrix
    plot_confusion_matrix(svm, X_test, y_test, display_labels=iris.target_names)
    plt.show()
