import numpy as np
from matplotlib import pyplot as plt


def multi_linear_regression_train(X_train, y_train, epochs=1000, learning_rate=0.01):
    '''
    Given a training subset of with m categories, each with n data points, this function will
    train a multivariate linear regression model by varying the bias (b) and weights of each
    (w_i) variable.

                                y = b + w_0*X_0 + w_1*X_1 + ...

    Inputs:
    -------
    - X_train: np.array((n,m))
        Array containing m categories and n data points for each category.
    - y_train: np.array((m,1))
        Array containing the correct outcome value for each row of data.
    - epochs: int
        The number of repeats of the training algorithm used to train the model.
    - learning_rate: float
        A multiplier for the change in variables at each epoch.

    Returns:
    --------
    - w: np.array((m+1))
        The bias (w[0,0]) and weight of each variable trained upon.
    - L: list
        The loss function, here defined by the mean squared error, at each epoch.
    '''

    N = len(X_train)
    ones = np.ones(N)
    Xp = np.c_[ones, X_train]  # concatenate

    y_train = y_train.values.reshape(1, -1)

    w = 2*np.random.rand( X_train.shape[1]+1 )-1  # random initial weight between -1 and 1

    L = []  # loss function

    print(f'{"epoch":<10}mse')
    for epoch in range(epochs):

        y_predicted = w @ Xp.T
        error = y_train-y_predicted
        mse = np.mean(error**2)  # mean squared error
        L.append(mse)

        # optimise variables using gradient descent
        gradient = - 1/N * error @ Xp
        w = w - learning_rate*gradient

        if epoch % (epochs/10) == 0:
            print(f'{epoch:<10}{mse}')

    print('\n')

    # Outputs

    print('Weights')
    # max_len = max( [len(str) for str in X.columns] + [len('Bias')] )
    if w[0, 0] >= 0:
        print(f'{"Intercept":<30}{w[0,0]}')
    else:
        print(f'{"Intercept":<29}{w[0,0]}')

    i = 1
    for col in X_train:
        if w[0, i] >= 0:
            print(f'{col:<30}{w[0,i]}')
        else:
            print(f'{col:<29}{w[0,i]}')
        i += 1

    return w, L


def multi_linear_regression_test(X_test, y_test, w):
    '''
    After training a model using multi_linear_regression_train, the resulting weights (w) can
    be tested on an independent set of data, X_test.

    Inputs:
    -------
    - X_test: np.array((n,m))
        Array containing m categories and n data points for each category.
    - y_test: np.array((m,1))
        Array containing the correct outcome value for each row of data.
    - w: np.array((m+1))
        The bias (w[0]) and weight of each variable.

    Returns:
    --------
    - mse: float
        the mean squared error between the y_predicted values and the y_test values. This is
        used as a metric to test the accuracy of the model.
    '''
    N = len(X_test)
    ones = np.ones(N)
    Xp = np.c_[ones, X_test]  # concatenate

    y_test = y_test.values.reshape(1,-1)

    y_predicted = w @ Xp.T
    error = y_test-y_predicted
    mse = np.mean(error**2)  # mean squared error
    mae = np.mean(abs(error))  # mean absolute error

    diff = y_test - np.mean(y_test)

    r2 = 1 - (np.sum(error**2)/np.sum(diff**2)) # R^2

    return mse, mae, r2


def plot_loss(L):
    '''
    A function that plots the evolution of the loss function of multi_linear_regression_train
    for each epoch. There are two plots: one with a reguarly scaled y-axis (left) and one with
    a logarithmicly scaled y-axis (right).

    Inputs:
    -------
    - L: list
        List of the loss function (mean squared error) for each epoch of the training.

    Returns:
    --------
    None
    '''

    fig, ax = plt.subplots(ncols=2, figsize=(20, 5))

    ax1 = ax[0]
    ax1.plot(list(range(len(L))), L)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Loss - Mean Squared Error')

    ax2 = ax[1]
    ax2.plot(list(range(len(L))), L)
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Loss - Mean Squared Error')
    ax2.set_yscale('log')

    plt.show()