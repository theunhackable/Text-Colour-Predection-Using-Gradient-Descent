from data1 import *

X = x
Y = y
m = X.shape[1]


def initiation():
    W1 = np.random.randn(4, 3) * 0.01
    W2 = np.random.randn(1, 4) * 0.01
    b1 = np.random.rand() * 0.1
    b2 = 0
    parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return parameters


def d_params(parm_data):
    Z1 = parm_data['Z1']
    A1 = parm_data['A1']
    Z2 = parm_data['Z2']
    A2 = parm_data['A2']
    W1 = parm_data['W1']
    b1 = parm_data['b1']
    W2 = parm_data['W2']
    b2 = parm_data['b2']

    ##############
    dZ2 = A2 - Y
    dW2 = (1 / m) * (np.dot(dZ2, A1.T))
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_params(parameters):
    learning_rate = 0.0008
    #  real parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    #  difference in parameters
    params_update = d_params(parameters)
    dW1 = params_update['dW1']
    db1 = params_update['db1']
    dW2 = params_update['dW2']
    db2 = params_update['db2']
    #  Updating Parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    params_update = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params_update


def computation():
    parameters = initiation()
    for i in range(150001):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        log_probes = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = -np.sum(log_probes) / m
        if i % 10000 == 0:
            print('iteration : {}, cost : {}'.format(i, cost))

        param = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2, 'A1': A1, 'A2': A2, 'Z1': Z1, 'Z2': Z2}
        parameters = update_params(param)
    return parameters


z = computation()


def test(X):
    W1 = z['W1']
    W2 = z['W2']
    b1 = z['b1']
    b2 = z['b2']
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    if A2 > 0.5:
        print('White')
    else:
        print('black')


x = np.array([[0, 51, 102]])
test(x.T)

x = np.array([[255, 204, 255]])
test(x.T)

x = np.array([[0, 51, 0]])
test(x.T)

x = np.array([[204, 255, 255]])
test(x.T)

x = np.array([[51, 51, 0]])
test(x.T)

x = np.array([[102,255,255]])
test(x.T)
