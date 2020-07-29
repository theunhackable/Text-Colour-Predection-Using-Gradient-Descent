import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


x = np.array([[255, 255, 255],
              [153, 51, 51],
              [255, 204, 255],
              [0, 0, 0]
              ])
y = np.array([0, 1, 0, 1])
print(y)
x = x.T
w = np.zeros((3, 1))
b = 0
lr = 0.003
for i in range(5000):
    a = sigmoid(np.dot(w.T, x) + b)
    dw = (1 / 3) * (np.dot(x, (a - y).T))
    db = (1 / 3) * (np.sum(a - y))
    w = w - lr * dw
    b = b - lr * db
    cost = (-1 / 3) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))  # compute cost
    if i%1000 == 0:
        print("cost:{:.16f}".format(float(cost)))
