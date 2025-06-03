import numpy as np
import matplotlib.pyplot as plt
import io

x = np.array([1, 2, 3, 4])  # This is the input features
y = np.array([1, 2, 3, 4])  # This is the training set or output variable
m = x.shape[0]  # Stores the length of x in m


def numpy_array(*args):
    out = []
    for _ in args:
        out.append(np.array(_))
    return out


def compute_cost(x, y, w, b):  # funciton to compute cost
    sum = 0
    m = x.shape[0]
    for i in range(m):
        f = w * x[i] + b  # calculate predicted value f
        sum = sum + (f - y[i]) ** 2
    j = (1 / (2 * m)) * sum  # j is the cost
    return j


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f = w * x[i] + b
        dj_db_i = f - y[i]
        dj_dw_i = (f - y[i]) * x[i]
        dj_db += dj_db_i
        dj_dw = dj_dw_i
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db


def gradient_descent(
    x, y, w_in, b_in, alpha, num_iters, cost_funtion, gradient_funciton
):
    w = w_in
    b = b_in
    cost_hist = []
    param_hist = []
    for i in range(num_iters):
        cost = compute_cost(x, y, w, b)
        cost_hist.append(cost)
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        param_hist.append([w, b])

    return w, b, cost_hist, param_hist


w_in, b_in = 0, 0  # intial values
iterations = 10000  # number of iterations to perform gradient descent
lr = 0.001  # learning rate
w, b, h, p = gradient_descent(
    x, y, w_in, b_in, lr, iterations, compute_cost, compute_gradient
)
print(f"w&b: {w,b}")
print(float(x) for x in h)
print(float(x) for x in p)


def plot():
    buf = io.BytesIO()
    plt.subplot(2, 1, 1)
    plt.plot([x[0] for x in p], label="w")
    plt.plot([x[1] for x in p], label="b")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(h, label="cost")
    plt.legend()
    plt.savefig(buf, format=format)
    plt.close()
    buf.seek(0)
    return buf.read()
