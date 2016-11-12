from random import randint, uniform
from matplotlib import pyplot as plt
from scipy import optimize
from sklearn import svm
import numpy as np
import sys

# Generate data
def generate_points(n):
    x = np.random.uniform(-1, 1, (n, 3))
    x[:, 0] = 1
    return x


def calculate_sign(data, f):
    y = np.dot(data, f)
    for i in range(y.size):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = -1
    return y


def PLA(data, y):
    # Initialize w
    w = np.zeros(3)
    flag = True
    while flag:
        temp = data[:]
        f = y[:]
        g = calculate_sign(data, w)
        while temp[:, 0].size != 0:
            i = randint(0, temp[:, 0].size - 1)
            if f[i] != g[i]:
                w += f[i] * temp[i]
                break
            else:
                temp = np.delete(temp, (i), axis=0)
                f = np.delete(f, i)
                g = np.delete(g, i)
        if temp[:, 0].size == 0:
            flag = False
    return w


def SVM(x, y, n):
    # Construct quadratic coefficients
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = y[i] * y[j] * (np.dot(x[i], x[j]) - 1)
    I = np.zeros(n) - 1
    
    def func(alpha):    
        return 0.5 * np.dot(np.dot(alpha, H), alpha) + np.dot(I, alpha)

    def con(alpha):
        return np.dot(alpha, y)

    cons = {'type': 'eq', 'fun': con}

    w = np.zeros(3)
    alpha = optimize.minimize(func, np.zeros(n), constraints=cons)['x']
    # print(alpha)
    for i in range(n):
        if alpha[i] > 0:
            w += y[i] * alpha[i] * x[i]

    w[0] = 0.
    for i in range(n):
        if alpha[i] > 0:
            w[0] = 1 / y[i] - np.dot(w, x[i])
            break
    return w
    
    
def accuracy(a, b):
    count = 0
    for i in range(a.size):
        if a[i] * b[i] < 0:
            count += 1
    return count / a.size


def scatter(x, y):
    for i in range(y.size):
        if y[i] >= 0:
            plt.scatter(x[i][1], x[i][2], color='red')
        else:
            plt.scatter(x[i][1], x[i][2], color='blue')


def draw_line(a, b, col='black'):
    x = np.r_[-1:1:100*1j]
    plt.plot(x, a * x + b, color=col)
    

# plt.ylim([-1, 1])
# plt.xlim([-1, 1])
# plt.xlabel('x')
# plt.ylabel('y')

n = 100

count = 0
support_vectors = 0
for test in range(1000):
    x = np.ones((n, 3))
    y = np.ones(n)
    f = np.zeros(3)
    a = np.r_[-1:1:100*1j]

    # Reset if all points are on one side of the line
    while True:
        # Create target function
        x1 = uniform(-1, 1)
        x2 = uniform(-1, 1)
        y1 = uniform(-1, 1)
        y2 = uniform(-1, 1)

        f = np.array([x1 * (y1 - y2) + y1 * (x2 - x1), y2 - y1, x1 - x2])

        x = generate_points(n)
        y = calculate_sign(x, f)
        if abs(np.sum(y)) < n:
            break
    # scatter(x, y)

    # Plot target function
    # draw_line((y1 - y2) / (x1 - x2), (y2 - y1) * x1 / (x1 - x2) + y1, 'purple')

    # Plot g_PLA
    g_PLA = PLA(x, y)
    # plt.plot(a, (-g_PLA[0] - g_PLA[1] * a) / g_PLA[2], color='black')

    clf = svm.SVC(C=sys.maxsize, kernel='linear')
    clf.fit(x, y)

    g_SVM = np.zeros(3)
    g_SVM[1] = clf.coef_[0][1]
    g_SVM[2] = clf.coef_[0][2]
    g_SVM[0] = clf.intercept_[0]
    support_vectors += len(clf.support_vectors_[:, 1])
    # w = SVM(x, y, n)
    # plt.plot(a, (-g_SVM[0] - g_SVM[1] * a) / g_SVM[2], color='green')

    x_out = generate_points(1000)
    y_out = calculate_sign(x_out, f)
    #scatter(x_out, y_out)

    y_PLA = calculate_sign(x_out, g_PLA)
    y_SVM = calculate_sign(x_out, g_SVM)

    if accuracy(y_out, y_SVM) < accuracy(y_out, y_PLA):
        count += 1
    # print("E_out_SVM: ", accuracy(y_out, y_SVM))
    # print("E_out_PLA: ", accuracy(y_out, y_PLA))
    print("Test", test, " is done")
    print("There is ", len(clf.support_vectors_[:, 1]), " support vectors")
    print()
# plt.show()
print(count * 100 / 1000)
print("Avg support vectors: ", support_vectors / 1000)