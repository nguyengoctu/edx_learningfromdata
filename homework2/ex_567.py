from matplotlib import pyplot as plt
from random import uniform
from random import randint
import numpy as np

def calculate_sign(w, x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

def draw_line(a, b, col='black'):
    x = np.r_[-1:1:100*1j]
    plt.plot(x, a * x + b, color=col)

# scatter point x (depend on sign)
def scatter(x):
    if calculate_sign(theta, x[j]) >= 1:
        plt.scatter(x[j][1], x[j][2], color='red')
    else:
        plt.scatter(x[j][1], x[j][2], color='blue')

# number of points
n = 10
num_of_runs = 10000
total_E_in = 0.0
dimension_of_x = 3
total_E_out = 0.0
size_of_x_out = 1000
total_iterations = 0

for i in range(num_of_runs):
    # matplotlib initiation
    # plt.ylim([-1, 1])
    # plt.xlim([-1, 1])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.text(0.5, 0.8, "N = " + str(n), fontsize=12)
    # plt.text(0.5, 0.6, "Purple line: f", fontsize=12)
    # plt.text(0.5, 0.5, "Black line: g", fontsize=12)
    
    # Randomly create two points in [-1, 1] x [-1, 1]
    x1 = uniform(-1, 1)
    y1 = uniform(-1, 1)
    x2 = uniform(-1, 1)
    y2 = uniform(-1, 1)

    theta = np.array([x1 * (y2 - y1) + y1 * (x1 - x2), y2 - y1, x1 - x2])

    # draw f
    # a = np.r_[-1:1:100*1j]
    # draw_line((y1 - y2) / (x1 - x2), (y1 - y2) * x1 / (x1 - x2) - y1, 'purple')

    # Initialize w  
    # w = np.zeros(dimension_of_x)
    
    # Generate points
    x = np.ones((n, 3))
    f = np.zeros(n)
    g = np.zeros(n)
    for j in range(n):
        x[j][1] = uniform(-1, 1)
        x[j][2] = uniform(-1, 1)
        # scatter(x)
        # f[j] = calculate_sign(theta, x[j])

    # Calculate w
    # w = pseudo-inverse(x) * f
    x_transpose = np.transpose(x)
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_transpose, x)), x_transpose), f)
    # ln, = plt.plot(a, (-w[0] - w[1] * a) / w[2], color='black')
    # plt.pause(1)
    # ln.remove()
    # miss_classified_points = 0
    # for j in range(n):
    #     if f[j] != calculate_sign(w, x[j]):
    #         miss_classified_points += 1
    # total_E_in += miss_classified_points / n
    #
    # x_out = np.ones((size_of_x_out, 3))
    # for j in range(size_of_x_out):
    #     x_out[j][1] = uniform(-1, 1)
    #     x_out[j][2] = uniform(-1, 1)
    # miss_classified_points = 0
    # for j in range(size_of_x_out):
    #     if calculate_sign(theta, x_out[j]) != calculate_sign(w, x_out[j]):
    #         miss_classified_points += 1
    # total_E_out += miss_classified_points / size_of_x_out

# print("Avg E_in: ", total_E_in / num_of_runs)
# print("Avg E_out: ", total_E_out / num_of_runs)
    # plt.plot(a, (-w[0] - w[1] * a) / w[2], color='black')
    # plt.show()
    iterations = 0
    flag = True
    while flag:
        x_temp = x[:]
        while x_temp[:, 0].size != 0:
            index = randint(0, x_temp[:, 0].size - 1)
            f = calculate_sign(theta, x_temp[index])
            g = calculate_sign(w, x_temp[index])
            if f != g:
                # Adjust w
                w = w + f * x_temp[index]
                iterations += 1
                # ln, = plt.plot(a, (-w[0] - w[1] * a) / w[2], color='black')
                # txt = plt.text(0.5, 0.7, "Iteration: " + str(iterations), fontsize=12)
                # plt.pause(0.25)
                # ln.remove()
                # txt.remove()
                break
            else:
                x_temp = np.delete(x_temp, (index), axis=0)

        # Converged
        if x_temp[:, 0].size == 0:
            flag = False
    # plt.show()
    print("Test ", i + 1, " took ", iterations, " iterations to converge")
    total_iterations += iterations
print()
print("Avg iterations per test: ", total_iterations / num_of_runs)
