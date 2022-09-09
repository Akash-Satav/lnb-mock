import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("E:\PycharmProjects\classproject\ML algos\data.txt", delimiter = ",")
X = data[:, 0]
Y = data[:, 1].reshape(X.size, 1)
X = np.vstack((np.ones((X.size, )), X)).T
print(X.shape)
print(Y.shape)
plt.scatter(X[:, 1], Y)
#plt.show()

def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((2, 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(X, theta)
    cost = (1/(2*m))*np.sum(np.square(y_pred - Y))
    d_theta = (1/m)*np.dot(X.T, y_pred - Y)
    theta = theta - learning_rate*d_theta
    cost_list.append(cost)
    return theta, cost_list

iteration = 100
learning_rate = 0.00000005
theta, cost_list = model(X, Y, learning_rate = learning_rate,
iteration = iteration)

new_houses = np.array([[1, 1547], [1, 1896], [1, 1934], [1,
2800], [1, 3400], [1, 5000]])
for house in new_houses :
    print("Our model predicts the price of house with",
    house[1], "sq. ft. area as : $", round(np.dot(house, theta)[0],2))