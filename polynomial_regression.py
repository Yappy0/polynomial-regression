import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.formats.format import _trim_zeros_single_float

data = pd.read_csv("test_data.csv")
x = np.asarray(data['x'].values.tolist())
y = np.asarray(data['y'].values.tolist())
x.reshape(-1, 1)
y.reshape(len(y), 1)

#Generates a matrix of x_n = x^n, treating each x_n as an independent feature
def poly_features(features, x):
    data = pd.DataFrame(np.zeros((x.shape[0],features + 1)))
    for i in range(0, features + 1):
        data.iloc[:, i] = (x ** i).reshape(-1,1)
    x_poly = np.array(data.values.tolist())
    return x_poly

#Splits the data into test and train
def split_data(x, y, test_size = 0.2, random_state = 0):
    np.random.seed(random_state)
    indices = np.random.permutation(len(x))
    data_test_size = int(x.shape[0] * test_size)

    train_indices = indices[data_test_size:]
    test_indices = indices[:data_test_size]

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

class polynomialRegression:

    def predict(self, weights, x_sample):
        return sum(weights * x_sample) #W_0 * X0 + W_1 * X_1 ...

    def forward(self, x, y, w):
        y_pred = self.predict(w, x)
        loss = ((y_pred - y) ** 2) / 2 #Using MSE for error
        return loss, y_pred

    def update_weights(self, x, y_pred, y_true, w, learning_rate):
        for i in range(x.shape[0]):
            #RHS is learning rate times derivative of loss function with respect to w
            #Trying to get loss' = 0
            w[i] -= learning_rate * ((y_pred - y_true) * x[i])
        return w

    def train(self, x, y, epochs = 10, learning_rate = 0.001, random_state = 0):

        print("TRAINING")
        print("--------")

        #Initialize weights
        num_rows = x.shape[0]
        num_cols = x.shape[1]
        w = np.random.randn(1,num_cols) / np.sqrt(num_rows)
        w = w[0]

        train_loss = []
        train_indices = [i for i in range(num_cols)]

        for j in range(epochs):
            cost = 0
            np.random.seed(random_state)
            np.random.shuffle(train_indices)
            for i in train_indices:
                loss, y_pred = self.forward(x[i], y[i], w)
                cost += loss
                w = self.update_weights(x[i], y_pred, y[i], w, learning_rate)
            train_loss.append(cost)
            if j % 100 == 99:
                print(f"{j + 1}: {cost}")
        return w, train_loss

    def test(self, x_test, y_test, w):

        #Testing
        test_pred = []
        test_loss = []
        test_indices = [i for i in range(x_test.shape[0])]
        for i in test_indices:
            loss, y_pred = self.forward(x[i], y[i], w)
            test_pred.append(y_pred)
            test_loss.append(loss)
        return test_pred, test_loss

x = poly_features(int(input()), x)
x_train, x_test, y_train, y_test = split_data(x, y)
print(x_train.shape)
regressor = polynomialRegression()
weights, train_loss = regressor.train(x_train, y_train, epochs=200000, learning_rate = 0.00005)
print("Weights:", weights)
print("Final Loss:", train_loss[-1])
