import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style("darkgrid")
# %matplotlib inline
df = pd.read_csv("iris.data.csv",names=['sepal_length','sepal_width','petal_length','petal_width','result'],header=None)
df = df.iloc[[1, 51, 101]]
"""Data preparation:
---
"""
# one-hot encoding
y = pd.get_dummies(df.result).values
N = y.size
x = df.drop('result', axis=1).values
print("Shape of the output value: \n\n",y.shape)
print("\nOne hot encoding Output\n\n ",y)
print("\n Labeled Feature",x)
print("Show the result with respect to MSE")
"""Activation Function:
---
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
Hyperparameters:
---
"""

learning_rate = 0.1

n_input = 4
n_hidden = 2
n_output = 3

np.random.seed(10)

weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden))   # (4, 2)
weights_2 = np.random.normal(scale=0.5, size=(n_hidden, n_output))  # (2, 3)
#print(weights_1)
#print("")
#print(weights_2)

"""Feedforward:
---
"""

# feedforward
hidden_layer_inputs = np.dot(x, weights_1)
hidden_layer_outputs = sigmoid(hidden_layer_inputs)

output_layer_inputs = np.dot(hidden_layer_outputs, weights_2)
output_layer_outputs = sigmoid(output_layer_inputs)

#error at output layer
#Error calculation at the output layer
output_layer_error = output_layer_outputs - y
#print(output_layer_error)

"""Backpropagation:
---
"""

# backpropagation
output_layer_delta = output_layer_error * output_layer_outputs * (1 - output_layer_outputs)

hidden_layer_error = np.dot(output_layer_delta, weights_2.T)
hidden_layer_delta = hidden_layer_error * hidden_layer_outputs * (1 - hidden_layer_outputs)

"""Weights updated:
---
"""

# weight updates
weights_2_update = np.dot(hidden_layer_outputs.T, output_layer_delta) / N
weights_1_update = np.dot(x.T, hidden_layer_delta) / N

weights_2 = weights_2 - learning_rate * weights_2_update
weights_1 = weights_1 - learning_rate * weights_1_update
#print(weights_2)
#print("")
#print(weights_1)

"""Mean Saqure Error before weight update:
---
"""

mse_1 = ((output_layer_outputs - y)**2).sum() / (2*N)
print("Mean saqure before updated weights",mse_1)

"""Mean Saqure Error after updated weights:
---
"""

# feedforward
hidden_layer_inputs = np.dot(x, weights_1)
hidden_layer_outputs = sigmoid(hidden_layer_inputs)

output_layer_inputs = np.dot(hidden_layer_outputs, weights_2)
output_layer_outputs = sigmoid(output_layer_inputs)

mse_2 = ((output_layer_outputs - y)**2).sum() / (2*N)
print("Mean saqure after updated weights",mse_2)
