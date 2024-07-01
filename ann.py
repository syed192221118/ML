import numpy as np

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights and biases
input_layer_neurons = X.shape[1]
hidden_layer_neurons = 2
output_neurons = 1

# Random weights and biases initialization
np.random.seed(42)
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))

# Learning rate
lr = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(X, wh, bh, wo, bo):
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, wo) + bo
    output = sigmoid(output_layer_input)
    return hidden_layer_activation, output

def backpropagation(X, y, hidden_layer_activation, output, wh, bh, wo, bo, lr):
    # Calculate the error
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_layer_error = output_delta.dot(wo.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_activation)

    # Update weights and biases
    wo += hidden_layer_activation.T.dot(output_delta) * lr
    bo += np.sum(output_delta, axis=0, keepdims=True) * lr
    wh += X.T.dot(hidden_layer_delta) * lr
    bh += np.sum(hidden_layer_delta, axis=0, keepdims=True) * lr

    return wh, bh, wo, bo

# Number of epochs
epochs = 10000

for epoch in range(epochs):
    hidden_layer_activation, output = forward_propagation(X, wh, bh, wo, bo)
    wh, bh, wo, bo = backpropagation(X, y, hidden_layer_activation, output, wh, bh, wo, bo, lr)

    if (epoch+1) % 1000 == 0:
        loss = np.mean(np.square(y - output))
        print(f'Epoch {epoch+1}, Loss: {loss}')

# Test the trained neural network
hidden_layer_activation, output = forward_propagation(X, wh, bh, wo, bo)
print("Predicted Output:")
print(output)
