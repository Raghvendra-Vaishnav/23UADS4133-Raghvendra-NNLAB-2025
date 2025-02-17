# 2. WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.

# Step 1: Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

class MLP_XOR:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
        self.errors = []

    def forward(self, X):
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = step_function(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.final_output = step_function(self.final_input)
        return self.final_output

    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                xi = X[i:i+1]  # Select single sample
                yi = y[i:i+1]
                
                # Forward pass
                hidden_input = np.dot(xi, self.W1) + self.b1
                hidden_output = step_function(hidden_input)
                final_input = np.dot(hidden_output, self.W2) + self.b2
                final_output = step_function(final_input)
                
                # Compute error
                error = yi - final_output
                total_error += np.abs(error)
                
                # Backpropagation (Weight Update)
                self.W2 += self.learning_rate * error * hidden_output.T
                self.b2 += self.learning_rate * error
                self.W1 += self.learning_rate * np.dot(xi.T, error * self.W2.T * hidden_output * (1 - hidden_output))
                self.b1 += self.learning_rate * error * self.W2.T * hidden_output * (1 - hidden_output)
            
            self.errors.append(total_error.sum())
            # Print error at intervals
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Total Error: {total_error.sum()}")

    def predict(self, X):
        return self.forward(X)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train MLP
mlp = MLP_XOR()
mlp.train(X, y)

# Test predictions
predictions = mlp.predict(X)
accuracy = np.mean(predictions == y) * 100
print("Predictions:", predictions.flatten())
print(f"Accuracy: {accuracy:.2f}%")

# Plot error graph
plt.plot(mlp.errors)
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.title("Training Error Over Time")
plt.show()