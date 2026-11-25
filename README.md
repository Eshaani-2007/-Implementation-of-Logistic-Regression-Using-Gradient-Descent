# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Python libraries.
2.Initialize the dataset, weights, bias and learning rate.
3.Apply the sigmoid function and train the model using Gradient Descent.
4.Predict the output and calculate the accuracy. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# Input dataset
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize parameters
w = 0
b = 0
learning_rate = 0.1
epochs = 1000
m = len(y)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(epochs):
    z = w * X + b
    y_pred = sigmoid(z)

    dw = (1/m) * np.sum((y_pred - y) * X)
    db = (1/m) * np.sum(y_pred - y)

    w = w - learning_rate * dw
    b = b - learning_rate * db

# Prediction
z = w * X + b
y_pred = sigmoid(z)

predicted = (y_pred >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predicted == y) * 100

print("Actual Output:", y)
print("Predicted Output:", predicted)
print("Accuracy:", accuracy, "%")

# Plot graph
plt.scatter(X, y)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:

<img width="862" height="650" alt="image" src="https://github.com/user-attachments/assets/b4ca3263-8381-4635-af5a-1cc1764623e0" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

