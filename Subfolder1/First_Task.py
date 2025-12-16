import csv
import numpy as np

X = []
y = []
X_test = []
y_test = []
predictions = []
correct = 0

with open("movie_train_data_edited.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    for row in reader:
        action = float(row[0])
        romance = float(row[1])
        liked = int(row[2])

        X.append([action, romance])
        y.append(liked)

X = np.array(X)
y = np.array(y).reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_vgd(X, y, lr=0.01, epochs=2000):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    for epoch in range(epochs):
        # Forward pass
        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        # -------- MSE LOSS --------
        cost = np.mean((A - y) ** 2) / 2

        # -------- MSE GRADIENTS --------
        dZ = (A - y) * A * (1 - A)   # derivative of sigmoid + mse
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ) / m

        # Update parameters
        W -= lr * dW
        b -= lr * db

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | MSE = {cost:.4f}")

    return W, b

def train_momentum(X, y, lr=0.01, epochs=2000, beta=0.9):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    # Momentum variables (velocity terms)
    vW = np.zeros((n, 1))    # momentum for weights
    vb = 0                   # momentum for bias

    for epoch in range(epochs):

        # Forward pass
        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        # MSE loss
        cost = np.mean((A - y) ** 2) / 2

        # Gradients for MSE + sigmoid
        dZ = (A - y) * A * (1 - A)
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ) / m

        # -------- MOMENTUM UPDATE --------
        vW = beta * vW + (1 - beta) * dW
        vb = beta * vb + (1 - beta) * db

        # Update parameters
        W -= lr * vW
        b -= lr * vb

        # Print progress
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | MSE = {cost:.4f}")

    return W, b

def train_nesterov(X, y, lr=0.01, epochs=2000, beta=0.9):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    # Velocity terms
    vW = np.zeros((n, 1))
    vb = 0

    for epoch in range(epochs):

        # -------- LOOK AHEAD STEP --------
        W_lookahead = W - beta * vW
        b_lookahead = b - beta * vb

        # Forward pass using look-ahead weights
        Z = np.dot(X, W_lookahead) + b_lookahead
        A = sigmoid(Z)

        # MSE loss
        cost = np.mean((A - y) ** 2) / 2

        # Gradient for MSE + sigmoid
        dZ = (A - y) * A * (1 - A)
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ) / m

        # -------- UPDATE VELOCITIES (momentum) --------
        vW = beta * vW + lr * dW
        vb = beta * vb + lr * db

        # -------- UPDATE PARAMETERS --------
        W -= vW
        b -= vb

        # Track progress
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | MSE = {cost:.4f}")

    return W, b

#-----------------------------------------main----------------------------------------------------
model = input("Choose optimization method (vgd-v/momentum-m/nesterov-n): ").strip().lower()

while model not in ['v', 'm', 'n']:
    model = input("Invalid choice. Please choose 'v' for VGD, 'm' for Momentum, or 'n' for Nesterov: ").strip().lower()

if model == 'v':
    W, b = train_vgd(X, y, lr=0.36, epochs=5000)
elif model == 'n':
    W, b = train_nesterov(X, y, lr=0.1, epochs=2000, beta=0.9)  
else:
    W, b = train_momentum(X, y, lr=0.1, epochs=2000, beta=0.9)

with open("movie_test_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  

    for row in reader:
        action = float(row[0])
        romance = float(row[1])
        liked = int(row[2])

        X_test.append([action, romance])
        y_test.append(liked)

X_test = np.array(X)
y_test = np.array(y).reshape(-1, 1)

def predict(action, romance):
    x = np.array([[action, romance]])
    prob = sigmoid(np.dot(x, W) + b)[0][0]
    return 1 if prob >= 0.5 else 0

for i in range(len(X_test)):
    pred = predict(X_test[i][0], X_test[i][1])
    predictions.append(pred)

for i in range(len(X_test)):
    correct += 1 if predictions[i] == y_test[i][0] else 0

print("Predictions:", predictions)
print("Actual:", y_test.flatten().tolist())
print(f"Accuracy: {correct / len(y_test) * 100:.2f}%")
