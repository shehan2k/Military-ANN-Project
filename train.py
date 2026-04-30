import numpy as np


# --- STEP 1: GENERATE MILITARY INTELLIGENCE (DATA) ---
# Features: [Speed (Mach), Radar Size (0-1), Altitude (k-ft)]
# Hostiles: Fast, Small, High Altitude, High Heat Signature
hostiles = np.random.rand(500, 4) * [4.0, 0.9, 45, 1100] + [1.1, 0.1, 41, 801]
# Friendlies: Slow, Large, Low Altitude, Low Heat Signature
friendlies = np.random.rand(500, 4) * [1, 2.1, 10, 450] + [0.5, 0.9, 30, 350]

X = np.vstack((hostiles, friendlies))
y = np.array([1]*500 + [0]*500) # 1=Hostile, 0=Friendly

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# --- STEP 2: INITIALIZE THE NEURAL NETWORK ---

weights = np.random.uniform(-1, 1, 4) 
bias = 0.0
lr = 0.01
totalerrors=0
# --- STEP 3: TRAINING (THE LEARNING PHASE) ---

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(50):
    for i in range(len(X)):
        # Linear Equation (Dot Product)
        z = np.dot(X[i], weights) + bias
        prediction = sigmoid(z)
        
        binary_prediction = 1 if prediction > 0.5 else 0
        if binary_prediction != y[i]: 
            totalerrors += 1

        # Adjust weights based on error
        error = y[i] - prediction
        adjustment = error * prediction * (1 - prediction)
        weights += lr * adjustment * X[i]
        bias += lr * adjustment
    print(totalerrors) 
    totalerrors=0

np.savez('model_weights.npz', weights=weights, bias=bias)

print("Targeting System Online. Weights Calibrated.")


