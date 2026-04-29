import numpy as np


# --- STEP 1: GENERATE MILITARY INTELLIGENCE (DATA) ---
# Features: [Speed (Mach), Radar Size (0-1), Altitude (k-ft)]
# Hostiles: Fast, Small, High Altitude, Heat Signature
hostiles = np.random.rand(50, 4) * [4.0, 0.9, 30, 10] + [0.5, 0.1, 5, 10]
# Friendlies: Slow, Large, Low Altitude
friendlies = np.random.rand(50, 4) * [1.0, 0.9, 10, 1] + [0.5, 0.5, 5, 1]

X = np.vstack((hostiles, friendlies))
y = np.array([1]*50 + [0]*50) # 1=Hostile, 0=Friendly

# --- STEP 2: INITIALIZE THE NEURAL NETWORK ---

weights = np.random.uniform(-1, 1, 4) 
bias = 0.0
lr = 0.1
totalerrors=0
# --- STEP 3: TRAINING (THE LEARNING PHASE) ---
for epoch in range(20):
    for i in range(len(X)):
        # Linear Equation (Dot Product)
        z = np.dot(X[i], weights) + bias
        prediction = 1 if z > 0 else 0
        
        # Adjust weights based on error
        error = y[i] - prediction
        if error != 0: 
            totalerrors+= 1
        weights += lr * error * X[i]
        bias += lr * error
    print(totalerrors) 
    totalerrors=0

np.savez('model_weights.npz', weights=weights, bias=bias)

print("Targeting System Online. Weights Calibrated.")


