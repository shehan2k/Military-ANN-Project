import numpy as np

# --- STEP 1: GENERATE MILITARY INTELLIGENCE (DATA) ---
# Features: [Speed (Mach), Radar Size (0-1), Altitude (k-ft)]
# Hostiles: Fast, Small, High Altitude, Heat Signature
hostiles = np.random.rand(20, 4) * [4.0, 0.9, 30, 10] + [0.5, 0.1, 5, 10]
# Friendlies: Slow, Large, Low Altitude
friendlies = np.random.rand(20, 4) * [1.0, 0.5, 10, 1] + [0.5, 0.1, 5, 1]

X = np.vstack((hostiles, friendlies))
y = np.array([1]*20 + [0]*20) # 1=Hostile, 0=Friendly

# --- STEP 2: INITIALIZE THE NEURAL NETWORK ---
weights = np.random.uniform(-1, 1, 4) 
bias = 0.0
lr = 0.1
totalerrors=0
# --- STEP 3: TRAINING (THE LEARNING PHASE) ---
for epoch in range(200):
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
print("Targeting System Online. Weights Calibrated.")


# --- STEP 4: INTERACTIVE TARGET IDENTIFICATION ---
def identify_target():
    print("\n--- AEGIS RADAR INPUT ---")
    try:
        s = float(input("Enter Speed (Mach 0-5): "))
        r = float(input("Enter Radar Size (0.1-1.0): "))
        a = float(input("Enter Altitude (k-ft): "))
        h = float(input("Enter Heat Signature: "))
        
        new_contact = np.array([s, r, a, h])
        
        # The ANN Prediction
        score = np.dot(new_contact, weights) + bias
        if score > 0:
            print(">>> WARNING: HOSTILE DETECTED. ENGAGE TARGET. <<<")
        else:
            print(">>> STATUS: FRIENDLY. CLEAR FOR APPROACH. <<<")
    except ValueError:
        print("Invalid data format.")

# Run the prediction
identify_target()