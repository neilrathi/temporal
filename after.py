import numpy as np

# Define the number of simulations
num_simulations = 5000000
count = 0
i = 0

# Perform simulations
while i < num_simulations:
    # Generate random values for X1, X2, X3, and X4
    X1, X2, X3, X4 = np.random.uniform(0, 1, size=4)

    # Calculate A1, A2, B1, and B2
    A1 = min(X1, X2)
    A2 = max(X1, X2)
    B1 = min(X3, X4)
    B2 = max(X3, X4)

    if not B1 < A2:
        continue

    if A1 < B1:
        count += 1

    i += 1

# Estimate the probability
probability = count / num_simulations
print("Estimated probability:", probability)