# Assignment - 3
# WAP in Python to implement simple linear regression
# using gradient descent to estimate slope and intercept
# Use learning rate = 0.01 and run for 1000 iterations

# Sample dataset
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

n = len(x)

# Initialize slope (m) and intercept (c)
m = 0.0
c = 0.0

learning_rate = 0.01
iterations = 1000

# Gradient Descent
for iteration in range(iterations):

    dm = 0
    dc = 0

    for i in range(n):
        y_cap = m * x[i] + c
        error = y_cap - y[i]

        dm += error * x[i]
        dc += error

    # Partial derivatives
    dm = (2 / n) * dm
    dc = (2 / n) * dc

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

# Final values
print("EstimatedSlope(m):", m)
print("EstimatedIntercept(c):", c)