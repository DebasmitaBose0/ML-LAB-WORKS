#WAP in Python to:
#Implement Gradient Descent algorithm for Simple Linear Regression to find optimal m and c values.
#This involves:
#i) Initializing m and c
#ii) Iteratively updating m and c using partial derivative of the cost function with respect to m and c and specified learning rate
#iii) Running the iteration for a fixed number of epochs (take epochs as 20)
#Kindly note: The above programme needs to be implemented without any inbuilt function/s.
# Assignment - 2
# Implement Gradient Descent Algorithm for Simple Linear Regression
# to find optimal m (slope) and c (intercept)
# Without using any inbuilt functions

# Sample dataset
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

n = len(x)

# 1) Initialize parameters
m = 0.0
c = 0.0

learning_rate = 0.01
epochs = 20

# 2) Gradient Descent Loop
for epoch in range(epochs):

    dm = 0
    dc = 0
    ss_error = 0

    for i in range(n):
        y_cap = m * x[i] + c
        error = y_cap - y[i]

        ss_error += error * error

        dm += error * x[i]
        dc += error

    # 3) Partial derivatives
    dm = (2 / n) * dm
    dc = (2 / n) * dc

    # 4) Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    print("Epoch:", epoch + 1)
    print("m =", m, "c =", c)
    print("SumOfSquaredError =", ss_error)
    print("--------------------------------------")