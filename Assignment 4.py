#Write a Python program to implement Simple Linear Regression from scratch using two different optimization methods — Ordinary Least Squares (Closed-form) and Gradient Descent (Iterative) — and evaluate their performance using five key metrics.
#1. Data Generation
#Generate a synthetic dataset with a linear relationship and some added noise:
#Generate 100 random values for x between 0 and 50.
#Compute:
#𝑦=3x+7+noise
#y=3x+7+noise
#where noise is a random Gaussian distribution.
#2. Implementation (No Libraries)
#You are prohibited from using scikit-learn for the model or metrics.
#You may use NumPy for operations and Matplotlib for plotting.
#3. Implement the Following Functions
#Function 1: fit_ols(x, y)
#Returns slope (m) and intercept (c) using closed-form equations.
#Function 2: fit_gd(x, y, learning_rate, epochs)
#Returns m and c after n iterations using gradient descent.
#Function 3: evaluate_metrics(y_true, y_pred, n_features)
#Returns a dictionary of the five metrics.
#4. Metrics to Calculate (for both models)
#MAE (Mean Absolute Error)
#MSE (Mean Squared Error)
#RMSE (Root Mean Squared Error)
#R² Score
#Adjusted R² Score
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Data Generation
# ===============================
np.random.seed(0)

x = np.random.uniform(0, 50, 100)
noise = np.random.normal(0, 25, 100)
y = 2 * x + 5 + noise


# ===============================
# 2. OLS (Closed Form Solution)
# ===============================
def fit_ols(x, y):

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    c = y_mean - m * x_mean

    return m, c


# ===============================
# 3. Gradient Descent
# ===============================
def fit_gd(x, y, learning_rate=0.01, epochs=1000):

    m = 0
    c = 0
    n = len(x)

    for _ in range(epochs):

        y_pred = m * x + c

        dm = (2 / n) * np.sum(x * (y_pred - y))
        dc = (2 / n) * np.sum(y_pred - y)

        m = m - learning_rate * dm
        c = c - learning_rate * dc

    return m, c


# ===============================
# 4. Evaluation Metrics
# ===============================
def evaluate_metrics(y_true, y_pred, n_features):

    n = len(y_true)

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj R2": adj_r2
    }


# ===============================
# 5. Train Both Models
# ===============================
m_ols, c_ols = fit_ols(x, y)
m_gd, c_gd = fit_gd(x, y)

y_pred_ols = m_ols * x + c_ols
y_pred_gd = m_gd * x + c_gd

metrics_ols = evaluate_metrics(y, y_pred_ols, 1)
metrics_gd = evaluate_metrics(y, y_pred_gd, 1)


# ===============================
# 6. Print Comparison Table
# ===============================
print(f"{'Metric':<10} | {'OLS':>10} | {'Gradient Descent':>20}")
print("-" * 50)

for key in metrics_ols:
    print(f"{key:<10} | {metrics_ols[key]:>10.4f} | {metrics_gd[key]:>20.4f}")


# ===============================
# 7. Plot Result
# ===============================
plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, y_pred_ols, color="red", label="OLS Line")
plt.plot(x, y_pred_gd, color="green", label="GD Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Simple Linear Regression from Scratch")
plt.legend()

plt.show()