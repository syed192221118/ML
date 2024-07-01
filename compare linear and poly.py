import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = 2 * np.pi * np.random.rand(100, 1)
y = np.sin(X) + np.random.randn(100, 1) * 0.1  # Adding noise to the sine function

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Fit polynomial regression model (degree=2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Evaluate models
X_test_poly = poly_features.transform(X_test)
y_pred_lin = lin_reg.predict(X_test)
y_pred_poly = poly_reg.predict(X_test_poly)
mse_lin = mean_squared_error(y_test, y_pred_lin)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Plot the data and regression lines
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred_lin, color='red', label=f'Linear Regression (MSE={mse_lin:.2f})')
plt.plot(np.sort(X_test, axis=0), np.sort(y_pred_poly, axis=0), color='green', label=f'Polynomial Regression (degree=2) (MSE={mse_poly:.2f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of Linear and Polynomial Regression')
plt.legend()
plt.show()
