import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 4)
y = 50 * X[:, 0] + 20 * X[:, 1] + 100 * X[:, 2] + 10 * X[:, 3] + np.random.normal(0, 1000, 1000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
test_rmse = np.sqrt(np.mean((y_test - rf_model.predict(X_test))**2))
print(f"Testing RMSE: {test_rmse:.2f}")

# Predict for a new data point
new_data = np.array([[0.5, 0.6, 0.7, 0.8]])  # Example features for a new data point
predicted_price = rf_model.predict(new_data)
print(f"Predicted price for new mobile: ${predicted_price[0]:.2f}")
