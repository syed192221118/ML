import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Example data (replace with your dataset)
data = {
    'Brand': ['Toyota', 'Honda', 'Toyota', 'Honda'],
    'Model': ['Camry', 'Accord', 'Corolla', 'Civic'],
    'Price': [18000, 28000, 15000, 22000]
}

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df.drop(columns=['Price'])
y = df['Price']

# One-hot encode categorical variables
column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
X_encoded = column_transformer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
print("Root Mean Squared Error:", rmse)
