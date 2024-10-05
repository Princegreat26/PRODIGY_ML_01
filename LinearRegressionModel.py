import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# Step 2: Preprocess the data
# Select relevant features (for simplicity, we'll use a subset of features)
features = ['GrLivArea', 'YearBuilt', 'BedroomAbvGr', 'FullBath', 'OverallQual']
X = train_data[features]
y = train_data['SalePrice']

# Handle missing values by filling them with the median
X.fillna(X.median(), inplace=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Step 7: Display the coefficients
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Example of making a prediction on new data
new_data = np.array([[1500, 2000, 3, 2, 7]])  # Example: 1500 sq ft, built in 2000, 3 bedrooms, 2 bathrooms, quality 7
new_data_scaled = scaler.transform(new_data)
predicted_price = model.predict(new_data_scaled)
print(f'Predicted Price: {predicted_price[0]}')