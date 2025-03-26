import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

# Load data
data = pd.read_csv("data/melbourne_data.csv")

# Clean and prepare data
data = data.dropna()
data = data.rename(columns={'Price': 'target'})

# Handle some outliers
for i in range(len(data)):
    if data.iloc[i, 2] > 10000000:  # Limitting high prices
        data.iloc[i, 2] = 10000000

# Create features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
if not os.path.exists("processed"):
    os.makedirs("processed")

with open("processed/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("processed/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("processed/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("processed/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Dane zostały przetworzone i zapisane!")

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("processed/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
from sklearn.metrics import mean_squared_error, r2_score
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plot predictions vs actual
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predictions vs Actual Prices')
plt.savefig("predictions_vs_actual.png")
