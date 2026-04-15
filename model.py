import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("="*60)
print("PART 3: MODEL BUILDING - STEP 1")
print("="*60)

# Load the data we saved in Part 2
X_train = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/X_train.csv')
X_test = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/X_test.csv')
y_train = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/y_train.csv')
y_test = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/y_test.csv')

# Load the scaler
scaler = joblib.load('C:/Users/krish/OneDrive/Desktop/sem2 projects/aiml/fig/scaler.pkl')

# Quick verification
print("\n✅ Data loaded successfully!")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Scaler type: {type(scaler)}")

# Look at first few rows
print("\n📋 First 2 rows of training data:")
print(X_train.head(2))

# Random forest
print("="*60)
print("STEP 2: TRAINING RANDOM FOREST")
print("="*60)

# Create the model (start simple)
rf_model = RandomForestRegressor(
    n_estimators=100,  # 100 trees
    random_state=42,    # reproducible results
    n_jobs=-1          # use all CPU cores
)

# Train the model
rf_model.fit(X_train, y_train.values.ravel())  # .ravel() flattens y to 1D array
joblib.dump(rf_model, 'rf_model.pkl')  # ← HERE!

print("✅ Model saved as 'rf_model.pkl'")

print("✅ Training complete!")

# Make predictions on test data
y_pred_rf = rf_model.predict(X_test)

# Calculate errors
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n📊 RESULTS:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f} cycles")
print(f"Root Mean Square Error (RMSE): {rmse_rf:.2f} cycles")

print("\n💡 INTERPRETATION:")
print(f"On average, Random Forest is off by {mae_rf:.1f} cycles")
