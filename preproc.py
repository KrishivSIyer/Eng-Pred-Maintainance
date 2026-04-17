# from eda import *
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Think about:
# - What function loads CSV files?
# - How do you look at the first few rows?
# - How do you see all column names?
# - How do you check basic info about the data?

df = pd.read_csv("C:/Users/krish/OneDrive/Desktop/sem2 projects/aiml/fig/engine_data_with_rul.csv")
print(df)
print(df.columns)

settings_cols = ['setting_1', 'setting_2', 'setting_3']
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
target_col = ['RUL']

all_cols = settings_cols + sensor_cols + target_col
features_cols = settings_cols + sensor_cols

grouped_df = df[all_cols]
X = df[features_cols]
Y = df[target_col]

columns_dict = {
    **{col: 'Settings' for col in settings_cols},
    **{col: 'Sensors' for col in sensor_cols},
    **{col: 'Target' for col in target_col}
}

grouped_df.columns = pd.MultiIndex.from_tuples(
    [(columns_dict[col], col) for col in grouped_df.columns]
)
print(grouped_df.head())

print('Features:')
print(X.head())

print('Target:')
print(Y.head())

# Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Shape: {X_scaled_df.shape}")
print(f"Mean of first feature: {X_scaled_df.iloc[:,0].mean():.2f}")  # Should be ~0
print(f"Std of first feature: {X_scaled_df.iloc[:,0].std():.2f}")    # Should be ~1

# Training/Testing split

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled_df,  # Your scaled features
    Y,            # Your target (RUL)
    test_size=0.2,  # 20% for testing
    random_state=42  # So we get same split every time
)

# Verify sizes
print(f"Total samples: {len(X_scaled_df)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X_scaled_df)*100:.0f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X_scaled_df)*100:.0f}%)")

# Save the splits
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
Y_train.to_csv('y_train.csv', index=False)
Y_test.to_csv('y_test.csv', index=False)


joblib.dump(scaler, 'scaler.pkl')

print(" All files saved!")
print("   - X_train.csv, X_test.csv")
print("   - y_train.csv, y_test.csv")
print("   - scaler.pkl (for scaling new data later)")
