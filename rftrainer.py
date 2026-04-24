import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

print("="*70)
print("RANDOM FOREST TRAINING ON FD002, FD003, FD004")
print("="*70)

# Column names for raw data
column_names = [
    'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

features = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]

def preprocess_dataset(dataset_name):
    """Load raw data, calculate RUL, scale features, split train/test"""
    
    print(f"\nPreprocessing {dataset_name}...")
    
    # Load training data
    train_path = f'data/raw/train_{dataset_name}.txt'
    train_df = pd.read_csv(train_path, sep='\s+', names=column_names, engine='python')
    train_df = train_df.dropna(how='all')
    train_df = train_df.drop_duplicates()
    
    # Calculate RUL for training data
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    
    # Prepare features and target
    X = train_df[features]
    y = train_df['RUL']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/validation (80/20)
    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    X_val = X_scaled[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    # Load test data
    test_path = f'data/raw/test_{dataset_name}.txt'
    test_df = pd.read_csv(test_path, sep='\s+', names=column_names, engine='python')
    test_df = test_df.dropna(how='all')
    test_df = test_df.drop_duplicates()
    
    # Get the LAST cycle of each engine in test data
    # (This is what NASA uses for evaluation)
    last_cycles = test_df.groupby('unit_number').last().reset_index()
    X_test = last_cycles[features]
    X_test_scaled = scaler.transform(X_test)
    
    # Load true RUL values (one per engine)
    rul_path = f'data/raw/RUL_{dataset_name}.txt'
    y_test = pd.read_csv(rul_path, names=['RUL'])
    
    # Verify matching number of test engines
    print(f"Test engines: {len(X_test)}")
    print(f"RUL values: {len(y_test)}")
    
    return X_train, X_val, y_train, y_val, X_test_scaled, y_test, scaler

def train_and_evaluate(dataset_name):
    """Train Random Forest on a dataset and return results"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name}")
    print('='*60)
    
    # Preprocess
    X_train, X_val, y_train, y_val, X_test, y_test, scaler = preprocess_dataset(dataset_name)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test engines: {len(X_test)}")
    
    # Train Random Forest
    print(f"\nTraining Random Forest on {dataset_name}...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train.values.ravel())
    
    # Validate
    y_val_pred = rf_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f"Validation MAE: {val_mae:.2f} cycles")
    
    # Test on official test set (last cycle of each engine)
    y_test_pred = rf_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n{dataset_name} RESULTS:")
    print(f"   Test MAE: {test_mae:.2f} cycles")
    print(f"   Test RMSE: {test_rmse:.2f} cycles")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, f'models/rf_{dataset_name}.pkl')
    joblib.dump(scaler, f'models/scaler_{dataset_name}.pkl')
    print(f"Saved: models/rf_{dataset_name}.pkl")
    print(f"Saved: models/scaler_{dataset_name}.pkl")
    
    return {
        'Dataset': dataset_name,
        'Train_Samples': len(X_train),
        'Test_Engines': len(X_test),
        'Val_MAE': round(val_mae, 2),
        'Test_MAE': round(test_mae, 2),
        'Test_RMSE': round(test_rmse, 2)
    }

# ============================================
# RUN ON FD002, FD003, FD004 ONLY
# ============================================
all_results = []

for dataset in ['FD002', 'FD003', 'FD004']:
    result = train_and_evaluate(dataset)
    all_results.append(result)

# ============================================
# ADD FD001 RESULT MANUALLY
# ============================================
fd001_result = {
    'Dataset': 'FD001',
    'Train_Samples': 'N/A',
    'Test_Engines': 'N/A',
    'Val_MAE': 'N/A',
    'Test_MAE': 29.70,
    'Test_RMSE': 41.52
}
all_results.insert(0, fd001_result)

# ============================================
# DISPLAY COMPLETE RESULTS
# ============================================
print("\n" + "="*70)
print("COMPLETE RANDOM FOREST RESULTS - ALL DATASETS")
print("="*70)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Save results
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/rf_all_datasets.csv', index=False)
print("\nResults saved to: results/rf_all_datasets.csv")

# Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
for r in all_results:
    print(f"{r['Dataset']}: MAE = {r['Test_MAE']:.2f} cycles, RMSE = {r['Test_RMSE']:.2f} cycles")

print("\nALL DONE! Random Forest trained on FD002, FD003, FD004")
print("FD001 result added manually: 29.70 MAE")