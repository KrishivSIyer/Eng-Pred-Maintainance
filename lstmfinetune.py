import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

print("="*70)
print("LSTM FINE-TUNING ON FD002, FD003, FD004")
print("="*70)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
SEQ_LENGTH = 50

def create_sequences(X, y, seq_length=SEQ_LENGTH):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y.iloc[i+seq_length] if isinstance(y, pd.Series) else y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def create_test_sequences(X, seq_length=SEQ_LENGTH):
    """Create sequences for test data - take LAST seq_length cycles"""
    if len(X) >= seq_length:
        return np.array([X[-seq_length:]])
    else:
        # If not enough cycles, pad with first cycle repeated
        padding = np.tile(X[0], (seq_length - len(X), 1))
        return np.array([np.vstack([padding, X])])

def load_and_prepare_data(dataset_name, is_training=True):
    """Load raw data, calculate RUL, scale features"""
    
    if is_training:
        print(f"\nPreparing {dataset_name} training data...")
        file_path = f'data/raw/train_{dataset_name}.txt'
        df = pd.read_csv(file_path, sep='\s+', names=column_names, engine='python')
        df = df.dropna(how='all')
        df = df.drop_duplicates()
        
        # Calculate RUL for training data
        max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycle']
        df = df.merge(max_cycles, on='unit_number', how='left')
        df['RUL'] = df['max_cycle'] - df['time_in_cycles']
        
        # Prepare features and target
        X = df[features].values
        y = df['RUL']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LENGTH)
        
        # Split into train/validation (80/20)
        split_idx = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:split_idx]
        X_val_seq = X_seq[split_idx:]
        y_train_seq = y_seq[:split_idx]
        y_val_seq = y_seq[split_idx:]
        
        print(f"Training sequences: {len(X_train_seq)}")
        print(f"Validation sequences: {len(X_val_seq)}")
        
        return X_train_seq, X_val_seq, y_train_seq, y_val_seq, scaler
    
    else:
        print(f"\nPreparing {dataset_name} test data...")
        file_path = f'data/raw/test_{dataset_name}.txt'
        df = pd.read_csv(file_path, sep='\s+', names=column_names, engine='python')
        df = df.dropna(how='all')
        df = df.drop_duplicates()
        
        # Group by engine and process each engine separately
        test_sequences = []
        engine_rul_list = []
        
        # Load true RUL values
        rul_path = f'data/raw/RUL_{dataset_name}.txt'
        true_rul_df = pd.read_csv(rul_path, names=['RUL'])
        
        engines = df['unit_number'].unique()
        
        for engine_id in engines:
            engine_data = df[df['unit_number'] == engine_id]
            engine_features = engine_data[features].values
            
            # Get last SEQ_LENGTH cycles
            if len(engine_features) >= SEQ_LENGTH:
                seq = engine_features[-SEQ_LENGTH:]
            else:
                # Pad if not enough cycles
                padding = np.tile(engine_features[0], (SEQ_LENGTH - len(engine_features), 1))
                seq = np.vstack([padding, engine_features])
            
            test_sequences.append(seq)
            # RUL is the same for all cycles of this engine (one value per engine)
            engine_rul_list.append(true_rul_df.iloc[engine_id - 1].values[0])
        
        X_test_seq = np.array(test_sequences)
        y_test = np.array(engine_rul_list)
        
        print(f"Test engines: {len(X_test_seq)}")
        print(f"Test sequences shape: {X_test_seq.shape}")
        
        return X_test_seq, y_test

def create_base_lstm_model():
    """Create the base LSTM model"""
    model = Sequential([
        tf.keras.layers.Input(shape=(SEQ_LENGTH, len(features))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ============================================
# LOAD PRE-TRAINED BASE MODEL
# ============================================
print("\n" + "="*60)
print("STEP 1: Loading Pre-trained Base LSTM Model")
print("="*60)

# Load pre-trained base model (must already exist)
base_model = load_model('models/lstm_base_fd001.keras')
print("Base model loaded from models/lstm_base_fd001.keras")

# ============================================
# FINE-TUNE ON FD002, FD003, FD004
# ============================================
results = []

for dataset in ['FD002', 'FD003', 'FD004']:
    print("\n" + "="*60)
    print(f"STEP 2: Fine-tuning on {dataset}")
    print("="*60)
    
    # Load training data
    X_train_seq, X_val_seq, y_train_seq, y_val_seq, scaler = load_and_prepare_data(dataset, is_training=True)
    
    # Load test data
    X_test_seq, y_test = load_and_prepare_data(dataset, is_training=False)
    
    # Clone base model
    fine_tuned_model = clone_model(base_model)
    fine_tuned_model.set_weights(base_model.get_weights())
    
    # Recompile with lower learning rate
    fine_tuned_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"Fine-tuning on {dataset}...")
    history = fine_tuned_model.fit(
        X_train_seq, y_train_seq,
        epochs=30,
        batch_size=32,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate
    y_pred = fine_tuned_model.predict(X_test_seq)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{dataset} Fine-tuned Results:")
    print(f"   Test MAE: {test_mae:.2f} cycles")
    print(f"   Test RMSE: {test_rmse:.2f} cycles")
    
    # Save fine-tuned model
    fine_tuned_model.save(f'models/lstm_{dataset}_finetuned.keras')
    print(f"Saved: models/lstm_{dataset}_finetuned.keras")
    
    results.append({'Dataset': dataset, 'MAE': round(test_mae, 2), 'RMSE': round(test_rmse, 2)})

# ============================================
# RESULTS
# ============================================
print("\n" + "="*70)
print("COMPLETE LSTM RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/lstm_results.csv', index=False)
print("\nResults saved to: results/lstm_results.csv")

print("\nALL DONE! LSTM fine-tuning completed on FD002, FD003, FD004")