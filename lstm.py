
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

print("="*60)
print("LSTM MODEL TRAINING")
print("="*60)

# ============================================
# STEP 1: Load the preprocessed data
# ============================================
print("\n Loading FD001 data...")

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

train_df = pd.read_csv('data/raw/train_FD001.txt', sep=r'\s+', names=column_names, engine='python')
max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']
train_df = train_df.merge(max_cycles, on='unit_number', how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']

X_train = train_df[features].values
y_train = train_df['RUL'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
train_df_scaled = pd.DataFrame(X_train_scaled, columns=features)
train_df_scaled['unit_number'] = train_df['unit_number'].values

seq_length = 50
X_seq, y_seq = [], []
for engine_id in train_df_scaled['unit_number'].unique():
    engine_idx = train_df_scaled['unit_number'] == engine_id
    X_engine = X_train_scaled[engine_idx]
    y_engine = y_train[engine_idx]
    
    if len(X_engine) > seq_length:
        for i in range(len(X_engine) - seq_length):
            X_seq.append(X_engine[i:i+seq_length])
            y_seq.append(y_engine[i+seq_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

split_idx = int(len(X_seq) * 0.8)
X_train_seq = X_seq[:split_idx]
X_test_seq = X_seq[split_idx:]
y_train_seq = y_seq[:split_idx]
y_test_seq = y_seq[split_idx:]

print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")

# ============================================
# STEP 3: Build LSTM Model
# ============================================
print("\n Building LSTM model...")

model = Sequential([
    # First LSTM layer
    LSTM(64, return_sequences=True, input_shape=(seq_length, X_train.shape[1])),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers for prediction
    Dense(16, activation='relu'),
    Dense(1)  # Output: RUL prediction
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ============================================
# STEP 4: Train the model
# ============================================
print("\n Training LSTM...")
print("   (This might take 2-5 minutes)")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-6, 
    verbose=1
)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================
# STEP 5: Make predictions
# ============================================
print("\n Making predictions...")
y_pred_lstm = model.predict(X_test_seq)

# ============================================
# STEP 6: Calculate errors
# ============================================
mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))

print("\n LSTM RESULTS:")
print(f"Mean Absolute Error (MAE): {mae_lstm:.2f} cycles")
print(f"Root Mean Square Error (RMSE): {rmse_lstm:.2f} cycles")

# ============================================
# STEP 7: Save the model
# ============================================

# ============================================
# STEP 8: Save the model
# ============================================
model.save('lstm_model.h5')
print("\n Model saved as 'lstm_model.h5'")

# ============================================
# STEP 9: Plot training history
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_seq, y_pred_lstm, alpha=0.3)
plt.plot([0, 350], [0, 350], 'r--', label='Perfect Prediction')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title(f'LSTM: Predictions vs Actual (MAE={mae_lstm:.1f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_results.png')
plt.show()

print("\n Plot saved as 'lstm_results.png'")
