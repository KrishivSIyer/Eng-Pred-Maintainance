"""
LSTM Model for Aircraft Engine RUL Prediction
Based on NASA C-MAPSS dataset
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

print("="*60)
print("LSTM MODEL TRAINING")
print("="*60)

# ============================================
# STEP 1: Load the preprocessed data
# ============================================
print("\n📂 Loading data...")

X_train = pd.read_csv('C:/Users/krish/OneDrive/Desktop/sem2 projects/aiml/fig/X_train.csv')
X_test = pd.read_csv('C:/Users/krish/OneDrive/Desktop/sem2 projects/aiml/fig/X_test.csv')
y_train = pd.read_csv('C:/Users/krish/OneDrive/Desktop/sem2 projects/aiml/fig/y_train.csv')
y_test = pd.read_csv('C:/Users/krish/OneDrive/Desktop/sem2 projects/aiml/fig/y_test.csv')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# ============================================
# STEP 2: Create sequences for LSTM
# ============================================
print("\n📊 Creating sequences for LSTM...")

def create_sequences(X, y, seq_length=50):
    """
    Convert data into sequences for LSTM
    
    Example:
    Input: 100 cycles of data
    Output: 51 sequences (each of length 50)
    Sequence 1: cycles 0-49 → predict cycle 50
    Sequence 2: cycles 1-50 → predict cycle 51
    etc.
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X.iloc[i:i+seq_length].values)
        y_seq.append(y.iloc[i+seq_length].values[0])
    
    return np.array(X_seq), np.array(y_seq)

# Create sequences
seq_length = 50
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

print(f"X_train_seq shape: {X_train_seq.shape}")  # (samples, 50, 24)
print(f"X_test_seq shape: {X_test_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}")
print(f"y_test_seq shape: {y_test_seq.shape}")

# ============================================
# STEP 3: Build LSTM Model
# ============================================
print("\n🏗️ Building LSTM model...")

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
print("\n🌳 Training LSTM...")
print("   (This might take 2-5 minutes)")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ============================================
# STEP 5: Make predictions
# ============================================
print("\n🔮 Making predictions...")
y_pred_lstm = model.predict(X_test_seq)

# ============================================
# STEP 6: Calculate errors
# ============================================
mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))

print("\n📊 LSTM RESULTS:")
print(f"Mean Absolute Error (MAE): {mae_lstm:.2f} cycles")
print(f"Root Mean Square Error (RMSE): {rmse_lstm:.2f} cycles")

# ============================================
# STEP 7: Compare with Random Forest
# ============================================
print("\n📊 COMPARISON:")
print(f"Random Forest MAE: 29.70 cycles")
print(f"LSTM MAE: {mae_lstm:.2f} cycles")

if mae_lstm < 29.70:
    improvement = 29.70 - mae_lstm
    print(f"✅ LSTM is better by {improvement:.2f} cycles!")
else:
    worse = mae_lstm - 29.70
    print(f"⚠️ Random Forest is better by {worse:.2f} cycles")

# ============================================
# STEP 8: Save the model
# ============================================
model.save('lstm_model.h5')
print("\n✅ Model saved as 'lstm_model.h5'")

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

print("\n✅ Plot saved as 'lstm_results.png'")