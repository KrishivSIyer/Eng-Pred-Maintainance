import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.get_logger().setLevel('ERROR')

# Column names
column_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

features = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
SEQ = 50

results = []

for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
    print(f"\n{ds}")
    
    # Load test data
    test_df = pd.read_csv(f'data/raw/test_{ds}.txt', sep='\s+', names=column_names, engine='python')
    y_true = pd.read_csv(f'data/raw/RUL_{ds}.txt', names=['RUL']).values.ravel()
    
    # RF: last cycle only
    last = test_df.groupby('unit_number').last().reset_index()
    X_rf = last[features].values
    scaler = joblib.load(f'models/scaler_{ds}.pkl')
    rf = joblib.load(f'models/rf_{ds}_combined.pkl')
    y_rf = rf.predict(scaler.transform(X_rf))
    
    # LSTM: last 50 cycles
    seqs = []
    for eid in test_df['unit_number'].unique():
        eng = test_df[test_df['unit_number'] == eid][features].values
        if len(eng) >= SEQ:
            seqs.append(eng[-SEQ:])
        else:
            pad = np.tile(eng[0], (SEQ - len(eng), 1))
            seqs.append(np.vstack([pad, eng]))
    lstm = load_model(f'models/lstm_{ds}_finetuned.keras')
    y_lstm = lstm.predict(np.array(seqs), verbose=0).flatten()
    
    # Metrics
    mae_rf = mean_absolute_error(y_true, y_rf)
    mae_lstm = mean_absolute_error(y_true, y_lstm)
    winner = 'RF' if mae_rf < mae_lstm else 'LSTM'
    
    results.append({'Dataset': ds, 'RF_MAE': round(mae_rf, 2), 'LSTM_MAE': round(mae_lstm, 2), 'Winner': winner})
    print(f"  RF: {mae_rf:.2f} | LSTM: {mae_lstm:.2f} | Winner: {winner}")

print("\n" + "="*40)
print(pd.DataFrame(results).to_string(index=False))