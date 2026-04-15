# lstm_model.py
# Separate file for LSTM implementation

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("="*60)
print("LSTM MODEL TRAINING")
print("="*60)

# Load the same preprocessed data
X_train = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/X_train.csv')
X_test = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/X_test.csv')
y_train = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/y_train.csv')
y_test = pd.read_csv('C:/Users/krish\OneDrive/Desktop/sem2 projects/aiml/fig/y_test.csv')

print(f"Data loaded: {X_train.shape}")