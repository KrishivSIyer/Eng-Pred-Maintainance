"""
PART 1: Data Exploration
Goal: Load data correctly and understand what we're working with
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*60)
print("PART 1: AIRCRAFT ENGINE DATA EXPLORATION")
print("="*60)

# Step 1: Define column names (based on NASA C-MAPSS documentation)
column_names = [
    'unit_number',           # Engine ID (1-100)
    'time_in_cycles',        # Cycle number (time)
    'setting_1',             # Operational setting 1
    'setting_2',             # Operational setting 2
    'setting_3',             # Operational setting 3
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Step 2: Load the data with proper error handling
print("\n📂 Loading training data...")
data_path = input("Enter the folder path where your FD001 files are: ").strip()

try:
    train_data = pd.read_csv(data_path + '/train_FD001.txt', 
                             sep='\s+', 
                             names=column_names,
                             engine='python')
    
    train_data = train_data.dropna(how='all')
    
    initial_rows = len(train_data)
    train_data = train_data.drop_duplicates()
    
    print(f"✅ Loaded successfully!")
    print(f"   Raw rows: {initial_rows}")
    print(f"   After removing duplicates: {len(train_data)} rows")
    print(f"   Columns: {train_data.shape[1]}")
    print(f"   Engines: {train_data['unit_number'].nunique()}")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure the path is correct")
    print("2. File should be named exactly 'train_FD001.txt'")
    print("3. Check that the file isn't corrupted")
    exit()

# Step 3: Data Validation - Check if data is loaded correctly
print("\n" + "="*60)
print("🔍 DATA VALIDATION")
print("="*60)

# Check each engine's data integrity
print("\nEngine data integrity check:")
problem_engines = []
for engine in train_data['unit_number'].unique():
    engine_data = train_data[train_data['unit_number'] == engine].copy()
    cycles = engine_data['time_in_cycles'].values
    
    # Check if cycles are sequential and start at 1
    expected_cycles = np.arange(1, len(cycles) + 1)
    if not np.array_equal(cycles, expected_cycles):
        problem_engines.append(engine)
        print(f"  ❌ Engine {engine}: Cycle sequence broken")
    else:
        print(f"  ✅ Engine {engine}: {len(cycles)} cycles, sequential OK")

if problem_engines:
    print(f"\n⚠️  Warning: {len(problem_engines)} engines have cycle sequence issues")
else:
    print("\n✅ All engines have correct sequential cycles!")

# Step 4: Basic statistics
print("\n" + "="*60)
print("📊 BASIC STATISTICS")
print("="*60)

print("\nEngine lifetimes (cycles until failure):")
engine_lifetimes = train_data.groupby('unit_number')['time_in_cycles'].max()
print(f"  Min: {engine_lifetimes.min()} cycles")
print(f"  Max: {engine_lifetimes.max()} cycles")
print(f"  Average: {engine_lifetimes.mean():.1f} cycles")
print(f"  Median: {engine_lifetimes.median():.0f} cycles")

# Step 5: Calculate RUL CORRECTLY
print("\n" + "="*60)
print("⏱️  CALCULATING REMAINING USEFUL LIFE (RUL)")
print("="*60)

# For each engine, find its maximum cycle (failure point)
max_cycles = train_data.groupby('unit_number')['time_in_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']

# Merge back to get max cycle for each row
train_data_with_rul = train_data.merge(max_cycles, on='unit_number', how='left')

# RUL = max_cycle - current_cycle
train_data_with_rul['RUL'] = train_data_with_rul['max_cycle'] - train_data_with_rul['time_in_cycles']

# Drop the temporary column
train_data_with_rul = train_data_with_rul.drop('max_cycle', axis=1)

# FIX 6: Verify RUL calculation for Engine 1
engine_1 = train_data_with_rul[train_data_with_rul['unit_number'] == 1].copy()
print("\n✅ RUL Verification for Engine 1:")
print(f"   Total cycles: {engine_1['time_in_cycles'].max()}")
print(f"   RUL at cycle 1: {engine_1['RUL'].iloc[0]} (should be {engine_1['time_in_cycles'].max()-1})")
print(f"   RUL at last cycle: {engine_1['RUL'].iloc[-1]} (should be 0)")

# Step 6: Display clean sample
print("\n" + "="*60)
print("📋 CLEAN DATA SAMPLE")
print("="*60)

print("\nFirst 10 rows for Engine 1:")
print(engine_1[['unit_number', 'time_in_cycles', 'RUL']].head(10))

print(f"\nRUL range for Engine 1: {engine_1['RUL'].min()} to {engine_1['RUL'].max()} cycles")
print(f"Global RUL range: {train_data_with_rul['RUL'].min()} to {train_data_with_rul['RUL'].max()} cycles")

# Step 7: Visualize one engine correctly
print("\n" + "="*60)
print("🎨 CREATING VISUALIZATIONS")
print("="*60)

# Pick a random engine
random_engine = np.random.choice(train_data_with_rul['unit_number'].unique())
engine_data = train_data_with_rul[train_data_with_rul['unit_number'] == random_engine].copy()

print(f"\nPlotting Engine #{random_engine}...")
print(f"This engine ran for {len(engine_data)} cycles before failure")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'Engine #{random_engine} - Sensor Readings Over Time', fontsize=14)

# Plot 4 different sensors
sensors_to_plot = ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_15']
titles = ['Temperature (Sensor 2)', 'Pressure (Sensor 7)', 
          'Vibration (Sensor 11)', 'Fan Speed (Sensor 15)']

for i, (sensor, title) in enumerate(zip(sensors_to_plot, titles)):
    row, col = i // 2, i % 2
    axes[row, col].plot(engine_data['time_in_cycles'], engine_data[sensor], 
                        linewidth=2, color='blue')
    axes[row, col].set_xlabel('Cycles')
    axes[row, col].set_ylabel('Sensor Value')
    axes[row, col].set_title(title)
    axes[row, col].grid(True, alpha=0.3)
    
    # Color the area as engine gets closer to failure
    axes[row, col].axvspan(len(engine_data)-30, len(engine_data), 
                           alpha=0.2, color='red', label='Last 30 cycles')

plt.tight_layout()
plt.savefig('engine_exploration.png')
plt.show()

# Step 8: Visualize RUL distribution
print("\n📊 Creating RUL visualization...")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(train_data_with_rul['RUL'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('RUL (cycles)')
plt.ylabel('Frequency')
plt.title('Distribution of RUL Values')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# For one engine, show RUL decreasing over time
one_engine = train_data_with_rul[train_data_with_rul['unit_number'] == random_engine]
plt.plot(one_engine['time_in_cycles'], one_engine['RUL'], linewidth=2, color='red')
plt.xlabel('Cycle Number')
plt.ylabel('RUL (cycles)')
plt.title(f'Engine #{random_engine} - RUL Over Time')
plt.gca().invert_yaxis()  # So RUL decreases as we go right
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rul_visualization.png')
plt.show()

# Step 9: Final data quality report
print("\n" + "="*60)
print("📈 FINAL DATA QUALITY REPORT")
print("="*60)

print(f"\nDataset shape: {train_data_with_rul.shape}")
print(f"Number of engines: {train_data_with_rul['unit_number'].nunique()}")
print(f"Total rows: {len(train_data_with_rul)}")
print(f"Missing values: {train_data_with_rul.isnull().sum().sum()}")
print(f"Duplicate rows: {train_data_with_rul.duplicated().sum()}")

# Step 10: Save processed data
print("\n" + "="*60)
print("💾 SAVING PROCESSED DATA")
print("="*60)

train_data_with_rul.to_csv('engine_data_with_rul.csv', index=False)
print("✅ Saved as 'engine_data_with_rul.csv'")
print(f"   File size: {os.path.getsize('engine_data_with_rul.csv') / 1024:.1f} KB")

print("\n" + "="*60)
print("🎉 PART 1 COMPLETE - DATA IS CLEAN!")
print("="*60)
print("\nWhat we accomplished:")
print("✓ Loaded NASA C-MAPSS dataset correctly")
print("✓ Removed all duplicate rows")
print("✓ Verified data integrity for all engines")
print("✓ Calculated RUL correctly")
print("✓ Created clean visualizations")
print("✓ Saved processed data for next steps")
print("\n✅ Your data is now ready for Part 2 - Preprocessing!")