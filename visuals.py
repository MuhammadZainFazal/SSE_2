import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure images folder exists
os.makedirs('images', exist_ok=True)

# Load CSV file
df = pd.read_csv('enegibridge_output/qwen2.5-coder_1.5b-instruct-q5_0_output.csv')

# Convert Time column to datetime if needed
df['Time'] = pd.to_datetime(df['Time'], unit='ns')

# Calculate additional metrics
df['CPU_USAGE_MEAN'] = df[[col for col in df.columns if 'CPU_USAGE' in col]].mean(axis=1)
df['CPU_FREQUENCY_MEAN'] = df[[col for col in df.columns if 'CPU_FREQUENCY' in col]].mean(axis=1)
df['GPU_MEMORY_UTILIZATION'] = (df['GPU0_MEMORY_USED'] / df['GPU0_MEMORY_TOTAL']) * 100

# Calculate Energy Delay Product (EDP)
df['EDP'] = df['PACKAGE_ENERGY (J)'] * df['Time'].diff().dt.total_seconds().fillna(0)

# Visualizing CPU usage over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='CPU_USAGE_MEAN', data=df)
plt.title('Average CPU Usage Over Time')
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.xticks(rotation=45)
plt.savefig('images/CPU_Usage_Over_Time.png')
plt.close()

# Visualizing energy consumption over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='DRAM_ENERGY (J)', data=df, label='DRAM Energy (J)')
sns.lineplot(x='Time', y='PACKAGE_ENERGY (J)', data=df, label='Package Energy (J)')
plt.title('Energy Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('Energy (J)')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('images/Energy_Consumption_Over_Time.png')
plt.close()

# Visualizing Energy Delay Product (EDP) over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='EDP', data=df)
plt.title('Energy Delay Product Over Time')
plt.xlabel('Time')
plt.ylabel('EDP (J·s)')
plt.xticks(rotation=45)
plt.savefig('images/EDP_Over_Time.png')
plt.close()

# Summary statistics
summary_stats = df.describe()
print(summary_stats)

# Save the summary statistics to a CSV file
summary_stats.to_csv('summary_statistics.csv')