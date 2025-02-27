import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure images folder exists
os.makedirs('images', exist_ok=True)

# List parquet files directly from the directory
parquet_dir = 'energibridge_output'
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
all_data = []

if not parquet_files:
    raise FileNotFoundError("No parquet files found in 'energibridge_output' folder")

for file in parquet_files:
    model_name = (os.path.splitext(file)[0]).rstrip('_dataframe')
    file_path = os.path.join(parquet_dir, file)
    df = pd.read_parquet(file_path)
    df['model'] = model_name
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

# Calculate time since start in seconds
df['Time'] = (df['Time'] - df['Time'].min()) / 1e6

print(df.columns)

# Calculate additional metrics
df['CPU_USAGE_MEAN'] = df[[col for col in df.columns if 'CPU_USAGE' in col]].mean(axis=1)
df['CPU_FREQUENCY_MEAN'] = df[[col for col in df.columns if 'CPU_FREQUENCY' in col]].mean(axis=1)
df['GPU_MEMORY_UTILIZATION'] = (df['GPU0_MEMORY_USED'] / df['GPU0_MEMORY_TOTAL']) * 100

# Calculate Energy Delay Product (EDP)
df['EDP'] = df['PACKAGE_ENERGY (J)'] * df['Time']

# Visualizing CPU usage over time per model
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='CPU_USAGE_MEAN', hue='model', data=df)
plt.title('Average CPU Usage Over Time by Model')
plt.xlabel('Time (ms)')
plt.ylabel('CPU Usage (%)')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.savefig('images/CPU_Usage_Over_Time_by_Model.png')
plt.close()

# Visualizing energy consumption over time per model
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='DRAM_ENERGY (J)', hue='model', data=df, legend='full')
sns.lineplot(x='Time', y='PACKAGE_ENERGY (J)', hue='model', data=df, legend='full')
plt.title('Energy Consumption Over Time by Model')
plt.xlabel('Time (ms)')
plt.ylabel('Energy (J)')
plt.legend(title='Model')
plt.xticks(rotation=45)
plt.savefig('images/Energy_Consumption_Over_Time_by_Model.png')
plt.close()

# Visualizing Energy Delay Product (EDP) over time per model
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='EDP', hue='model', data=df)
plt.title('Energy Delay Product Over Time by Model')
plt.xlabel('Time (ms)')
plt.ylabel('EDP (J·s)')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.savefig('images/EDP_Over_Time_by_Model.png')
plt.close()

# Visualizing EDP distribution by model using violin plots
plt.figure(figsize=(12, 12))
sns.violinplot(x='model', y='EDP', data=df, inner='quartile', scale='width')
plt.title('Energy Delay Product Distribution by Model')
plt.xlabel('Model')
plt.ylabel('EDP (J·s)')
plt.xticks(rotation=45)
plt.savefig('images/EDP_Distribution_by_Model.png')
plt.close()

# Summary statistics by model
summary_stats = df.groupby('model').describe()
print(summary_stats)

# Save the summary statistics to a CSV file
summary_stats.to_csv('summary_statistics_by_model.csv')
