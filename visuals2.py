import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Ensure images folder exists
os.makedirs('images', exist_ok=True)

def convert_to_mj(value):
    return value / 1e6

# Function to parse filenames and extract metadata
def parse_filename(filename):
    pattern = r"([^_]+)_(.*)\.xml_(total_energy_PP[01]_ENERGY_\(J\))(?:_(without_idle))?"
    match = re.match(pattern, filename)
    if match:
        project, ruleset, energy_type, idle_status = match.groups()
        idle_status = "without_idle" if idle_status else "with_idle"
        return project, ruleset, energy_type, idle_status
    return None

# Load all CSV files in the directory
data = []
directory = "energy_analysis_output"

for file in os.listdir(directory):
    if file.endswith(".csv"):
        parsed = parse_filename(file)
        if parsed:
            project, ruleset, energy_type, idle_status = parsed
            df = pd.read_csv(os.path.join(directory, file))
            df["Project"] = project
            df["Ruleset"] = ruleset
            df["Energy Type"] = energy_type
            df["Idle Status"] = idle_status
            data.append(df)

df_all = pd.concat(data, ignore_index=True)

df_all["PP0_ENERGY (J)"] = df_all["PP0_ENERGY (J)"].fillna(0)
df_all["PP1_ENERGY (J)"] = df_all["PP1_ENERGY (J)"].fillna(0)

print(df_all)

# Separate data into idle and non-idle
df_idle = df_all[df_all["Idle Status"] == "with_idle"]
df_non_idle = df_all[df_all["Idle Status"] == "without_idle"]

# # Aggregate average energy consumption per project
# project_energy = df_all.groupby("Project").mean().reset_index()
project_energy_idle = df_idle.groupby("Project")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].mean().reset_index()
project_energy_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]] = project_energy_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].applymap(convert_to_mj)
print(project_energy_idle)

project_energy_non_idle = df_non_idle.groupby("Project")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].mean().reset_index()
project_energy_non_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]] = project_energy_non_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]]
print(project_energy_non_idle)

# # Aggregate average energy per ruleset
# ruleset_energy = df_all.groupby("Ruleset").mean().reset_index()
ruleset_energy_idle = df_idle.groupby("Ruleset")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].mean().reset_index()
ruleset_energy_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]] = ruleset_energy_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].applymap(convert_to_mj)
print(ruleset_energy_idle)

ruleset_energy_non_idle = df_non_idle.groupby("Ruleset")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].mean().reset_index()
ruleset_energy_non_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]] = ruleset_energy_non_idle[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]]
print(ruleset_energy_non_idle)


# 1. Energy consumption by project (PP0 and PP1)
plt.figure(figsize=(10, 6))
project_energy_idle_melted = project_energy_idle.melt(id_vars=["Project"], var_name="Energy Type", value_name="Energy (MJ)")
sns.barplot(data=project_energy_idle_melted, x="Project", y="Energy (MJ)", hue="Energy Type")
# plt.xticks(rotation=45)
plt.title("Average Energy Consumption per Project (PP0 vs PP1) Idle")
plt.ylabel("Energy (MJ)")
plt.xlabel("Project")
plt.legend(title="Energy Type")
max_value = project_energy_idle_melted["Energy (MJ)"].max()
padding = max_value * 0.1  # Add 10% padding
plt.ylim(0, max_value + padding)
plt.savefig('images/Energy_Consumption_By_Project_Idle.png')
plt.close()

# 2. Energy consumption by project (PP0 and PP1) Non Idle
plt.figure(figsize=(10, 6))
project_energy_non_idle_melted = project_energy_non_idle.melt(id_vars=["Project"], var_name="Energy Type", value_name="Energy (J)")
sns.barplot(data=project_energy_non_idle_melted, x="Project", y="Energy (J)", hue="Energy Type")
# plt.xticks(rotation=45)
plt.title("Average Energy Consumption per Project (PP0 vs PP1) Non Idle")
plt.ylabel("Energy (J)")
plt.xlabel("Project")
plt.legend(title="Energy Type")
max_value = project_energy_non_idle_melted["Energy (J)"].max()
padding = max_value * 0.1  # Add 10% padding
plt.ylim(0, max_value + padding)
plt.savefig('images/Energy_Consumption_By_Project_Non_Idle.png')
plt.close()

# 3. Energy consumption by ruleset Idle
plt.figure(figsize=(10, 5))
ruleset_energy_idle_melted = ruleset_energy_idle.melt(id_vars=["Ruleset"], var_name="Energy Type", value_name="Energy (MJ)")
max_value = ruleset_energy_idle_melted["Energy (MJ)"].max()
sns.barplot(data=ruleset_energy_idle_melted, x="Ruleset", y="Energy (MJ)", hue="Energy Type")
plt.title("Average Energy Consumption per Ruleset (PP0 vs PP1) Idle")
plt.ylabel("Energy (MJ)")
plt.xlabel("Ruleset")
padding = max_value * 0.1  # Add 10% padding
plt.ylim(0, max_value + padding)  # Set y-axis limits
plt.legend(title="Energy Type")
plt.savefig('images/Energy_Consumption_By_Ruleset_Idle.png')
plt.close()

# 4. Energy consumption by ruleset Non Idle
plt.figure(figsize=(10, 5))
ruleset_energy_non_idle_melted = ruleset_energy_non_idle.melt(id_vars=["Ruleset"], var_name="Energy Type", value_name="Energy (J)")
max_value = ruleset_energy_non_idle_melted["Energy (J)"].max()
sns.barplot(data=ruleset_energy_non_idle_melted, x="Ruleset", y="Energy (J)", hue="Energy Type")
plt.title("Average Energy Consumption per Ruleset (PP0 vs PP1) Non Idle")
plt.ylabel("Energy (J)")
plt.xlabel("Ruleset")
padding = max_value * 0.1  # Add 10% padding
plt.ylim(0, max_value + padding)  # Set y-axis limits
plt.legend(title="Energy Type")
plt.savefig('images/Energy_Consumption_By_Ruleset_Non_Idle.png')
plt.close()

