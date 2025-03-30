import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def convert_to_mj(value):
    return value / 1e6

def parse_filename(filename):
    pattern = r"([^_]+)_(.*)\.xml_(total_energy_PP[01]_ENERGY_\(J\))(?:_(without_idle))?"
    match = re.match(pattern, filename)
    if match:
        project, ruleset, energy_type, idle_status = match.groups()
        idle_status = "without_idle" if idle_status else "with_idle"
        return project, ruleset, energy_type, idle_status
    return None

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

# Separate data into idle and non-idle
df_idle = df_all[df_all["Idle Status"] == "with_idle"]
df_non_idle = df_all[df_all["Idle Status"] == "without_idle"]

# Aggregate mean and std for Projects
project_energy_idle = df_idle.groupby("Project")["PP0_ENERGY (J)"].agg(["mean", "std"]).reset_index()
project_energy_idle.columns = ["Project", "Energy Mean (MJ)", "Energy Std (MJ)"]
project_energy_idle[["Energy Mean (MJ)", "Energy Std (MJ)"]] = project_energy_idle[["Energy Mean (MJ)", "Energy Std (MJ)"]].applymap(convert_to_mj)

project_energy_non_idle = df_non_idle.groupby("Project")["PP0_ENERGY (J)"].agg(["mean", "std"]).reset_index()
project_energy_non_idle.columns = ["Project", "Energy Mean (J)", "Energy Std (J)"]

# Aggregate mean and std for Rulesets
ruleset_energy_idle = df_idle.groupby("Ruleset")["PP0_ENERGY (J)"].agg(["mean", "std"]).reset_index()
ruleset_energy_idle.columns = ["Ruleset", "Energy Mean (MJ)", "Energy Std (MJ)"]
ruleset_energy_idle[["Energy Mean (MJ)", "Energy Std (MJ)"]] = ruleset_energy_idle[["Energy Mean (MJ)", "Energy Std (MJ)"]].applymap(convert_to_mj)

ruleset_energy_non_idle = df_non_idle.groupby("Ruleset")["PP0_ENERGY (J)"].agg(["mean", "std"]).reset_index()
ruleset_energy_non_idle.columns = ["Ruleset", "Energy Mean (J)", "Energy Std (J)"]

# Plot functions
def plot_energy(data, x_col, y_col, y_err, title, ylabel, filename):
    plt.figure(figsize=(10, 6))

    # Calculate error bars
    y_errors = data[y_err].values if len(data) > 1 else None  # Ensure yerr is None if only one data point

    # Create the bar plot
    sns.barplot(data=data, x=x_col, y=y_col, capsize=0.2, width=0.4)

    # Add error bars
    if y_errors is not None:
        plt.errorbar(data[x_col], data[y_col], yerr=y_errors, fmt='none', capsize=5, color='black')

    # Calculate the limits for the y-axis to include the full range of mean ± std
    min_y = max(0, data[y_col].min() - data[y_err].max())  # Ensure min_y is not below zero
    max_y = data[y_col].max() + data[y_err].max()  # mean + std

    # Set y-axis limits
    plt.ylim(min_y, max_y)

    # Add titles and labels
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_col)

    # Save the plot to a file
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.savefig(f'images/{filename}.png')
    plt.close()


# Generate plots
plot_energy(project_energy_idle, "Project", "Energy Mean (MJ)", "Energy Std (MJ)", "Average Energy Consumption per Project (PP0) Idle", "Energy (MJ)", "Energy_Consumption_By_Project_Idle")
plot_energy(project_energy_non_idle, "Project", "Energy Mean (J)", "Energy Std (J)", "Average Energy Consumption per Project (PP0) Non Idle", "Energy (J)", "Energy_Consumption_By_Project_Non_Idle")
plot_energy(ruleset_energy_idle, "Ruleset", "Energy Mean (MJ)", "Energy Std (MJ)", "Average Energy Consumption per Ruleset (PP0) Idle", "Energy (MJ)", "Energy_Consumption_By_Ruleset_Idle")
plot_energy(ruleset_energy_non_idle, "Ruleset", "Energy Mean (J)", "Energy Std (J)", "Average Energy Consumption per Ruleset (PP0) Non Idle", "Energy (J)", "Energy_Consumption_By_Ruleset_Non_Idle")
