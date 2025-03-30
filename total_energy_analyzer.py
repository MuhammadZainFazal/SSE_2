import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
            df["Idle Status"] = idle_status
            data.append(df)

df_all = pd.concat(data, ignore_index=True)
df_all["PP0_ENERGY (J)"] = df_all["PP0_ENERGY (J)"].fillna(0)
df_all["PP1_ENERGY (J)"] = df_all["PP1_ENERGY (J)"].fillna(0)

# Separate data into idle and non-idle
df_idle = df_all[df_all["Idle Status"] == "with_idle"]
df_non_idle = df_all[df_all["Idle Status"] == "without_idle"]

# Aggregate mean and std for Projects
def compute_aggregates(df):
    agg_df = df.groupby("Project")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["Project", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)"]

    # Filter out negative values
    agg_df = agg_df[(agg_df["PP0 Mean (J)"] > 0) & (agg_df["PP1 Mean (J)"] > 0)]

    return agg_df

project_energy_idle = compute_aggregates(df_idle)
project_energy_idle.iloc[:, 1:] = project_energy_idle.iloc[:, 1:].applymap(convert_to_mj)  # Convert to MJ

project_energy_non_idle = compute_aggregates(df_non_idle)

# Plot function with centered error bars & positive Y filtering
def plot_energy_p0_p1(data, x_col, y_col_p0, y_col_p1, y_err_p0, y_err_p1, title, ylabel, filename):
    if data.empty:
        print(f"Skipping {filename}: No positive Y values.")
        return

    plt.figure(figsize=(12, 6))  # Increased figure width

    bar_width = 0.4
    x_positions = np.arange(len(data))
    x_positions_p0 = x_positions - bar_width / 2
    x_positions_p1 = x_positions + bar_width / 2

    bars_p0 = plt.bar(x_positions_p0, data[y_col_p0], yerr=data[y_err_p0], width=bar_width, capsize=5, label="PP0 Energy", color='blue', alpha=0.7)
    bars_p1 = plt.bar(x_positions_p1, data[y_col_p1], yerr=data[y_err_p1], width=bar_width, capsize=5, label="PP1 Energy", color='red', alpha=0.7)

    plt.xticks(x_positions, data[x_col], rotation=45, ha="right", fontsize=10)  # Rotate & align labels

    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    min_y = max(0, min(data[y_col_p0].min() - data[y_err_p0].max(), data[y_col_p1].min() - data[y_err_p1].max()))
    max_y = max(data[y_col_p0].max() + data[y_err_p0].max(), data[y_col_p1].max() + data[y_err_p1].max())
    plt.ylim(min_y, max_y)

    plt.subplots_adjust(bottom=0.25)  # Add space for long labels

    plt.savefig(f'images/{filename}.png', bbox_inches="tight")  # Ensure full labels
    plt.close()


# Generate plots with centered error bars & positive Y filtering
plot_energy_p0_p1(project_energy_idle, "Project", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)", "PP1 Std (J)",
                   "Average Energy Consumption per Project (Idle)", "Energy (MJ)", "Energy_Consumption_By_Project_Idle")

plot_energy_p0_p1(project_energy_non_idle, "Project", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)", "PP1 Std (J)",
                   "Average Energy Consumption per Project (Non Idle)", "Energy (J)", "Energy_Consumption_By_Project_Non_Idle")
