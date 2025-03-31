import os
import re
import numpy as np
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
project_energy_idle = df_idle.groupby("Project")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
project_energy_idle.columns = ["Project", "PP0 Mean (MJ)", "PP0 Std (MJ)", "PP1 Mean (MJ)", "PP1 Std (MJ)"]
project_energy_idle[["PP0 Mean (MJ)", "PP0 Std (MJ)", "PP1 Mean (MJ)", "PP1 Std (MJ)"]] = project_energy_idle[["PP0 Mean (MJ)", "PP0 Std (MJ)", "PP1 Mean (MJ)", "PP1 Std (MJ)"]].applymap(convert_to_mj)

project_energy_non_idle = df_non_idle.groupby("Project")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
project_energy_non_idle.columns = ["Project", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)"]

# Aggregate mean and std for Rulesets
ruleset_energy_idle = df_idle.groupby("Ruleset")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
ruleset_energy_idle.columns = ["Ruleset", "PP0 Mean (MJ)", "PP0 Std (MJ)", "PP1 Mean (MJ)", "PP1 Std (MJ)"]
ruleset_energy_idle[["PP0 Mean (MJ)", "PP0 Std (MJ)", "PP1 Mean (MJ)", "PP1 Std (MJ)"]] = ruleset_energy_idle[["PP0 Mean (MJ)", "PP0 Std (MJ)", "PP1 Mean (MJ)", "PP1 Std (MJ)"]].applymap(convert_to_mj)

ruleset_energy_non_idle = df_non_idle.groupby("Ruleset")[["PP0_ENERGY (J)", "PP1_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
ruleset_energy_non_idle.columns = ["Ruleset", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)"]

# Plot function for PP0 & PP1
def plot_energy_p0_p1(data, x_col, y_col_p0, y_col_p1, y_err_p0, y_err_p1, title, ylabel, filename):
    if data.empty:
        print(f"Skipping {filename}: No positive Y values.")
        return

    plt.figure(figsize=(12, 6))

    # Only include positive Y values
    data = data[(data[y_col_p0] > 0) & (data[y_col_p1] > 0)]
    if data.empty:
        print(f"Skipping {filename}: No positive Y values after filtering.")
        return

    bar_width = 0.4
    x_positions = np.arange(len(data))
    x_positions_p0 = x_positions - bar_width / 2
    x_positions_p1 = x_positions + bar_width / 2

    # Plot bars with error bars
    plt.bar(x_positions_p0, data[y_col_p0], yerr=data[y_err_p0], width=bar_width, capsize=5, label="PP0 Energy", color='blue', alpha=0.7, error_kw={'elinewidth': 1, 'capsize': 5})
    plt.bar(x_positions_p1, data[y_col_p1], yerr=data[y_err_p1], width=bar_width, capsize=5, label="PP1 Energy", color='red', alpha=0.7, error_kw={'elinewidth': 1, 'capsize': 5})

    # Set x-axis labels
    plt.xticks(x_positions, data[x_col], rotation=45, ha="right", fontsize=10)

    # Labels and title
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Adjust y-axis
    min_y = max(0, min(data[y_col_p0].min() - data[y_err_p0].max(), data[y_col_p1].min() - data[y_err_p1].max()))
    max_y = max(data[y_col_p0].max() + data[y_err_p0].max(), data[y_col_p1].max() + data[y_err_p1].max())
    plt.ylim(min_y, max_y)

    # Prevent label cutoff
    plt.subplots_adjust(bottom=0.25)

    # Save graph
    plt.savefig(f'images/{filename}.png', bbox_inches="tight")
    plt.close()

# Generate graphs for Projects
plot_energy_p0_p1(project_energy_idle, "Project", "PP0 Mean (MJ)", "PP1 Mean (MJ)", "PP0 Std (MJ)", "PP1 Std (MJ)", "Average Energy Consumption per Project (Idle)", "Energy (MJ)", "Energy_Consumption_By_Project_Idle")
plot_energy_p0_p1(project_energy_non_idle, "Project", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)", "PP1 Std (J)", "Average Energy Consumption per Project (Non Idle)", "Energy (J)", "Energy_Consumption_By_Project_Non_Idle")

# Generate graphs for Rulesets
plot_energy_p0_p1(ruleset_energy_idle, "Ruleset", "PP0 Mean (MJ)", "PP1 Mean (MJ)", "PP0 Std (MJ)", "PP1 Std (MJ)", "Average Energy Consumption per Ruleset (Idle)", "Energy (MJ)", "Energy_Consumption_By_Ruleset_Idle")
plot_energy_p0_p1(ruleset_energy_non_idle, "Ruleset", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)", "PP1 Std (J)", "Average Energy Consumption per Ruleset (Non Idle)", "Energy (J)", "Energy_Consumption_By_Ruleset_Non_Idle")


# Boxplot function
def plot_boxplot(df, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[x_col], y=df[y_col])
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'images/{filename}.png', bbox_inches="tight")
    plt.close()

# Violin plot function
def plot_violin(df, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=df[x_col], y=df[y_col])
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'images/{filename}.png', bbox_inches="tight")
    plt.close()

# Generate boxplots
plot_boxplot(df_idle, "Project", "PP0_ENERGY (J)", "PP0 Energy Consumption by Project", "PP0 Energy (MJ)", "Boxplot_PP0_By_Project")
plot_boxplot(df_idle, "Project", "PP1_ENERGY (J)", "PP1 Energy Consumption by Project", "PP1 Energy (J)", "Boxplot_PP1_By_Project")
plot_boxplot(df_idle, "Ruleset", "PP0_ENERGY (J)", "PP0 Energy Consumption by Ruleset", "PP0 Energy (J)", "Boxplot_PP0_By_Ruleset")
plot_boxplot(df_idle, "Ruleset", "PP1_ENERGY (J)", "PP1 Energy Consumption by Ruleset", "PP1 Energy (J)", "Boxplot_PP1_By_Ruleset")

# Generate violin plots
plot_violin(df_idle, "Project", "PP0_ENERGY (J)", "PP0 Energy Distribution by Project", "PP0 Energy (J)", "Violin_PP0_By_Project")
plot_violin(df_idle, "Project", "PP1_ENERGY (J)", "PP1 Energy Distribution by Project", "PP1 Energy (J)", "Violin_PP1_By_Project")
plot_violin(df_idle, "Ruleset", "PP0_ENERGY (J)", "PP0 Energy Distribution by Ruleset", "PP0 Energy (J)", "Violin_PP0_By_Ruleset")
plot_violin(df_idle, "Ruleset", "PP1_ENERGY (J)", "PP1 Energy Distribution by Ruleset", "PP1 Energy (J)", "Violin_PP1_By_Ruleset")
