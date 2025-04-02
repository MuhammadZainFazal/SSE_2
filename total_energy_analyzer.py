import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def convert_to_mj(value):
    return value / 1e6


def parse_filename(filename):
    # Updated pattern to handle PACKAGE_ENERGY as well
    pattern = r"([^_]+)_(.*)\.xml_(total_energy_(?:PP[01]|PACKAGE)_ENERGY_\(J\))(?:_(without_idle))?"
    match = re.match(pattern, filename)
    if match:
        project, ruleset, energy_type, idle_status = match.groups()
        idle_status = "without_idle" if idle_status else "with_idle"
        return project, ruleset, energy_type, idle_status
    return None


def remove_outliers(df, group_cols, value_cols):
    """
    Remove min and max values for each group defined by multiple columns in the dataframe.

    Parameters:
    df (DataFrame): Input dataframe
    group_cols (list): List of column names to group by
    value_cols (list): List of column names for which outliers should be removed

    Returns:
    DataFrame: Dataframe with outliers removed
    """
    result_df = pd.DataFrame()

    # Group by all grouping columns
    for group_name, group_data in df.groupby(group_cols):
        filtered_group = group_data.copy()

        for col in value_cols:
            if not filtered_group.empty and len(filtered_group) > 2:  # Only remove if we have more than 2 values
                min_idx = filtered_group[col].idxmin()
                max_idx = filtered_group[col].idxmax()
                # Remove both min and max
                filtered_group = filtered_group.drop([min_idx, max_idx])

        result_df = pd.concat([result_df, filtered_group])

    return result_df


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
df_all["PACKAGE_ENERGY (J)"] = df_all["PACKAGE_ENERGY (J)"].fillna(0)  # Handle PACKAGE_ENERGY column

# Create a version without outliers, grouped by Project, Ruleset, and Idle Status
df_all_no_outliers = remove_outliers(
    df_all,
    ["Project", "Ruleset", "Idle Status"],
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]  # Include PACKAGE_ENERGY
)

# Separate data into non-idle only (original data)
df_non_idle = df_all[df_all["Idle Status"] == "without_idle"]

# Separate data into non-idle only (data without outliers)
df_non_idle_no_outliers = df_all_no_outliers[df_all_no_outliers["Idle Status"] == "without_idle"]

# Process dataframes with outliers
# Aggregate mean and std for Projects
project_energy_non_idle = df_non_idle.groupby("Project")[
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
project_energy_non_idle.columns = ["Project", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)",
                                   "PACKAGE Mean (J)", "PACKAGE Std (J)"]

# Aggregate mean and std for Rulesets
ruleset_energy_non_idle = df_non_idle.groupby("Ruleset")[
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
ruleset_energy_non_idle.columns = ["Ruleset", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)",
                                   "PACKAGE Mean (J)", "PACKAGE Std (J)"]

# Process dataframes without outliers
# Aggregate mean and std for Projects
project_energy_non_idle_no_outliers = df_non_idle_no_outliers.groupby("Project")[
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
project_energy_non_idle_no_outliers.columns = ["Project", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)",
                                               "PACKAGE Mean (J)", "PACKAGE Std (J)"]

# Aggregate mean and std for Rulesets without outliers
ruleset_energy_non_idle_no_outliers = df_non_idle_no_outliers.groupby("Ruleset")[
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
ruleset_energy_non_idle_no_outliers.columns = ["Ruleset", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)", "PP1 Std (J)",
                                               "PACKAGE Mean (J)", "PACKAGE Std (J)"]

# Also add project-ruleset combined analysis
project_ruleset_energy_non_idle = df_non_idle.groupby(["Project", "Ruleset"])[
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
project_ruleset_energy_non_idle.columns = ["Project", "Ruleset", "PP0 Mean (J)", "PP0 Std (J)", "PP1 Mean (J)",
                                           "PP1 Std (J)", "PACKAGE Mean (J)", "PACKAGE Std (J)"]

project_ruleset_energy_non_idle_no_outliers = df_non_idle_no_outliers.groupby(["Project", "Ruleset"])[
    ["PP0_ENERGY (J)", "PP1_ENERGY (J)", "PACKAGE_ENERGY (J)"]].agg(["mean", "std"]).reset_index()
project_ruleset_energy_non_idle_no_outliers.columns = ["Project", "Ruleset", "PP0 Mean (J)", "PP0 Std (J)",
                                                       "PP1 Mean (J)", "PP1 Std (J)", "PACKAGE Mean (J)",
                                                       "PACKAGE Std (J)"]


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
    plt.bar(x_positions_p0, data[y_col_p0], yerr=data[y_err_p0], width=bar_width, capsize=5, label="PP0 Energy",
            color='blue', alpha=0.7, error_kw={'elinewidth': 1, 'capsize': 5})
    plt.bar(x_positions_p1, data[y_col_p1], yerr=data[y_err_p1], width=bar_width, capsize=5, label="PP1 Energy",
            color='red', alpha=0.7, error_kw={'elinewidth': 1, 'capsize': 5})

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


# New function to plot all three energy types
def plot_energy_all(data, x_col, y_cols, y_errs, colors, title, ylabel, filename):
    if data.empty:
        print(f"Skipping {filename}: No data.")
        return

    plt.figure(figsize=(14, 6))

    # Only include positive Y values for all energy types
    mask = data[y_cols[0]] > 0
    for col in y_cols[1:]:
        mask = mask & (data[col] > 0)

    data = data[mask]
    if data.empty:
        print(f"Skipping {filename}: No positive Y values after filtering.")
        return

    bar_width = 0.25
    x_positions = np.arange(len(data))

    # Adjust positions for each energy type
    positions = []
    for i in range(len(y_cols)):
        positions.append(x_positions + (i - 1) * bar_width)

    # Plot bars with error bars for each energy type
    for i, (y_col, y_err, color, label) in enumerate(
            zip(y_cols, y_errs, colors, ["PP0 Energy", "PP1 Energy", "Package Energy"])):
        plt.bar(positions[i], data[y_col], yerr=data[y_err], width=bar_width, capsize=5,
                label=label, color=color, alpha=0.7, error_kw={'elinewidth': 1, 'capsize': 5})

    # Set x-axis labels
    plt.xticks(x_positions, data[x_col], rotation=45, ha="right", fontsize=10)

    # Labels and title
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Prevent label cutoff
    plt.subplots_adjust(bottom=0.25)

    # Save graph
    plt.savefig(f'images/{filename}.png', bbox_inches="tight")
    plt.close()


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


# Generate graphs for Projects with outliers - PP0 & PP1
plot_energy_p0_p1(project_energy_non_idle, "Project", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)", "PP1 Std (J)",
                  "Average PP0 & PP1 Energy Consumption per Project (Non Idle)", "Energy (J)",
                  "Energy_Consumption_PP0_PP1_By_Project_Non_Idle")

# Generate graphs for Projects with outliers - All energy types
plot_energy_all(project_energy_non_idle, "Project",
                ["PP0 Mean (J)", "PP1 Mean (J)", "PACKAGE Mean (J)"],
                ["PP0 Std (J)", "PP1 Std (J)", "PACKAGE Std (J)"],
                ['blue', 'red', 'green'],
                "Average Energy Consumption per Project (Non Idle)", "Energy (J)",
                "Energy_Consumption_All_By_Project_Non_Idle")

# Generate graphs for Rulesets with outliers - PP0 & PP1
plot_energy_p0_p1(ruleset_energy_non_idle, "Ruleset", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)", "PP1 Std (J)",
                  "Average PP0 & PP1 Energy Consumption per Ruleset (Non Idle)", "Energy (J)",
                  "Energy_Consumption_PP0_PP1_By_Ruleset_Non_Idle")

# Generate graphs for Rulesets with outliers - All energy types
plot_energy_all(ruleset_energy_non_idle, "Ruleset",
                ["PP0 Mean (J)", "PP1 Mean (J)", "PACKAGE Mean (J)"],
                ["PP0 Std (J)", "PP1 Std (J)", "PACKAGE Std (J)"],
                ['blue', 'red', 'green'],
                "Average Energy Consumption per Ruleset (Non Idle)", "Energy (J)",
                "Energy_Consumption_All_By_Ruleset_Non_Idle")

# Generate boxplots with outliers
plot_boxplot(df_non_idle, "Project", "PP0_ENERGY (J)", "PP0 Energy Consumption by Project (Non Idle)", "PP0 Energy (J)",
             "Boxplot_PP0_By_Project_Non_Idle")
plot_boxplot(df_non_idle, "Project", "PP1_ENERGY (J)", "PP1 Energy Consumption by Project (Non Idle)", "PP1 Energy (J)",
             "Boxplot_PP1_By_Project_Non_Idle")
plot_boxplot(df_non_idle, "Project", "PACKAGE_ENERGY (J)", "PACKAGE Energy Consumption by Project (Non Idle)",
             "PACKAGE Energy (J)", "Boxplot_PACKAGE_By_Project_Non_Idle")
plot_boxplot(df_non_idle, "Ruleset", "PP0_ENERGY (J)", "PP0 Energy Consumption by Ruleset (Non Idle)", "PP0 Energy (J)",
             "Boxplot_PP0_By_Ruleset_Non_Idle")
plot_boxplot(df_non_idle, "Ruleset", "PP1_ENERGY (J)", "PP1 Energy Consumption by Ruleset (Non Idle)", "PP1 Energy (J)",
             "Boxplot_PP1_By_Ruleset_Non_Idle")
plot_boxplot(df_non_idle, "Ruleset", "PACKAGE_ENERGY (J)", "PACKAGE Energy Consumption by Ruleset (Non Idle)",
             "PACKAGE Energy (J)", "Boxplot_PACKAGE_By_Ruleset_Non_Idle")

# Generate violin plots with outliers
plot_violin(df_non_idle, "Project", "PP0_ENERGY (J)", "PP0 Energy Distribution by Project (Non Idle)", "PP0 Energy (J)",
            "Violin_PP0_By_Project_Non_Idle")
plot_violin(df_non_idle, "Project", "PP1_ENERGY (J)", "PP1 Energy Distribution by Project (Non Idle)", "PP1 Energy (J)",
            "Violin_PP1_By_Project_Non_Idle")
plot_violin(df_non_idle, "Project", "PACKAGE_ENERGY (J)", "PACKAGE Energy Distribution by Project (Non Idle)",
            "PACKAGE Energy (J)", "Violin_PACKAGE_By_Project_Non_Idle")
plot_violin(df_non_idle, "Ruleset", "PP0_ENERGY (J)", "PP0 Energy Distribution by Ruleset (Non Idle)", "PP0 Energy (J)",
            "Violin_PP0_By_Ruleset_Non_Idle")
plot_violin(df_non_idle, "Ruleset", "PP1_ENERGY (J)", "PP1 Energy Distribution by Ruleset (Non Idle)", "PP1 Energy (J)",
            "Violin_PP1_By_Ruleset_Non_Idle")
plot_violin(df_non_idle, "Ruleset", "PACKAGE_ENERGY (J)", "PACKAGE Energy Distribution by Ruleset (Non Idle)",
            "PACKAGE Energy (J)", "Violin_PACKAGE_By_Ruleset_Non_Idle")

# Generate graphs for Projects without outliers - PP0 & PP1
plot_energy_p0_p1(project_energy_non_idle_no_outliers, "Project", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)",
                  "PP1 Std (J)", "Average PP0 & PP1 Energy Consumption per Project (Non Idle, Without Outliers)",
                  "Energy (J)", "Energy_Consumption_PP0_PP1_By_Project_Non_Idle_without_outliers")

# Generate graphs for Projects without outliers - All energy types
plot_energy_all(project_energy_non_idle_no_outliers, "Project",
                ["PP0 Mean (J)", "PP1 Mean (J)", "PACKAGE Mean (J)"],
                ["PP0 Std (J)", "PP1 Std (J)", "PACKAGE Std (J)"],
                ['blue', 'red', 'green'],
                "Average Energy Consumption per Project (Non Idle, Without Outliers)", "Energy (J)",
                "Energy_Consumption_All_By_Project_Non_Idle_without_outliers")

# Generate graphs for Rulesets without outliers - PP0 & PP1
plot_energy_p0_p1(ruleset_energy_non_idle_no_outliers, "Ruleset", "PP0 Mean (J)", "PP1 Mean (J)", "PP0 Std (J)",
                  "PP1 Std (J)", "Average PP0 & PP1 Energy Consumption per Ruleset (Non Idle, Without Outliers)",
                  "Energy (J)", "Energy_Consumption_PP0_PP1_By_Ruleset_Non_Idle_without_outliers")

# Generate graphs for Rulesets without outliers - All energy types
plot_energy_all(ruleset_energy_non_idle_no_outliers, "Ruleset",
                ["PP0 Mean (J)", "PP1 Mean (J)", "PACKAGE Mean (J)"],
                ["PP0 Std (J)", "PP1 Std (J)", "PACKAGE Std (J)"],
                ['blue', 'red', 'green'],
                "Average Energy Consumption per Ruleset (Non Idle, Without Outliers)", "Energy (J)",
                "Energy_Consumption_All_By_Ruleset_Non_Idle_without_outliers")

# Generate boxplots without outliers
plot_boxplot(df_non_idle_no_outliers, "Project", "PP0_ENERGY (J)",
             "PP0 Energy Consumption by Project (Non Idle, Without Outliers)", "PP0 Energy (J)",
             "Boxplot_PP0_By_Project_Non_Idle_without_outliers")
plot_boxplot(df_non_idle_no_outliers, "Project", "PP1_ENERGY (J)",
             "PP1 Energy Consumption by Project (Non Idle, Without Outliers)", "PP1 Energy (J)",
             "Boxplot_PP1_By_Project_Non_Idle_without_outliers")
plot_boxplot(df_non_idle_no_outliers, "Project", "PACKAGE_ENERGY (J)",
             "PACKAGE Energy Consumption by Project (Non Idle, Without Outliers)", "PACKAGE Energy (J)",
             "Boxplot_PACKAGE_By_Project_Non_Idle_without_outliers")
plot_boxplot(df_non_idle_no_outliers, "Ruleset", "PP0_ENERGY (J)",
             "PP0 Energy Consumption by Ruleset (Non Idle, Without Outliers)", "PP0 Energy (J)",
             "Boxplot_PP0_By_Ruleset_Non_Idle_without_outliers")
plot_boxplot(df_non_idle_no_outliers, "Ruleset", "PP1_ENERGY (J)",
             "PP1 Energy Consumption by Ruleset (Non Idle, Without Outliers)", "PP1 Energy (J)",
             "Boxplot_PP1_By_Ruleset_Non_Idle_without_outliers")
plot_boxplot(df_non_idle_no_outliers, "Ruleset", "PACKAGE_ENERGY (J)",
             "PACKAGE Energy Consumption by Ruleset (Non Idle, Without Outliers)", "PACKAGE Energy (J)",
             "Boxplot_PACKAGE_By_Ruleset_Non_Idle_without_outliers")

# Generate violin plots without outliers
plot_violin(df_non_idle_no_outliers, "Project", "PP0_ENERGY (J)",
            "PP0 Energy Distribution by Project (Non Idle, Without Outliers)", "PP0 Energy (J)",
            "Violin_PP0_By_Project_Non_Idle_without_outliers")
plot_violin(df_non_idle_no_outliers, "Project", "PP1_ENERGY (J)",
            "PP1 Energy Distribution by Project (Non Idle, Without Outliers)", "PP1 Energy (J)",
            "Violin_PP1_By_Project_Non_Idle_without_outliers")
plot_violin(df_non_idle_no_outliers, "Project", "PACKAGE_ENERGY (J)",
            "PACKAGE Energy Distribution by Project (Non Idle, Without Outliers)", "PACKAGE Energy (J)",
            "Violin_PACKAGE_By_Project_Non_Idle_without_outliers")
plot_violin(df_non_idle_no_outliers, "Ruleset", "PP0_ENERGY (J)",
            "PP0 Energy Distribution by Ruleset (Non Idle, Without Outliers)", "PP0 Energy (J)",
            "Violin_PP0_By_Ruleset_Non_Idle_without_outliers")
plot_violin(df_non_idle_no_outliers, "Ruleset", "PP1_ENERGY (J)",
            "PP1 Energy Distribution by Ruleset (Non Idle, Without Outliers)", "PP1 Energy (J)",
            "Violin_PP1_By_Ruleset_Non_Idle_without_outliers")
plot_violin(df_non_idle_no_outliers, "Ruleset", "PACKAGE_ENERGY (J)",
            "PACKAGE Energy Distribution by Ruleset (Non Idle, Without Outliers)", "PACKAGE Energy (J)",
            "Violin_PACKAGE_By_Ruleset_Non_Idle_without_outliers")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_dual_violin(df, x_col, y_col1, y_col2, title, y_label1, y_label2, filename):
    """
    Plot two violin plots side by side with dual y-axes for different energy metrics.

    Parameters:
    df (DataFrame): Input dataframe with energy data
    x_col (str): Column name to use for x-axis categories
    y_col1 (str): First energy metric column name (PP0 Energy in J)
    y_col2 (str): Second energy metric column name (Package Energy in J, converted to MJ)
    title (str): Plot title
    y_label1 (str): Y-axis label for PP0 Energy
    y_label2 (str): Y-axis label for Package Energy
    filename (str): Output filename without extension
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    df = df.copy()
    df[y_col2] = df[y_col2] / 1e6

    ax2 = ax1.twinx()

    sns.violinplot(
        x=x_col, y=y_col1, data=df,
        ax=ax1, color="blue", inner="quart", alpha=0.6
    )

    sns.violinplot(
        x=x_col, y=y_col2, data=df,
        ax=ax2, color="green", inner="quart", alpha=0.6
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    ax1.set_ylabel(y_label1, fontsize=12, color="blue")
    ax2.set_ylabel(y_label2, fontsize=12, color="green")

    plt.title(title, fontsize=14)

    plt.tight_layout()

    plt.savefig(f'images/{filename}.png', bbox_inches="tight")
    plt.close()


# Generate dual violin plots by Project - with outliers
plot_dual_violin(
    df_non_idle,
    "Project",
    "PP0_ENERGY (J)",
    "PACKAGE_ENERGY (J)",
    "PP0 vs Package Energy Distribution by Project (Non Idle)",
    "PP0 Energy (J)",
    "Package Energy (MJ)",
    "Dual_Violin_PP0_Package_By_Project_Non_Idle"
)

# Generate dual violin plots by Ruleset - with outliers
plot_dual_violin(
    df_non_idle,
    "Ruleset",
    "PP0_ENERGY (J)",
    "PACKAGE_ENERGY (J)",
    "PP0 vs Package Energy Distribution by Ruleset (Non Idle)",
    "PP0 Energy (J)",
    "Package Energy (MJ)",
    "Dual_Violin_PP0_Package_By_Ruleset_Non_Idle"
)

# Generate dual violin plots by Project - without outliers
plot_dual_violin(
    df_non_idle_no_outliers,
    "Project",
    "PP0_ENERGY (J)",
    "PACKAGE_ENERGY (J)",
    "PP0 vs Package Energy Distribution by Project (Non Idle, Without Outliers)",
    "PP0 Energy (J)",
    "Package Energy (MJ)",
    "Dual_Violin_PP0_Package_By_Project_Non_Idle_without_outliers"
)

# Generate dual violin plots by Ruleset - without outliers
plot_dual_violin(
    df_non_idle_no_outliers,
    "Ruleset",
    "PP0_ENERGY (J)",
    "PACKAGE_ENERGY (J)",
    "PP0 vs Package Energy Distribution by Ruleset (Non Idle, Without Outliers)",
    "PP0 Energy (J)",
    "Package Energy (MJ)",
    "Dual_Violin_PP0_Package_By_Ruleset_Non_Idle_without_outliers"
)
