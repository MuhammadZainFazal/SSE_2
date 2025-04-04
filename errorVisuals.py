import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set styles
sns.set(style="whitegrid")

# File path
file_path = "final_error_summary.txt"

# Create a directory to save images
output_dir = "pm_error_plots"
os.makedirs(output_dir, exist_ok=True)

# Storage for project data
project_data = {}
current_project = None
current_category = None

# Parsing the file
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()

        # Match project name
        proj_match = re.match(r'^Project:\s+(.*)', line)
        if proj_match:
            current_project = proj_match.group(1)
            project_data[current_project] = {'Total Errors': 0, 'Categories': {}}
            continue

        # Match total errors
        total_match = re.match(r'^Total Errors:\s+(\d+)', line)
        if total_match and current_project:
            project_data[current_project]['Total Errors'] = int(total_match.group(1))
            continue

        # Match category header
        cat_match = re.match(r'^(\w+):\s+(\d+)\s+errors', line)
        if cat_match and current_project:
            current_category = cat_match.group(1)
            category_count = int(cat_match.group(2))
            project_data[current_project]['Categories'][current_category] = category_count
            continue

# ---- Visualization ----

# Prepare data for a single grouped bar plot
categories = list(set([cat for data in project_data.values() for cat in data['Categories'].keys()]))
categories.sort()  # Sort categories alphabetically for consistency

# Prepare the error counts for each project and category
error_counts = {category: [] for category in categories}
project_names = list(project_data.keys())

for project in project_names:
    for category in categories:
        error_counts[category].append(project_data[project]['Categories'].get(category, 0))

# Create a bar plot with different colors for each project
fig, ax = plt.subplots(figsize=(15, 8))

# Create the grouped bar chart
width = 0.15  # Width of the bars
x = range(len(categories))  # Positions for the categories on the x-axis

colors = sns.color_palette("Set2", len(project_names))  # Distinct colors for each project

# Plot bars for each project
for i, project in enumerate(project_names):
    ax.bar(
        [p + width * i for p in x],  # Shift bars for each project
        [error_counts[category][i] for category in categories],
        width=width,
        label=project,
        color=colors[i]
    )

# Labels and title
ax.set_xlabel('Category')
ax.set_ylabel('Number of Errors')
ax.set_title('PMD Errors per Category for All Projects')
ax.set_xticks([p + width * (len(project_names) - 1) / 2 for p in x])
ax.set_xticklabels(categories, rotation=45, ha="right")

# Add a legend
ax.legend(title="Projects")

# Adjust layout
plt.tight_layout()

# Save the combined figure as a .png file
plt.savefig(f"{output_dir}/grouped_errors_for_all_projects.png")
plt.close()  # Close the plot to avoid display
