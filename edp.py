import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("energy_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
energy_cols = ['PP0_ENERGY (J)', 'PACKAGE_ENERGY (J)']
# Ensure images folder exists
os.makedirs('images', exist_ok=True)
logger.info("Created or verified 'images' directory")


def parse_filename(filename):
    pattern = r'([^_]+)_(.*)\.xml_dataframe\.parquet'
    match = re.match(pattern, filename)
    if match:
        project, ruleset = match.groups()
        return project, ruleset

    return None, None


def remove_outliers(df, time_col='Time', n_outliers=1):
    """
    Remove all values associated with the shortest and longest runs from the dataframe.

    Parameters:
    df (DataFrame): Input dataframe
    time_col (str): Column name containing time values (default: 'Time')
    n_outliers (int): Number of shortest and longest runs to remove (default: 1)

    Returns:
    DataFrame: Dataframe with outlier runs removed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Removing {n_outliers} shortest and longest runs.")

    # Compute total run time for each run
    run_times = df.groupby('run')[time_col].sum()

    # Identify the runs with the shortest and longest times
    shortest_runs = run_times.nsmallest(n_outliers).index.tolist()
    longest_runs = run_times.nlargest(n_outliers).index.tolist()
    outlier_runs = set(shortest_runs + longest_runs)

    logger.info(f"Removed {len(outlier_runs)} outlier runs: {outlier_runs}")

    # Filter out the runs identified as outliers
    result_df = df[~df['run'].isin(outlier_runs)].copy()

    logger.info(f"Outlier removal complete. Original size: {len(df)}, After removal: {len(result_df)}")
    return result_df.reset_index(drop=True)


def generate_data_without_idle(df, energy_cols=['PP0_ENERGY (J)', 'PACKAGE_ENERGY (J)'], idle_seconds=5):
    """
    Generate data with idle energy consumption subtracted.
    Uses the first n seconds (default 5) of each run as the baseline idle energy

    Args:
        df: The dataframe containing energy measurements
        energy_cols: List of columns containing energy measurements
        idle_seconds: Number of seconds at the beginning of each run to use as idle baseline

    Returns:
        A dataframe with the same structure as df but with idle energy subtracted.
    """
    logger.info(f"Generating data with idle energy subtracted using {idle_seconds}s baseline")

    adjusted_data = df.copy()

    # Make sure all energy columns exist
    existing_energy_cols = [col for col in energy_cols if col in adjusted_data.columns]
    if not existing_energy_cols:
        logger.warning(f"None of the specified energy columns {energy_cols} exist in dataset.")
        return adjusted_data

    for file in adjusted_data['File'].unique():
        # Create a mask for the current run
        run_mask = adjusted_data['File'] == file
        run_data = adjusted_data[run_mask]

        # Only process runs that have data
        if len(run_data) > 0:
            idle_data = run_data[run_data['Time'] <= idle_seconds]

            if len(idle_data) > 0:
                for energy_col in existing_energy_cols:
                    # Calculate the mean energy during the idle period
                    mean_idle_energy = idle_data[energy_col].mean()

                    # Subtract this constant baseline from all measurements in the run
                    adjusted_data.loc[run_mask, energy_col] = run_data[energy_col] - mean_idle_energy

                    # Ensure no negative values (can happen if idle > active in some cases)
                    adjusted_data.loc[run_mask & (adjusted_data[energy_col] < 0), energy_col] = 0

                logger.info(f"Adjusted idle energy for run: {file}")
            else:
                logger.warning(f"Run {file} has no data within the first {idle_seconds} seconds.")

    logger.info(
        f"Generated data with idle energy subtracted using mean energy from first {idle_seconds} seconds as baseline.")
    return adjusted_data


parquet_dir = 'energibridge_output'
logger.info(f"Scanning '{parquet_dir}' directory for parquet files")

try:
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    logger.info(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        raise FileNotFoundError("No parquet files found in 'energibridge_output' folder")

    all_data = []
    processed_files = 0

    for file in parquet_files:
        file_path = os.path.join(parquet_dir, file)
        logger.info(f"Processing file: {file}")

        try:
            df = pd.read_parquet(file_path)

            project, ruleset = parse_filename(file)
            if project and ruleset:
                df['Project'] = project
                df['Ruleset'] = ruleset
                df['File'] = file
                for run in df['File'].unique():
                    run_mask = df['File'] == run
                    df.loc[run_mask, 'Time'] = (df.loc[run_mask, 'Time'] - df.loc[run_mask, 'Time'].min()) / 1e6
                energy_cols = ['PP0_ENERGY (J)', 'PACKAGE_ENERGY (J)']
                df = generate_data_without_idle(df, energy_cols=energy_cols, idle_seconds=5)
                df = remove_outliers(df, n_outliers=3)
                all_data.append(df)
                processed_files += 1
                logger.info(f"Successfully parsed project: '{project}', ruleset: '{ruleset}' from {file}")
            else:
                logger.warning(f"Could not parse project and ruleset from filename: {file}")
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")

    logger.info(f"Successfully processed {processed_files} out of {len(parquet_files)} files")

    if all_data:
        logger.info("Combining data from all files")
        df_no_outliers = pd.concat(all_data, ignore_index=True)

        logger.info(f"Columns in dataset: {df_no_outliers.columns.tolist()}")
        logger.info("Calculating additional metrics")
        df_no_outliers['CPU_USAGE_MEAN'] = df_no_outliers[
            [col for col in df_no_outliers.columns if 'CPU_USAGE' in col]].mean(axis=1)
        df_no_outliers['CPU_FREQUENCY_MEAN'] = df_no_outliers[
            [col for col in df_no_outliers.columns if 'CPU_FREQUENCY' in col]].mean(axis=1)
        if 'GPU0_MEMORY_USED' in df_no_outliers.columns and 'GPU0_MEMORY_TOTAL' in df_no_outliers.columns:
            df_no_outliers['GPU_MEMORY_UTILIZATION'] = (df_no_outliers['GPU0_MEMORY_USED'] / df_no_outliers[
                'GPU0_MEMORY_TOTAL']) * 100
            logger.info("Added GPU memory utilization metric")

        # Calculate EDP on both original and idle-adjusted data
        df_no_outliers['EDP'] = df_no_outliers['PACKAGE_ENERGY (J)'] * df_no_outliers['Time']

        # Process both datasets in parallel
        datasets = {
            'no_idle_no_outliers': df_no_outliers,
        }

        for data_type, current_df in datasets.items():
            logger.info(f"Processing {data_type} dataset")
            total_edp_by_run = current_df.groupby(['Project', 'Ruleset', 'File'])['EDP'].sum().reset_index()
            logger.info(f"Calculated total EDP per run for {data_type} dataset")

            logger.info(f"Generating EDP visualizations for {data_type} dataset")

            # Plot EDP by Project with bar chart
            logger.info(f"Creating EDP by Project bar chart ({data_type})")
            plt.figure(figsize=(12, 8))
            project_edp = total_edp_by_run.groupby('Project')['EDP'].mean().sort_values(ascending=False)
            ax = sns.barplot(x=project_edp.index, y=project_edp.values, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Average Energy Delay Product by Project ({data_type}) - Outliers Removed')
            plt.xlabel('Project')
            plt.ylabel('EDP (J·s)')
            plt.tight_layout()
            plt.savefig(f'images/EDP_by_Project_{data_type}.png')
            plt.close()

            # Plot EDP by Ruleset with bar chart
            logger.info(f"Creating EDP by Ruleset bar chart ({data_type})")
            plt.figure(figsize=(12, 8))
            ruleset_edp = total_edp_by_run.groupby('Ruleset')['EDP'].mean().sort_values(ascending=False)
            ax = sns.barplot(x=ruleset_edp.index, y=ruleset_edp.values, palette='plasma')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Average Energy Delay Product by Ruleset ({data_type}) - Outliers Removed')
            plt.xlabel('Ruleset')
            plt.ylabel('EDP (J·s)')
            plt.tight_layout()
            plt.savefig(f'images/EDP_by_Ruleset_{data_type}.png')
            plt.close()

            # Box plot of EDP by Project
            logger.info(f"Creating EDP box plot by Project ({data_type})")
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='Project', y='EDP', data=total_edp_by_run, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Energy Delay Product Distribution by Project ({data_type}) - Outliers Removed')
            plt.xlabel('Project')
            plt.ylabel('EDP (J·s)')
            plt.tight_layout()
            plt.savefig(f'images/EDP_Boxplot_by_Project_{data_type}.png')
            plt.close()

            # Box plot of EDP by Ruleset
            logger.info(f"Creating EDP box plot by Ruleset ({data_type})")
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='Ruleset', y='EDP', data=total_edp_by_run, palette='plasma')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Energy Delay Product Distribution by Ruleset ({data_type}) - Outliers Removed')
            plt.xlabel('Ruleset')
            plt.ylabel('EDP (J·s)')
            plt.tight_layout()
            plt.savefig(f'images/EDP_Boxplot_by_Ruleset_{data_type}.png')
            plt.close()

            # EDP Violin plots
            logger.info(f"Creating EDP violin plot by Project ({data_type})")
            plt.figure(figsize=(14, 10))
            sns.violinplot(x='Project', y='EDP', hue='Project', data=total_edp_by_run,
                           inner='box', palette='viridis', legend=False)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Energy Delay Product Distribution by Project ({data_type}) - Outliers Removed')
            plt.xlabel('Project')
            plt.ylabel('EDP (J·s)')
            plt.tight_layout()
            plt.savefig(f'images/EDP_Violin_by_Project_{data_type}.png')
            plt.close()

            logger.info(f"Creating EDP violin plot by Ruleset ({data_type})")
            plt.figure(figsize=(14, 10))
            sns.violinplot(x='Ruleset', y='EDP', hue='Ruleset', data=total_edp_by_run,
                           inner='box', palette='plasma', legend=False)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Energy Delay Product Distribution by Ruleset ({data_type}) - Outliers Removed')
            plt.xlabel('Ruleset')
            plt.ylabel('EDP (J·s)')
            plt.tight_layout()
            plt.savefig(f'images/EDP_Violin_by_Ruleset_{data_type}.png')
            plt.close()

            # Heatmap of average EDP by Project and Ruleset
            logger.info(f"Creating EDP heatmap by Project and Ruleset ({data_type})")
            plt.figure(figsize=(16, 12))
            try:
                pivot_df = total_edp_by_run.pivot_table(index='Project', columns='Ruleset', values='EDP',
                                                        aggfunc='mean')
                sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
                plt.title(f'Average Energy Delay Product by Project and Ruleset ({data_type}) - Outliers Removed')
                plt.tight_layout()
                plt.savefig(f'images/EDP_Heatmap_Project_Ruleset_{data_type}.png')
            except Exception as e:
                logger.error(f"Error creating heatmap for {data_type}: {str(e)}")
            plt.close()

            logger.info("Generating Average Energy Over Time by Project plot")

            project_avg_energy = current_df.groupby(['Project', 'Time'])[energy_cols[1]].mean().reset_index()

            # Plotting
            plt.figure(figsize=(14, 8))
            for project in project_avg_energy['Project'].unique():
                project_data = project_avg_energy[project_avg_energy['Project'] == project]
                plt.plot(project_data['Time'], project_data[energy_cols[1]], label=f"{project} {energy_cols[1]}")

            plt.title('Average Energy Over Time by Project')
            plt.xlabel('Time (s)')
            plt.ylabel('Average Energy (J)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'images/Average_Energy_Over_Time_by_Project.png')
            plt.close()
            logger.info("Saved Average Energy Over Time by Project plot")


            logger.info("Generating Average Energy Over Time by Ruleset plot")

            current_df.to_parquet('energibridge_output/current_df.parquet', index=False)
            # Group the data by Ruleset and Time, then calculate the mean of energy columns
            ruleset_avg_energy = current_df.groupby(['Ruleset', 'Time'])[energy_cols[1]].mean().reset_index()
            ruleset_avg_energy.to_parquet('energibridge_output/ruleset_avg_energy.parquet', index=False)

            # Plotting
            plt.figure(figsize=(14, 8))
            for ruleset in ruleset_avg_energy['Ruleset'].unique():
                ruleset_data = ruleset_avg_energy[ruleset_avg_energy['Ruleset'] == ruleset]
                plt.plot(ruleset_data['Time'], ruleset_data[energy_cols[1]], label=f"{ruleset} {energy_cols[1]}")

            plt.title('Average Energy Over Time by Ruleset')
            plt.xlabel('Time (s)')
            plt.ylabel('Average Energy (J)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'images/Average_Energy_Over_Time_by_Ruleset.png')
            plt.close()
            logger.info("Saved Average Energy Over Time by Ruleset plot")

        logger.info("Analysis complete with outlier removal")
    else:
        logger.error("No data was successfully processed from any files.")

except Exception as e:
    logger.error(f"Error occurred: {str(e)}")