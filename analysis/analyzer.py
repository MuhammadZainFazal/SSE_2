import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from scipy.stats import energy_distance


class PowerAnalyzer:
    def __init__(self, parquet_file, name, idle_parquet_file=None):
        """Initialize the PowerAnalyzer with a parquet file path.

        Args:
            parquet_file (str): Path to the parquet file containing the data
        """
        self.parquet_file = parquet_file
        self.name = name
        self.idle_parquet_file = idle_parquet_file
        self.data = None
        self.idle_data = None
        self.runs = None
        self.output_dir = "energy_analysis_output"

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True)

        # Load the data
        self.load_data()
        self.load_idle_data()

    def load_data(self):
        """Load data from the parquet file and perform initial processing."""
        try:
            self.data = pd.read_parquet(self.parquet_file)
            print(f"Data loaded successfully with {len(self.data)} rows")

            if 'run' not in self.data.columns and 'RUN' in self.data.columns:
                self.data = self.data.rename(columns={'RUN': 'run'})

            self.runs = self.data['run'].unique()
            print(f"Found {len(self.runs)} unique runs: {self.runs}")

            if 'Time' in self.data.columns:
                # Sort data by run and time
                self.data = self.data.sort_values(['run', 'Time'])

                # Add relative time column (seconds from start of each run)
                self.data['relative_time'] = self.data.groupby('run')['Time'].transform(
                    lambda x: (x - x.iloc[0]) / 1000  # Convert to seconds
                )

            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
    def load_idle_data(self):
        """Load idle data from the parquet file and perform initial processing."""
        if (self.idle_parquet_file == self.parquet_file) or (self.idle_parquet_file is None):
            print("No idle data available for comparison or idle data is the same as the main data.")
            return

        try:
            self.idle_data = pd.read_parquet(self.idle_parquet_file)
            print(f"Sleep data loaded successfully with {len(self.idle_data)} rows")

            if 'run' not in self.idle_data.columns and 'RUN' in self.idle_data.columns:
                self.idle_data = self.idle_data.rename(columns={'RUN': 'run'})

            idle_runs = self.idle_data['run'].unique()
            print(f"Found {len(idle_runs)} unique runs in idle data: {idle_runs}")

            if 'Time' in self.idle_data.columns:
                # Sort data by run and time
                self.idle_data = self.idle_data.sort_values(['run', 'Time'])

                # Add relative time column (seconds from start of each run)
                self.idle_data['relative_time'] = self.idle_data.groupby('run')['Time'].transform(
                    lambda x: (x - x.iloc[0]) / 1000  # Convert to seconds
                )

            return True
        except Exception as e:
            print(f"Error loading idle data: {e}")
            return False

    def analyze_energy_by_run(self, data, energy_col='PP0_ENERGY (J)'):
        """Analyze energy usage grouped by run."""
        if data is None:
            print("No data loaded. Please load data first.")
            return

        if energy_col not in data.columns:
            print("No energy data found in the dataset")
            return

        # Group by run and calculate statistics
        energy_stats = data.groupby('run')[energy_col].agg(['mean', 'min', 'max', 'std'])
        energy_stats.columns = ['Average Energy (J)', 'Min Energy (J)', 'Max Energy (J)', 'Std Dev (J)']

        return energy_stats

    def plot_energy_by_run(self, data, filename_suffix, save=True, energy_col='PP0_ENERGY (J)'):
        """Plot system energy usage over time for each run."""
        if data is None:
            print("No data loaded. Please load data first.")
            return

        if energy_col not in data.columns:
            print("No energy data found in the dataset")
            return

        if 'relative_time' not in data.columns:
            print("No time data available for plotting")
            return

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot each run with a different color
        for run in self.runs:
            run_data = data[data['run'] == run]
            plt.plot(run_data['relative_time'], run_data[energy_col],
                     label=f'Run {run}', alpha=0.8)

        plt.title(f"System Energy Usage Over Time by Run {energy_col}")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if save:
            file_name = os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_{filename_suffix}.png")
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {file_name}")

        plt.tight_layout()
        return plt.gcf()

    def plot_average_energy_with_error(self, data, filename_suffix, save=True, energy_col='PP0_ENERGY (J)'):
        """
        Plot average system energy usage over time across all runs,
        with error bars representing 2 standard deviations.
        """
        if data is None:
            print("No data loaded. Please load data first.")
            return

        if energy_col not in data.columns:
            print("No energy data found in the dataset")
            return

        if 'relative_time' not in data.columns:
            print("No time data available for plotting")
            return

        plt.figure(figsize=(12, 6))

        # We need to create a common time axis for averaging
        # First, find the min and max relative time across all runs
        max_times = []
        for run in self.runs:
            run_data = data[data['run'] == run]
            max_times.append(run_data['relative_time'].max())

        max_time = min(max_times)  # Use the minimum of the max times to ensure all runs have data

        # Create a common time axis with 70 points (we have 10s per run, interval 200ms, so each run should have 50 points but energibridge isn't that accurate)
        common_times = np.linspace(0, max_time, 70)

        # For each run, interpolate the energy values at these common times
        all_energies = []

        for run in self.runs:
            run_data = data[data['run'] == run]

            # Only use data up to the common max time
            run_data = run_data[run_data['relative_time'] <= max_time]

            # Interpolate energy values at the common time points
            interpolated_energy = np.interp(
                common_times,
                run_data['relative_time'],
                run_data[energy_col]
            )

            all_energies.append(interpolated_energy)

        all_energies_array = np.array(all_energies)

        mean_energy = np.mean(all_energies_array, axis=0)
        std_energy = np.std(all_energies_array, axis=0)

        # Plot the average energy with error bands (2 standard deviations)
        plt.plot(common_times, mean_energy, 'b-', linewidth=2, label=f"Average Energy {energy_col}")
        plt.fill_between(
            common_times,
            mean_energy - 2 * std_energy,
            mean_energy + 2 * std_energy,
            alpha=0.3,
            color='blue',
            label='±2σ (95% Confidence)'
        )

        # Add individual runs as light lines if there are not too many
        if len(self.runs) <= 10:
            for i, run in enumerate(self.runs):
                plt.plot(
                    common_times,
                    all_energies[i],
                    alpha=0.3,
                    linewidth=1,
                    linestyle='--',
                    label=f'Run {run}'
                )

        plt.title('Average System Energy Usage Over Time (All Runs)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if save:
            file_name = os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_{filename_suffix}.png")
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {file_name}")

        plt.tight_layout()
        return plt.gcf()

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("Generating comprehensive energy and CPU usage analysis report...")

        energy_stats = self.analyze_energy_by_run(self.data)
        if energy_stats is not None:
            print("\nEnergy Usage Statistics by Run:")
            print(energy_stats)

            # Save to CSV
            energy_stats.to_csv(os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_energy_stats.csv"))

        self.plot_energy_by_run(self.data, 'energy_by_run', save=True)
        self.plot_average_energy_with_error(self.data, 'average_energy_with_error', save=True)

        energy_stats_pp1 = self.analyze_energy_by_run(self.data, energy_col='PP1_ENERGY (J)')
        if energy_stats is not None:
            print("\nEnergy Usage Statistics by Run:")
            print(energy_stats)

            # Save to CSV
            energy_stats.to_csv(os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_energy_stats_pp1.csv"))

        self.plot_energy_by_run(self.data, 'energy_by_run_pp1', save=True, energy_col='PP1_ENERGY (J)')
        self.plot_average_energy_with_error(self.data, 'average_energy_with_error_pp1', save=True, energy_col='PP1_ENERGY (J)')

        print(f"\nAnalysis complete. Reports and visualizations saved to {self.output_dir}/")

        self.generate_report_without_idle()

        return True

    def generate_report_without_idle(self):
        if (self.idle_parquet_file == self.parquet_file) or (self.idle_parquet_file is None):
            print("No idle data available for comparison or idle data is the same as the main data.")
            return

        if self.idle_data is None:
            print("No idle data loaded. Please load idle data first.")
            return

        energy_col = 'PP0_ENERGY (J)'
        if energy_col not in self.data.columns:
            print("No energy data found in the dataset")
            return

        idle_energy_col = 'PP0_ENERGY (J)'
        if idle_energy_col not in self.idle_data.columns:
            print("No energy data found in the idle dataset")
            return
        
        data_without_idle_pp0 = self.generate_data_without_idle()

        energy_stats = self.analyze_energy_by_run(data_without_idle_pp0)
        if energy_stats is not None:
            print("\nEnergy Usage Statistics by Run without Idle:")
            print(energy_stats)

            # Save to CSV
            energy_stats.to_csv(os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_energy_stats_without_idle_pp0.csv"))

        self.plot_energy_by_run(data_without_idle_pp0, 'energy_by_run_without_idle_pp0', save=True)
        self.plot_average_energy_with_error(data_without_idle_pp0, 'average_energy_with_error_without_idle_pp0', save=True)

        data_without_idle_pp1 = self.generate_data_without_idle(energy_col='PP1_ENERGY (J)')

        energy_stats = self.analyze_energy_by_run(data_without_idle_pp1)
        if energy_stats is not None:
            print("\nEnergy Usage Statistics by Run without Idle:")
            print(energy_stats)

            # Save to CSV
            energy_stats.to_csv(
                os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_energy_stats_without_idle_pp1.csv"))

        self.plot_energy_by_run(data_without_idle_pp1, 'energy_by_run_without_idle_pp1', save=True, energy_col='PP1_ENERGY (J)')
        self.plot_average_energy_with_error(data_without_idle_pp1, 'average_energy_with_error_without_idle_pp1', save=True, energy_col='PP1_ENERGY (J)')

        print(f"\nAnalysis complete. Reports and visualizations saved to {self.output_dir}/")
        return True

    def generate_data_without_idle(self, energy_col='PP0_ENERGY (J)'):
        """
        Generate data with idle energy consumption subtracted.
        Returns a dataframe with the same structure as self.data but with idle energy subtracted.
        """
        if self.idle_data is None:
            self.load_idle_data()

        if self.idle_data is None:
            print("No idle data available for comparison.")
            return self.data.copy()

        if energy_col not in self.data.columns or energy_col not in self.idle_data.columns:
            print("Energy column missing in main or idle dataset.")
            return self.data.copy()

        # Create a copy of the original data
        adjusted_data = self.data.copy()

        # Calculate average idle energy at each timestamp across all idle runs
        # Group by relative_time and calculate mean energy
        if 'relative_time' in self.idle_data.columns:
            # We need to create a common time axis for averaging
            # First, find the min and max relative time across all runs
            max_times = []
            for run in self.idle_data['run'].unique():
                run_data = self.idle_data[self.idle_data['run'] == run]
                max_times.append(run_data['relative_time'].max())

            max_time = min(max_times)
            common_times = np.linspace(0, max_time, 70)

            # For each run, interpolate the energy values at these common times
            all_idle_energies = []

            for run in self.idle_data['run'].unique():
                run_data = self.idle_data[self.idle_data['run'] == run]

                # Only use data up to the common max time
                run_data = run_data[run_data['relative_time'] <= max_time]

                # Interpolate energy values at the common time points
                interpolated_energy = np.interp(
                    common_times,
                    run_data['relative_time'],
                    run_data[energy_col]
                )

                all_idle_energies.append(interpolated_energy)

            all_idle_energies_array = np.array(all_idle_energies)
            mean_idle_energy = np.mean(all_idle_energies_array, axis=0)

            # For each run in the actual data, subtract the interpolated idle energy
            for run in self.runs:
                run_data = adjusted_data[adjusted_data['run'] == run]

                # Create a mask for the current run
                run_mask = adjusted_data['run'] == run

                # Only process runs that have data
                if len(run_data) > 0:
                    # Interpolate the idle energy values to match the timestamps in this run
                    interpolated_idle_energy = np.interp(
                        run_data['relative_time'],
                        common_times,
                        mean_idle_energy,
                        left=mean_idle_energy[0],
                        right=mean_idle_energy[-1]
                    )

                    # Subtract the interpolated idle energy from the actual energy
                    adjusted_data.loc[run_mask, energy_col] = run_data[energy_col] - interpolated_idle_energy

                    # Ensure no negative values (can happen if idle > active in some cases)
                    adjusted_data.loc[run_mask & (adjusted_data[energy_col] < 0), energy_col] = 0
        else:
            print("No relative_time column in idle data. Cannot subtract idle energy.")
            return self.data.copy()

        print(f"Generated data with idle energy subtracted.")
        return adjusted_data

    