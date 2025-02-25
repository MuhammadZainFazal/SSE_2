import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

from scipy.stats import energy_distance


class PowerAnalyzer:
    def __init__(self, parquet_file, name):
        """Initialize the PowerAnalyzer with a parquet file path.

        Args:
            parquet_file (str): Path to the parquet file containing the data
        """
        self.parquet_file = parquet_file
        self.name = name
        self.data = None
        self.runs = None
        self.output_dir = "energy_analysis_output"

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True)

        # Load the data
        self.load_data()

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

    def analyze_energy_by_run(self):
        """Analyze energy usage grouped by run."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        energy_col = 'PP0_ENERGY (J)'
        if energy_col not in self.data.columns:
            print("No energy data found in the dataset")
            return

        # Group by run and calculate statistics
        energy_stats = self.data.groupby('run')[energy_col].agg(['mean', 'min', 'max', 'std'])
        energy_stats.columns = ['Average Energy (J)', 'Min Energy (J)', 'Max Energy (J)', 'Std Dev (J)']

        return energy_stats

    def plot_energy_by_run(self, save=True):
        """Plot system energy usage over time for each run."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        energy_col = 'PP0_ENERGY (J)'
        if energy_col not in self.data.columns:
            print("No energy data found in the dataset")
            return

        if 'relative_time' not in self.data.columns:
            print("No time data available for plotting")
            return

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot each run with a different color
        for run in self.runs:
            run_data = self.data[self.data['run'] == run]
            plt.plot(run_data['relative_time'], run_data[energy_col],
                     label=f'Run {run}', alpha=0.8)

        plt.title('System Energy Usage Over Time by Run')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if save:
            file_name = os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_energy_by_run.png")
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {file_name}")

        plt.tight_layout()
        return plt.gcf()

    def plot_average_energy_with_error(self, save=True):
        """
        Plot average system energy usage over time across all runs,
        with error bars representing 2 standard deviations.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        energy_col = 'PP0_ENERGY (J)'
        if energy_col not in self.data.columns:
            print("No energy data found in the dataset")
            return

        if 'relative_time' not in self.data.columns:
            print("No time data available for plotting")
            return

        plt.figure(figsize=(12, 6))

        # We need to create a common time axis for averaging
        # First, find the min and max relative time across all runs
        max_times = []
        for run in self.runs:
            run_data = self.data[self.data['run'] == run]
            max_times.append(run_data['relative_time'].max())

        max_time = min(max_times)  # Use the minimum of the max times to ensure all runs have data

        # Create a common time axis with 70 points (we have 10s per run, interval 200ms, so each run should have 50 points but energibridge isn't that accurate)
        common_times = np.linspace(0, max_time, 70)

        # For each run, interpolate the energy values at these common times
        all_energies = []

        for run in self.runs:
            run_data = self.data[self.data['run'] == run]

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
        plt.plot(common_times, mean_energy, 'b-', linewidth=2, label='Average Energy')
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
            file_name = os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_average_energy_with_error.png")
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {file_name}")

        plt.tight_layout()
        return plt.gcf()
    
    

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("Generating comprehensive energy and CPU usage analysis report...")

        energy_stats = self.analyze_energy_by_run()
        if energy_stats is not None:
            print("\nEnergy Usage Statistics by Run:")
            print(energy_stats)

            # Save to CSV
            energy_stats.to_csv(os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_energy_stats.csv"))

        plot_energy_by_run = self.plot_energy_by_run(save=True)
        plot_average_energy = self.plot_average_energy_with_error(save=True)

        print(f"\nAnalysis complete. Reports and visualizations saved to {self.output_dir}/")
        return True