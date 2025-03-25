import os
import subprocess
import time
from pathlib import Path

import ollama
import pandas as pd

from EnergiBridgeRunner import EnergiBridgeRunner
import logging

logger = logging.getLogger("ScenarioLogger")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Scenario:
    def __init__(self, name, description, src_dir, ruleset_path, runner):
        self.output_dir = "energibridge_output"
        Path(self.output_dir).mkdir(exist_ok=True)
        self.name = name
        self.description = description

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.join(script_dir, src_dir)
        self.ruleset_path = os.path.join(script_dir, ruleset_path)
        
        self.runner = runner
        self.logger = logger  # Fixed the duplicate assignment
        self.output_file = os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_output.csv")
        self.dataframe_file = os.path.join(self.output_dir, f"{self.name.replace(':', '_')}_dataframe.parquet")
        self.logger.info(f"Output file expected at: {os.path.abspath(self.output_file)}")
        self.logger.info(f"Dataframe file expected at: {os.path.abspath(self.dataframe_file)}")

        # Initialize run counter by checking existing data
        self.current_run = self._get_next_run_number()

    def _get_next_run_number(self):
        """Determine the next run number based on existing data."""
        if os.path.exists(self.dataframe_file):
            try:
                existing_data = pd.read_parquet(self.dataframe_file)
                if 'run' in existing_data.columns and not existing_data.empty:
                    return existing_data['run'].max() + 1
            except Exception as e:
                self.logger.error(f"Error reading existing run numbers: {e}")
        return 1  # Start with run 1 if no existing data

    def run(self):
        """Run the scenario and collect results."""
        self.logger.info(f"Running scenario: {self.name} (Run #{self.current_run})")

        self.runner.start(results_file=self.output_file)
        time.sleep(5)
        self.run_pmd()

    def run_pmd(self):
        """Run the scenario with PMD, ensuring it runs for at least 10 seconds."""

        # Start PMD
        self.logger.info("Running PMD...")
        pmd_command = [
            "pmd", "check",
            "-d", self.src_dir,
            "-R", self.ruleset_path,
            "-f", "text", "--no-fail-on-violation"
        ]
        self.logger.info(f"Running PMD command: {' '.join(pmd_command)}")

        process = subprocess.run(pmd_command, check=True, shell=True)

        energy, duration = self.runner.stop()
        self.logger.info(f"Energy consumption: {energy} J, Duration: {duration} s")
        self.process_results()

    def process_results(self):
        """Load results from output_file and append them to dataframe_file."""
        if not os.path.exists(self.output_file):
            self.logger.warning(f"Output file {self.output_file} not found.")
            return

        try:
            # Load new data from CSV (It gets overwritten every time when the scenario is run)
            new_data = pd.read_csv(self.output_file)

            # Add the run number to all rows in the new data
            new_data['run'] = self.current_run

            #self.logger.info(f"New data for run #{self.current_run}: {new_data}")

            # Load existing data if parquet file exists and combine with new data
            if os.path.exists(self.dataframe_file):
                existing_data = pd.read_parquet(self.dataframe_file)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined_data = new_data

            # Save back to parquet file
            combined_data.to_parquet(self.dataframe_file, index=False)
            os.remove(self.output_file)
            self.logger.info(f"Results successfully processed and saved to {self.dataframe_file}")

            # Increment run counter for next run
            self.current_run += 1
        except Exception as e:
            self.logger.error(f"Error processing results: {e}")


