import os
import subprocess
import time
from pathlib import Path

import ollama
import pandas as pd
from langchain_ollama import ChatOllama

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
    def __init__(self, name, description, model, prompt, runner, temperature=0, num_ctx=16384):
        self.output_dir = "enegibridge_output"
        Path(self.output_dir).mkdir(exist_ok=True)
        self.name = name
        self.description = description
        self.model = model
        self.prompt = prompt
        self.runner = runner
        self.messages = [("human", self.prompt)]
        if (self.model is None) or (self.prompt is None):
            self.llm = None
        else:
            self.llm = ChatOllama(model=self.model, temperature=temperature, num_ctx=num_ctx)
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
        self.start_ollama()
        time.sleep(5)
        self.ensure_no_model_running()
        time.sleep(5)
        self.runner.start(results_file=self.output_file)

        self.run_llm()

    def ensure_no_model_running(self):
        models_data = ollama.ps()
        models = models_data.models

        if not models:
            self.logger.info("No models running. (That's good!)")
            return
        for model in models:
            self.logger.info(f"Stopping model: {model.model}")
            subprocess.run(["ollama", "stop", model.model], check=True)
            time.sleep(2)

        models_data = ollama.ps()
        models = models_data.models
        if not models:
            self.logger.info("No models running. (That's good!)")
        else:
            self.logger.error("Failed to stop all models. (There is leftover model running.)")

    def run_llm(self):
        """Run the scenario with an LLM model."""

        time.sleep(5)
        start_time = time.time()
        while time.time() - start_time < 15:
            self.llm.invoke(self.messages)
            # self.logger.info(f"Response from LLM: {response}")
            # time.sleep(0.1)
        self.stop_ollama()
        time.sleep(40)
        energy, duration = self.runner.stop()
        self.process_results()

    def start_ollama(self):
        """Starts the Ollama server if it's not already running."""
        try:
            self.logger.info("Starting Ollama...")
            subprocess.run(["ollama", "serve"], check=True)
            self.logger.info("Ollama started successfully.")
        except subprocess.CalledProcessError:
            self.logger.info("Ollama might already be running.")

    def stop_ollama(self):
        """Stops Ollama without removing the model."""
        try:
            self.logger.info("Stopping Ollama...")
            self.ensure_no_model_running()
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], check=True)
            self.logger.info("Ollama stopped successfully.")
        except subprocess.CalledProcessError:
            self.logger.error("Failed to stop Ollama.")

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


