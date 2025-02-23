import os
import time

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
        self.name = name
        self.description = description
        self.model = model
        self.prompt = prompt
        self.runner = runner
        self.messages = [("human", self.prompt)]
        self.llm = ChatOllama(model=self.model, temperature=temperature, num_ctx=num_ctx)
        self.logger = self.logger = logger
        self.output_file = f"{self.name.replace(':', '_')}_output.csv"
        self.dataframe_file = f"{self.name.replace(':', '_')}_dataframe.parquet"
        self.logger.info(f"Output file expected at: {os.path.abspath(self.output_file)}")
        self.logger.info(f"Dataframe file expected at: {os.path.abspath(self.dataframe_file)}")

    def run(self):
        """Run the scenario and collect results."""
        self.logger.info(f"Running scenario: {self.name}")
        self.runner.start(results_file=self.output_file)
        start_time = time.time()

        response = self.llm.invoke(self.messages)
        end_time = time.time()

        energy, duration = self.runner.stop()
        self.logger.info(f"Response from LLM: {response}")
        self.process_results()


    def process_results(self):
        """Load results from output_file and append them to dataframe_file."""
        if not os.path.exists(self.output_file):
            self.logger.warning(f"Output file {self.output_file} not found.")
            return

        try:
            # Load new data from CSV (It gets overwritten every time when the scenario is run)
            new_data = pd.read_csv(self.output_file)
            logger.info(f"New data: {new_data}")

            # Load existing data if parquet file exists and combine with new data
            if os.path.exists(self.dataframe_file):
                existing_data = pd.read_parquet(self.dataframe_file)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined_data = new_data

            # Save back to parquet file
            combined_data.to_parquet(self.dataframe_file, index=False)
            self.logger.info(f"Results successfully processed and saved to {self.dataframe_file}")
        except Exception as e:
            self.logger.error(f"Error processing results: {e}")


