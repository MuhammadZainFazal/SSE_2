import random
from time import sleep


class Runner:
    def __init__(self, scenarios, number_of_runs=30):
        self.scenarios = scenarios
        self.number_of_runs = number_of_runs
        self.run_order = []

    def run(self):
        """Run each scenario in the initialized random order."""
        self.initialize_random_order()
        for scenario in self.run_order:
            scenario.run()
            sleep(60)

    def initialize_random_order(self):
        """Initialize a random order of scenarios to run 30 times each."""
        self.run_order = random.sample(self.scenarios * self.number_of_runs, len(self.scenarios) * self.number_of_runs)
