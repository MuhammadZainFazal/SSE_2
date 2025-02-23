import time

from EnergiBridgeRunner import EnergiBridgeRunner

runner = EnergiBridgeRunner()
runner.start(results_file="output_test123.csv")


# Perform some task
time.sleep(10)

# Stop the data collection and retrieve results
energy, duration = runner.stop()
print(f"Energy consumption (J): {energy}; Execution time (s): {duration}")