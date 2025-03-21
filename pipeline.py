import time
import os

from EnergiBridgeRunner import EnergiBridgeRunner

from analysis.analyzer import PowerAnalyzer
from scenario.runner import Runner
from scenario.scenario import Scenario

# Directories for Java files and rulesets
project_dir = os.path.dirname(os.path.abspath(__file__))
java_files_directory = [os.path.join(project_dir, 'java_files', f) for f in os.listdir(os.path.join(project_dir, 'java_files')) if f.endswith('.java')]
#ruleset_paths = [os.path.join(project_dir, 'rulesets', f) for f in os.listdir(os.path.join(project_dir, 'rulesets')) if f.endswith('.xml')]
ruleset_paths = [os.path.join(project_dir, 'rulesets\\rulesets\\java\\quickstart.xml', )]


energibridge_runner = EnergiBridgeRunner()
print(java_files_directory)
scenarios = [
    Scenario(
        name=os.path.basename(file),
        description=f"Test {os.path.basename(file)}",
        src_dir=os.path.dirname(file),
        ruleset_path=rule,
        runner=energibridge_runner
    )
    for file in java_files_directory
    for rule in ruleset_paths
]

runner = Runner(scenarios, number_of_runs=6)
runner.run()

for scenario in scenarios:
    analyzer = PowerAnalyzer(scenario.dataframe_file, scenario.name)
    analyzer.generate_report()