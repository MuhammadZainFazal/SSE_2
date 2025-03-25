import time
import os

from EnergiBridgeRunner import EnergiBridgeRunner

from analysis.analyzer import PowerAnalyzer
from scenario.runner import Runner
from scenario.scenario import Scenario

# Directories for Java files and rulesets
project_dir = os.path.dirname(os.path.abspath(__file__))
java_files_directory = [os.path.join(project_dir, 'java_files', 'junit5-r5.12.1')]
ruleset_paths = [os.path.join(project_dir, 'rulesets', 'category', 'java', f) for f in os.listdir(os.path.join(project_dir, 'rulesets', 'category', 'java')) if f.endswith('.xml')]
#ruleset_paths = [os.path.join(project_dir, 'rulesets\\rulesets\\java\\quickstart.xml', )]


energibridge_runner = EnergiBridgeRunner()
print(java_files_directory)
scenarios = [
    Scenario(
        name=os.path.basename(file) + ':' + os.path.basename(rule),
        description=f"Test {os.path.basename(file)}",
        src_dir=os.path.dirname(file),
        ruleset_path=rule,
        runner=energibridge_runner
    )
    for file in java_files_directory
    for rule in ruleset_paths
]

runner = Runner(scenarios, number_of_runs=10)
runner.run()

for scenario in scenarios:
    analyzer = PowerAnalyzer(scenario.dataframe_file, scenario.name)
    analyzer.generate_report()