import time

import ollama

from EnergiBridgeRunner import EnergiBridgeRunner
from langchain_ollama import ChatOllama

from analysis.analyzer import PowerAnalyzer
from scenario.runner import Runner
from scenario.scenario import Scenario

#local_models = ["qwen2.5-coder:1.5b-instruct-q5_0", "deepseek-r1:1.5b", "llama3.2:1b"]
local_models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]

# Pull the models if they are not available locally
for model in local_models:
    available_models = set(map(lambda x: x.model, ollama.list().models))
    if model not in available_models:
        print("Pulling model", model)
        ollama.pull(model)

energibridge_runner = EnergiBridgeRunner()

scenarios = []
for model in local_models:
    scenarios.append(Scenario(name=model, description=f"Test {model}", model=model, prompt="What is your name?", runner=energibridge_runner))

# runner = Runner(scenarios, number_of_runs=30)
# runner.run()

for scenario in scenarios:
    analyzer = PowerAnalyzer(scenario.dataframe_file, scenario.name)
    analyzer.generate_report()