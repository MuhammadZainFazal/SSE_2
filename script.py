import time

import ollama

from EnergiBridgeRunner import EnergiBridgeRunner
from langchain_ollama import ChatOllama  # Assuming this is the correct import for the LLM interface


# List of local models for the experiment
local_models = ["qwen2.5-coder:1.5b-instruct-q5_0"]

for model in local_models:
    available_models = set(map(lambda x: x.model, ollama.list().models))
    if model not in available_models:
        print("Pulling model", model)
        ollama.pull(model)

# Initialize LLM model (ChatOllama)
model = local_models[0]
llm = ChatOllama(model=model, temperature=0, num_ctx=16384)

runner = EnergiBridgeRunner()

prompt = "Please summarize the benefits of energy-efficient computing."
messages = [
                ("human", prompt)
            ]

runner.start(results_file="output.csv")
start_time = time.time()

# Get the response from the LLM
response = llm.invoke(messages)
end_time = time.time()

# Stop the data collection and retrieve results
energy, duration = runner.stop()

# Output the results
print(f"Energy consumption (J): {energy}")
print(f"Execution time (s): {duration}")
print(f"LLM Response: {response}")
