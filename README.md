# SSE

To run the project:
1. Install LibreHardwareMonitor or OpenHardwareMonitor to access the CPU registry.
2. Download Energibridge from the [GitHub page](https://github.com/tdurieux/EnergiBridge).
3. Install pyEnergiBridge:
```shell
    git clone https://github.com/luiscruz/pyEnergiBridge.git
    cd pyEnergiBridge
    pip install .
```
4. Create a `pyenergibridge_config.json` file in the root directory on this project.
5. Configure the path to energibridge.exe in `pyenergibridge_config.json`.
```shell
    {
        "binary_path": "<your_absolute_path_to_energibridge.exe>"
    }
```
6. Install [Ollama](https://ollama.com/download).
7. Run Ollama and LibreHardwareMonitor/OpenHardwareMonitor or RAPL service (with admin access).
8. Run `python pipeline.py` from terminal with admin permissions.
9. To create more visuals you can run visuals.py