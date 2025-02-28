# SSE

To run the project:
1. Install LibreHardwareMonitor/OpenHardwareMonitor to access the CPU registry.
2. Download Energibridge from the release
3. Install pyEnergiBridge:
```shell
    git clone https://github.com/luiscruz/pyEnergiBridge.git
    cd pyEnergiBridge
    pip install .
```
4. Configure path to energibridge.exe in `pyenergibridge_config.json` in the root directory on this project
5. Install Ollama
6. Run Ollama, LibreHardwareMonitor/OpenHardwareMonitor or RAPL service, and run `python pipeline.py`
from terminal with admin permissions
7. To create more visuals you can run visuals.py
