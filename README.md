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
6. Run LibreHardwareMonitor/OpenHardwareMonitor or RAPL service (with admin access).
7. Download pmd, unzip it and add its path to the system variables
8. Download the following open source projects : JUnit 5 - https://github.com/junit-team/junit5/archive/refs/tags/r5.12.1.zip, spring-framework - https://github.com/spring-projects/spring-framework/archive/refs/heads/main.zip, JabRef - https://github.com/JabRef/jabref/archive/refs/heads/main.zip, Terasology - https://github.com/MovingBlocks/Terasology/archive/refs/heads/develop.zip and unzip them inside the java-files folder.
9. Run `python pipeline.py` from terminal with admin permissions.
10. To create more visuals you can run visuals.py