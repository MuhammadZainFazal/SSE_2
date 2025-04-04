import os
import subprocess

class PMDRunner:
    def __init__(self, src_root, ruleset_dir, output_dir, logger):
        self.src_root = src_root  # Directory containing multiple Java projects
        self.ruleset_dir = ruleset_dir  # Directory containing multiple ruleset XML files
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_pmd_on_projects(self):
        """Runs PMD on each project inside the java_files directory with each ruleset and saves output."""
        rulesets = [os.path.join("rulesets" , "category", self.ruleset_dir, f) for f in os.listdir("rulesets\\category\\java") if f.endswith(".xml")]
        print("hi")
        for project in os.listdir(self.src_root):
            project_path = os.path.join(self.src_root, project)
            if os.path.isdir(project_path):  # Ensure it's a directory
                for ruleset in rulesets:
                    self.run_pmd(project, project_path, ruleset)
                    print("mibombo")
        print("woops")
    
    def run_pmd(self, project_name, project_path, ruleset):
        """Run PMD on a single project with a specific ruleset and save results."""
        ruleset_name = os.path.basename(ruleset).replace(".xml", "")
        self.logger.info(f"Running PMD on project: {project_name} with ruleset: {ruleset_name}")
        output_file = os.path.join(self.output_dir, f"{project_name}_{ruleset_name}_pmd_report.txt")
        
        pmd_command = [
            "pmd", "check",
            "-d", project_path,
            "-R", ruleset,
            "-f", "text",
            "--no-fail-on-violation", "--no-fail-on-error"
        ]
        
        with open(output_file, "w") as f:
            process = subprocess.run(" ".join(pmd_command), shell=True, stdout=f, stderr=subprocess.STDOUT)
        
        self.logger.info(f"PMD report saved to {output_file}")

import logging

def setup_logger():
    logger = logging.getLogger("PMDLogger")
    logger.setLevel(logging.INFO)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger

# Usage example:
logger = setup_logger()
print("boop")
runner = PMDRunner("java_files", "java", "pmd_reports", logger)
print("hello")
runner.run_pmd_on_projects()
