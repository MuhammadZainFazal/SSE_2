import os
import re
from collections import defaultdict

class PMDFinalReportGenerator:
    def __init__(self, report_dir, final_summary_file):
        self.report_dir = report_dir
        self.final_summary_file = final_summary_file

    def generate_final_report(self):
        project_data = defaultdict(lambda: defaultdict(int))  # project -> ruleset -> error count
        project_top_errors = defaultdict(lambda: defaultdict(list))  # project -> ruleset -> top 5 errors

        for report_file in os.listdir(self.report_dir):
            if report_file.endswith("_summary.txt") and not report_file.startswith("error_summary"):
                report_path = os.path.join(self.report_dir, report_file)
                project, ruleset = report_file.replace("_summary.txt", "").split("_", 1)
                total_errors, top_errors = self.process_report(report_path)
                
                project_data[project][ruleset] = total_errors
                project_top_errors[project][ruleset] = top_errors

        self.save_final_summary(project_data, project_top_errors)

    def process_report(self, report_path):
        total_errors = 0
        error_counts = {}

        with open(report_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    error_name, count = parts[0], int(parts[1])
                    total_errors += count
                    error_counts[error_name] = count

        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return total_errors, top_errors

    def save_final_summary(self, project_data, project_top_errors):
        with open(self.final_summary_file, "w", encoding="utf-8") as file:
            for project, rulesets in project_data.items():
                total_project_errors = sum(rulesets.values())
                file.write(f"Project: {project}\n")
                file.write(f"Total Errors: {total_project_errors}\n")
                for ruleset, count in rulesets.items():
                    file.write(f"  {ruleset}: {count} errors\n")
                    top_errors = project_top_errors[project][ruleset]
                    if top_errors:
                        file.write("    Top 5 Errors:\n")
                        for error_name, error_count in top_errors:
                            percentage = (error_count / count) * 100 if count > 0 else 0
                            file.write(f"      {error_name}: {error_count} occurrences ({percentage:.2f}%)\n")
                file.write("\n")
        print(f"Final summary saved to {self.final_summary_file}")

# Usage example:
final_report = PMDFinalReportGenerator("pmd_reports", "final_error_summary.txt")
final_report.generate_final_report()
