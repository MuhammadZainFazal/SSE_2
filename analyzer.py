import os
import re
from collections import defaultdict

class PMDReportAnalyzer:
    def __init__(self, report_dir, global_summary_file):
        self.report_dir = report_dir
        self.global_summary_file = global_summary_file

    def analyze_reports(self):
        global_error_counts = defaultdict(int)
        grouped_error_counts = defaultdict(lambda: defaultdict(int))  # (project, ruleset) -> error counts

        for report_file in os.listdir(self.report_dir):
            if report_file.endswith(".txt"):
                report_path = os.path.join(self.report_dir, report_file)

                # Expect filename format: <project>_<ruleset>_pmd_report.txt
                parts = report_file.replace("_pmd_report.txt", "").split("_")
                if len(parts) >= 2:
                    project = parts[0]
                    ruleset = "_".join(parts[1:])  # handles compound names like "code_style"
                    key = (project, ruleset)

                    self.process_report(report_path, global_error_counts, grouped_error_counts[key])

        self.save_results(global_error_counts, self.global_summary_file)
        self.save_grouped_results(grouped_error_counts)

    def process_report(self, report_path, global_error_counts, grouped_error_counts):
        with open(report_path, "r", encoding="utf-8") as file:
            for line in file:
                match = re.search(r'\d+:\s*(\w+):', line)
                if match:
                    error_name = match.group(1)
                    global_error_counts[error_name] += 1
                    grouped_error_counts[error_name] += 1

    def save_results(self, error_counts, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{error}: {count}\n")
        print(f"Global summary saved to {output_file}")

    def save_grouped_results(self, grouped_counts):
        for (project, ruleset), errors in grouped_counts.items():
            filename = f"{project}_{ruleset}_summary.txt"
            file_path = os.path.join(self.report_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                for error, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{error}: {count}\n")
            print(f"{project}/{ruleset} summary saved to {file_path}")

# Usage example:
analyzer = PMDReportAnalyzer("pmd_reports", "error_summary.txt")
analyzer.analyze_reports()
