import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

# Update this path if your version number is different!
LOG_DIR = "tb_logs/endoscopy_classification/version_4"

def print_all_metrics():
    if not os.path.exists(LOG_DIR):
        print(f"Directory not found: {LOG_DIR}")
        return

    print(f"Loading logs from {LOG_DIR}...\n")

    # Load the event accumulator
    event_acc = EventAccumulator(LOG_DIR)
    event_acc.Reload()

    # Get all available scalar metrics
    tags = event_acc.Tags().get('scalars', [])

    # The metrics we want to extract and display
    metrics_to_find = ['val/F1', 'val/Precision', 'val/Recall', 'val/AUROC']
    
    # Dictionary to group metrics by their logged step: {step: {metric_name: value}}
    step_data = defaultdict(dict)

    # Extract data for each metric
    for metric in metrics_to_find:
        if metric in tags:
            events = event_acc.Scalars(metric)
            for event in events:
                step_data[event.step][metric] = event.value
        else:
            print(f"Warning: '{metric}' not found in logs yet.")

    if not step_data:
        print("\nNo matching metrics found. Currently available metrics:", tags)
        return

    # Print the formatted table header
    header = f"{'Step':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'AUROC':<10}"
    print(header)
    print("-" * len(header))

    # Sort by step and print each row
    for step in sorted(step_data.keys()):
        data = step_data[step]
        
        # safely get the metric value or default to N/A if it's missing for some reason
        f1 = f"{data.get('val/F1', 0):.4f}" if 'val/F1' in data else "N/A"
        prec = f"{data.get('val/Precision', 0):.4f}" if 'val/Precision' in data else "N/A"
        rec = f"{data.get('val/Recall', 0):.4f}" if 'val/Recall' in data else "N/A"
        auroc = f"{data.get('val/AUROC', 0):.4f}" if 'val/AUROC' in data else "N/A"
        
        print(f"{step:<10} | {f1:<10} | {prec:<10} | {rec:<10} | {auroc:<10}")

if __name__ == "__main__":
    print_all_metrics()
