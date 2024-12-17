import random
from datetime import datetime, timedelta
import pandas as pd

# Define log levels
LOG_LEVELS = ["INFO", "WARNING", "ERROR"]

# Define pod names, namespaces, and event templates
PODS = ["nginx-pod", "api-server", "db-pod", "redis-pod", "frontend-pod"]
NAMESPACES = ["default", "kube-system", "monitoring", "production"]
EVENTS = {
    "INFO": [
        "Pod {} started successfully in namespace {}.",
        "Deployment {} scaled to {} replicas.",
        "Node {} is ready for scheduling.",
        "Backup job completed successfully."
    ],
    "WARNING": [
        "High memory usage detected in pod {}.",
        "DiskPressure condition detected on node {}.",
        "Slow response time observed for pod {}.",
        "Replica count for deployment {} dropped below threshold."
    ],
    "ERROR": [
        "Pod {} crashed due to out of memory (OOMKilled).",
        "Node {} is NotReady for scheduling pods.",
        "Failed to mount volume for pod {}.",
        "Deployment {} failed to roll out successfully."
    ],
}

# Function to generate a random timestamp
def random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

# Function to generate a single log entry
def generate_log():
    log_level = random.choice(LOG_LEVELS)
    template = random.choice(EVENTS[log_level])
    
    # Populate placeholders in the log template
    if "{}" in template:
        if "deployment" in template.lower():
            resource = f"deployment-{random.randint(1, 10)}"
        elif "node" in template.lower():
            resource = f"node-{random.randint(1, 5)}"
        else:
            resource = random.choice(PODS)
        
        namespace = random.choice(NAMESPACES)
        log_text = template.format(resource, namespace)
    else:
        log_text = template
    
    return {
        "timestamp": random_timestamp(datetime.now() - timedelta(days=7), datetime.now()),
        "log_text": log_text,
        "log_level": log_level
    }

# Function to generate a dataset of logs
def generate_log_dataset(num_logs=1000):
    logs = [generate_log() for _ in range(num_logs)]
    return pd.DataFrame(logs)

# Generate and save the dataset
if __name__ == "__main__":
    log_df = generate_log_dataset(10000)  # Generate 1000 logs
    log_df.to_csv("data/kubernetes_synthetic_logs.csv", index=False)
    print("Synthetic Kubernetes logs saved to 'data/kubernetes_synthetic_logs.csv'.")
