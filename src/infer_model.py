from transformers import pipeline

# Step 1: Load the Model from Hugging Face Hub
model_id = "satishgonella/devops-log-classifier"  # Replace with your Hugging Face model repo name
classifier = pipeline("text-classification", model=model_id)

# Step 2: Log Examples for Inference
logs = [
    "Error: Kubernetes pod out of memory.",
    "Warning: Disk usage has reached 90%.",
    "Info: Backup process completed successfully.",
    "Error: Jenkins pipeline failed at step 3.",
    "Warning: High memory usage detected on node01."
]

# Step 3: Run Inference
print("Log Classification Results:")
for log in logs:
    result = classifier(log)
    print(f"Log: {log} => Prediction: {result[0]['label']}")
