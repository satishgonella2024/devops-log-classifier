from transformers import pipeline

# Step 1: Load the Model from Hugging Face Hub
model_id = "satishgonella/devops-log-classifier"  # Replace with your Hugging Face model repo name
classifier = pipeline("text-classification", model=model_id)

label_mapping = {0: "ERROR", 1: "WARNING", 2: "INFO"}

test_logs = [
    "Error: Kubernetes pod out of memory.",
    "Warning: Disk usage has reached 90%.",
    "Info: Backup process completed successfully.",
    "Error: Jenkins pipeline failed at step 3.",
    "Warning: High memory usage detected on node01.",
]

for log in test_logs:
    prediction = classifier(log)
    label = label_mapping[int(prediction[0]['label'].split('_')[1])]
    print(f"Log: {log} => Prediction: {label}")
