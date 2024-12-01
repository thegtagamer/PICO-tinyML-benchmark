import os
import time
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import psutil  # For monitoring CPU and memory usage

# Constants
MODEL_PATH = "model.tflite"  # Path to the TFLite model
DATASET_DIR = "./dataset"  # Directory containing images
VISUALIZATION_DIR = "./visualizations"
RESULTS_FILE = os.path.join(VISUALIZATION_DIR, "gesture_benchmark_results.csv")
MAX_INFERENCES = 100  # Maximum number of inferences to perform

# Create visualization directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Load dataset
def load_dataset(dataset_dir, max_files=MAX_INFERENCES):
    """
    Load image paths from the dataset directory.
    """
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= max_files:
                    break
        if len(image_paths) >= max_files:
            break
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {dataset_dir}")
    return image_paths

# Preprocess image
def preprocess_image(image_path, input_shape):
    """
    Preprocess the input image to match the model's input requirements.
    """
    img = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    img = img.resize((input_shape[2], input_shape[1]))  # Resize to match model's height and width
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Benchmark function
def run_benchmark(interpreter, dataset_paths):
    """
    Run inference on the dataset and collect benchmark metrics.
    """
    inference_times = []
    predictions = []
    cpu_usages = []
    memory_usages = []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    for file_path in tqdm(dataset_paths, desc="Benchmarking Inference"):
        input_data = preprocess_image(file_path, input_shape)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Monitor system metrics before inference
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().percent

        # Perform inference
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()

        # Monitor system metrics after inference
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent

        # Calculate metrics
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        cpu_usages.append((cpu_before + cpu_after) / 2)  # Average CPU usage
        memory_usages.append(mem_after)  # Memory usage at the end of inference

        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output)

    return inference_times, predictions, cpu_usages, memory_usages

# Visualization functions
def save_visualizations(inference_times, predictions, cpu_usages, memory_usages):
    """
    Create and save visualizations for benchmark metrics.
    """
    # Inference Time Distribution
    plt.figure()
    plt.hist(inference_times, bins=20, color='blue', alpha=0.7)
    plt.title("Inference Time Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "inference_time_distribution.png"))
    plt.close()

    # CPU Usage Plot
    plt.figure()
    plt.plot(cpu_usages, color='orange')
    plt.title("CPU Usage Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("CPU Usage (%)")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "cpu_usage_plot.png"))
    plt.close()

    # Memory Usage Plot
    plt.figure()
    plt.plot(memory_usages, color='green')
    plt.title("Memory Usage Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Memory Usage (%)")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "memory_usage_plot.png"))
    plt.close()

    # Prediction Distribution
    plt.figure()
    predicted_labels = [np.argmax(pred) for pred in predictions]  # Get class with highest probability
    plt.hist(predicted_labels, bins=len(set(predicted_labels)), color='purple', alpha=0.7)
    plt.title("Prediction Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "prediction_distribution.png"))
    plt.close()

# Save results to CSV
def save_results_to_csv(inference_times, predictions, cpu_usages, memory_usages):
    """
    Save inference results to a CSV file.
    """
    with open(RESULTS_FILE, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Inference Time (s)", "Predicted Label", "CPU Usage (%)", "Memory Usage (%)"])
        for i in range(len(inference_times)):
            predicted_label = np.argmax(predictions[i])  # Get the predicted class
            csvwriter.writerow([i + 1, inference_times[i], predicted_label, cpu_usages[i], memory_usages[i]])
    print(f"Benchmark results saved to {RESULTS_FILE}")

# Main function
def main():
    """
    Main function to load the model, process the dataset, and run benchmarks.
    """
    # Load TFLite model
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Load dataset
    dataset_paths = load_dataset(DATASET_DIR, MAX_INFERENCES)

    # Run benchmark
    print("Running Benchmark...")
    inference_times, predictions, cpu_usages, memory_usages = run_benchmark(interpreter, dataset_paths)

    # Save visualizations
    print("Saving Visualizations...")
    save_visualizations(inference_times, predictions, cpu_usages, memory_usages)

    # Save results to CSV
    print("Saving Results to CSV...")
    save_results_to_csv(inference_times, predictions, cpu_usages, memory_usages)

    print("Benchmark Complete!")

if __name__ == "__main__":
    main()
