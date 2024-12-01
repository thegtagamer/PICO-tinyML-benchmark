import os
import time
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite

# Constants
MODEL_PATH = "quantized_model.tflite"
NUM_IMAGES = 100
IMAGE_SIZE = (513, 513)  # Update based on your model
VISUALIZATION_DIR = "./visualizations"
RESULTS_FILE = os.path.join(VISUALIZATION_DIR, "benchmark_results.csv")

# Create visualization directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Simulated dataset for testing
def generate_test_image(size):
    return np.random.randint(0, 256, size=(1, *size, 3), dtype=np.uint8)

# Run Powertop to fetch power consumption data
def get_power_consumption():
    try:
        # Runs `powertop` and fetches the "Power consumption" value
        result = os.popen("powertop --time=5 --csv=temp.csv && tail -n 1 temp.csv").read()
        os.remove("temp.csv")  # Clean up the temporary file
        # Extract the power consumption from the CSV
        power = float(result.split(',')[-2])
        return power
    except Exception:
        return np.nan  # Return NaN if powertop fails

# Benchmark function
def run_benchmark(interpreter, num_images):
    inference_times = []
    cpu_usages = []
    memory_usages = []
    power_consumptions = []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]

    for _ in tqdm(range(num_images), desc="Benchmarking Inference"):
        # Generate a test image
        input_data = generate_test_image(input_shape)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Measure inference time
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()

        # Fetch results
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # Simulate CPU and Memory usage
        cpu_usages.append(np.random.uniform(5, 15))  # Placeholder for actual CPU usage
        memory_usages.append(np.random.uniform(1, 10))  # Placeholder for actual memory usage

        # Measure power consumption
        power_consumptions.append(get_power_consumption())

    return inference_times, cpu_usages, memory_usages, power_consumptions

# Visualization functions
def save_visualizations(inference_times, cpu_usages, memory_usages, power_consumptions):
    # Plot Inference Time
    plt.figure()
    plt.hist(inference_times, bins=20, color='blue', alpha=0.7)
    plt.title("Inference Time Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "inference_time_distribution.png"))
    plt.close()

    # Plot CPU Usage
    plt.figure()
    plt.plot(cpu_usages, color='orange')
    plt.title("CPU Usage Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("CPU Usage (%)")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "cpu_usage_plot.png"))
    plt.close()

    # Plot Memory Usage
    plt.figure()
    plt.plot(memory_usages, color='green')
    plt.title("Memory Usage Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Memory Usage (%)")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "memory_usage_plot.png"))
    plt.close()

    # Plot Power Consumption
    plt.figure()
    plt.plot(power_consumptions, color='red')
    plt.title("Power Consumption Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Power Consumption (W)")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "power_consumption_plot.png"))
    plt.close()

# Save metrics to CSV
def save_results_to_csv(inference_times, cpu_usages, memory_usages, power_consumptions):
    with open(RESULTS_FILE, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Inference Time (s)", "CPU Usage (%)", "Memory Usage (%)", "Power Consumption (W)"])
        for i in range(len(inference_times)):
            csvwriter.writerow([i + 1, inference_times[i], cpu_usages[i], memory_usages[i], power_consumptions[i]])
    print(f"Raw benchmark results saved to {RESULTS_FILE}")

# Main Execution
def main():
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print("Running Benchmark...")
    inference_times, cpu_usages, memory_usages, power_consumptions = run_benchmark(interpreter, NUM_IMAGES)

    print("Saving visualizations...")
    save_visualizations(inference_times, cpu_usages, memory_usages, power_consumptions)

    print("Saving results to CSV...")
    save_results_to_csv(inference_times, cpu_usages, memory_usages, power_consumptions)

    print("Benchmark complete!")

if __name__ == "__main__":
    main()
