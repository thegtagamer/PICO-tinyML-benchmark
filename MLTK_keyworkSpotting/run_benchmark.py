import os
import time
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import tflite_runtime.interpreter as tflite
import psutil  # For monitoring CPU and memory usage

# Constants
MODEL_PATH = "./models/keyword_spotting.tflite"
DATASET_DIR = "./dataset"
VISUALIZATION_DIR = "./visualizations"
RESULTS_FILE = os.path.join(VISUALIZATION_DIR, "benchmark_results.csv")
MAX_INFERENCES = 100

# Create visualization directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Load dataset
def load_dataset(dataset_dir, max_files=MAX_INFERENCES):
    audio_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_paths.append(os.path.join(root, file))
                if len(audio_paths) >= max_files:
                    break  # Stop after collecting 1000 files
        if len(audio_paths) >= max_files:
            break
    if not audio_paths:
        raise FileNotFoundError(f"No audio files found in {dataset_dir}")
    return audio_paths

# Preprocess audio
def preprocess_audio(file_path, input_shape):
    """
    Preprocess the input audio to match the model's input requirements.
    """
    try:
        rate, audio = wav.read(file_path)

        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        # Generate spectrogram
        spectrogram = np.abs(np.fft.rfft(audio, n=input_shape[-1]))  # Use FFT to approximate a spectrogram

        # Resize to match model input
        spectrogram = np.resize(spectrogram, input_shape[:-1])  # Resize to [time, features]
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

        return spectrogram.astype(np.float32)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Benchmark function
def run_benchmark(interpreter, dataset_paths):
    """
    Run inference on the dataset and collect benchmark metrics.
    """
    inference_times = []
    predictions = []
    cpu_usages = []
    memory_usages = []
    confidence_scores = []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:]  # Exclude batch dimension

    for file_path in tqdm(dataset_paths, desc="Benchmarking Inference"):
        input_data = preprocess_audio(file_path, input_shape)
        if input_data is None:
            continue  # Skip files that failed preprocessing

        try:
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Monitor system metrics before inference
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.virtual_memory().percent
    

            # Measure inference time
            start_time = time.perf_counter()
            interpreter.invoke()
            end_time = time.perf_counter()
            
            # Monitor system metrics after inference
            cpu_after = psutil.cpu_percent(interval=None)
            mem_after = psutil.virtual_memory().percent

            # Fetch inference time
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            cpu_usages.append((cpu_before + cpu_after) / 2)  # Average CPU usage
            memory_usages.append(mem_after)  # Memory usage at the end of inference
            
             # Get the output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output)
            confidence_scores.append(np.max(output))  # <==== Extract confidence score

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return inference_times, predictions, confidence_scores, cpu_usages, memory_usages

# Visualization functions
def save_visualizations(inference_times, predictions, confidence_scores, cpu_usages, memory_usages):
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
    
    # Confidence Scores Over Time
    plt.figure()
    plt.plot(confidence_scores, color='purple')
    plt.title("Confidence Scores Over Time (Keyword Spotting)")
    plt.xlabel("Iteration")
    plt.ylabel("Confidence Score")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "keyword_confidence_scores_plot.png"))
    plt.close()

    # Predicted Labels Over Time
    plt.figure()
    predicted_labels = [np.argmax(pred) for pred in predictions]
    plt.plot(predicted_labels, color='cyan')
    plt.title("Predicted Labels Over Time (Keyword Spotting)")
    plt.xlabel("Iteration")
    plt.ylabel("Predicted Label")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "keyword_predicted_labels_plot.png"))
    plt.close()

# Save metrics to CSV
def save_results_to_csv(inference_times, predictions, confidence_scores, cpu_usages, memory_usages):
    """
    Save inference results to a CSV file.
    """
    with open(RESULTS_FILE, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Iteration", "Inference Time (s)", "Predicted Label", "Confidence Score", "CPU Usage (%)", "Memory Usage (%)"])  # <====
        for i in range(len(inference_times)):
            predicted_label = np.argmax(predictions[i])  # <====
            csvwriter.writerow([i + 1, inference_times[i], predicted_label, confidence_scores[i], cpu_usages[i], memory_usages[i]])  # <====
    print(f"Benchmark results saved to {RESULTS_FILE}")

# Main Execution
def main():
    """
    Main function to load model, process dataset, run benchmarks, and save results.
    """
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Load dataset
    dataset_paths = load_dataset(DATASET_DIR, MAX_INFERENCES)

    print("Running Benchmark...")
    inference_times, predictions, confidence_scores, cpu_usages, memory_usages = run_benchmark(interpreter, dataset_paths)

    print("Saving Visualizations...")
    save_visualizations(inference_times, predictions, confidence_scores, cpu_usages, memory_usages)

    print("Saving Results to CSV...")
    save_results_to_csv(inference_times, predictions, confidence_scores, cpu_usages, memory_usages)

    print("Benchmark Complete!")

if __name__ == "__main__":
    main()
