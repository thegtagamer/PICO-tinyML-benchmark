import tensorflow as tf

# Define the path to the saved model
saved_model_dir = "./models"

# Load the saved model and convert it to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable optimization for size and latency
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Specify that you want to perform INT8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# (Optional) Provide a representative dataset function for calibrating INT8 quantization
def representative_data_gen():
    # Example: Replace with your actual dataset for calibration
    for _ in range(100):
        # Replace `input_shape` with the input shape of your model
        yield [tf.random.normal([1, 224, 224, 3])]

converter.representative_dataset = representative_data_gen

# Ensure that the model inputs and outputs are quantized
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert the model
quantized_tflite_model = converter.convert()

# Save the quantized model to a file
output_tflite_file = "mobileNetV2_quantized.tflite"
with open(output_tflite_file, "wb") as f:
    f.write(quantized_tflite_model)

print(f"Quantized model saved to {output_tflite_file}")

