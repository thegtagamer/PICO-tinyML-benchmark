import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="./models/quantized_deeplabv3_mobilenet_v2.tflite")
interpreter.allocate_tensors()
print(interpreter.get_tensor_details())

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: The file {image_path} could not be read as an image.")
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    image = (image * 127.5 + 127.5).astype(np.uint8)  # Quantize to UINT8
    image = np.expand_dims(image, axis=0)
    return image

# Perform inference
def classify_image(image_path):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get classification results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), np.max(output_data)

# Test the pipeline
image_path = "test_image.jpg"  # Replace with your image path
label, confidence = classify_image(image_path)
print(f"Predicted label: {label}, Confidence: {confidence}")

