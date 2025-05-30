import tkinter as tk
import numpy as np
import tensorflow as tf

# Load your trained model
interpreter = tf.lite.Interpreter(model_path='model3.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# List of your final feature columns in exact order used during training
# Update this list with actual column names if you have them
final_feature_columns = [
    'URL_LENGTH',
    'URL_DIGITS',
    'URL_LETTERS',
    'URL_SPECIALS',
    'FEATURE_5',
    'FEATURE_6',
    'FEATURE_7',
    'FEATURE_8',
    'FEATURE_9',
    'FEATURE_10'
]

def extract_features(url):
    url = url.strip()
    url_length = len(url)
    url_digits = sum(c.isdigit() for c in url)
    url_letters = sum(c.isalpha() for c in url)
    url_specials = sum(not c.isalnum() for c in url)

    # For missing 6 features, fill zeros for now
    extra_features = [0] * (len(final_feature_columns) - 4)

    # Combine all features in the correct order
    features = [url_length, url_digits, url_letters, url_specials] + extra_features

    # Convert to numpy array with shape (1, 10)
    features_array = np.array([features], dtype=np.float32)
    return features_array

def predict_url():
    url = url_entry.get()
    if not url:
        result_label.config(text="Please enter a URL")
        return
    
    try:
        features = extract_features(url)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], features)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        prediction_prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        prediction = int(prediction_prob > 0.5)
        result_label.config(text=f"Prediction: {'Phishing' if prediction == 1 else 'Legitimate'} ({prediction_prob:.3f})")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def reset_fields():
    url_entry.delete(0, tk.END)
    result_label.config(text="")

# Tkinter GUI setup
root = tk.Tk()
root.title("Phishing URL Detector")
root.geometry("450x200")

tk.Label(root, text="Enter URL:", font=("Arial", 12)).pack(pady=10)
url_entry = tk.Entry(root, width=60)
url_entry.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=15)

predict_button = tk.Button(button_frame, text="Predict", command=predict_url, width=15)
predict_button.grid(row=0, column=0, padx=10)

reset_button = tk.Button(button_frame, text="Reset", command=reset_fields, width=15)
reset_button.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="blue")
result_label.pack(pady=10)

root.mainloop()
