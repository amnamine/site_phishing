import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tldextract

# Load all three models
interpreter1 = tf.lite.Interpreter(model_path='model1.tflite')
interpreter1.allocate_tensors()

interpreter2 = tf.lite.Interpreter(model_path='model2.tflite')
interpreter2.allocate_tensors()

interpreter3 = tf.lite.Interpreter(model_path='model3.tflite')
interpreter3.allocate_tensors()

# Get input and output details for all models
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

input_details3 = interpreter3.get_input_details()
output_details3 = interpreter3.get_output_details()

# Load and prepare scaler for model1
data = pd.read_csv('dataset1.csv')
data['status'] = data['status'].map({'legitimate': 0, 'phishing': 1})
data_features = data.drop(['url', 'status'], axis=1)
scaler1 = StandardScaler()
scaler1.fit(data_features)

# Feature extraction functions
def extract_features_model1(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    
    features = {
        'length_url': len(url),
        'length_hostname': len(parsed.hostname) if parsed.hostname else 0,
        'ip': 1 if re.fullmatch(r'(\d{1,3}\.){3}\d{1,3}', parsed.hostname or '') else 0,
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_or': url.lower().count('or'),
        'nb_eq': url.count('='),
        'nb_underscore': url.count('_'),
        'nb_tilde': url.count('~'),
        'nb_percent': url.count('%'),
        'nb_slash': url.count('/'),
        'nb_star': url.count('*'),
        'nb_colon': url.count(':'),
        'nb_comma': url.count(','),
        'nb_semicolumn': url.count(';'),
        'nb_dollar': url.count('$'),
        'nb_space': url.count(' '),
        'nb_www': url.lower().count('www'),
        'nb_com': url.lower().count('.com'),
        'nb_dslash': url.count('//'),
        'http_in_path': 1 if 'http' in parsed.path.lower() else 0,
        'https_token': 1 if 'https' in ext.subdomain.lower() else 0,
        'ratio_digits_url': sum(c.isdigit() for c in url) / len(url),
        'ratio_digits_host': sum(c.isdigit() for c in (parsed.hostname or '')) / len(parsed.hostname or '1'),
        'punycode': 1 if 'xn--' in url else 0,
        'port': 1 if parsed.port else 0,
        'tld_in_path': 1 if ext.suffix in parsed.path else 0,
        'tld_in_subdomain': 1 if ext.suffix in ext.subdomain else 0,
        'abnormal_subdomain': 1 if len(ext.subdomain.split('.')) > 3 else 0,
        'nb_subdomains': len(ext.subdomain.split('.')),
        'prefix_suffix': 1 if '-' in ext.domain else 0,
        'random_domain': 0,
        'shortening_service': 1 if any(x in url for x in ['bit.ly', 'tinyurl', 'goo.gl']) else 0,
        'path_extension': 1 if re.search(r'\.\w{1,5}$', parsed.path) else 0,
        'nb_redirection': url.count('//') - 1,
        'nb_external_redirection': 0,
        'length_words_raw': len(re.findall(r'\w+', url)),
        'char_repeat': max([url.count(c) for c in set(url)]),
        'shortest_words_raw': min([len(w) for w in re.findall(r'\w+', url)]) if re.findall(r'\w+', url) else 0,
        'shortest_word_host': min([len(w) for w in (parsed.hostname or '').split('.')]) if parsed.hostname else 0,
        'shortest_word_path': min([len(w) for w in parsed.path.split('/')]) if parsed.path else 0,
        'longest_words_raw': max([len(w) for w in re.findall(r'\w+', url)]) if re.findall(r'\w+', url) else 0,
        'longest_word_host': max([len(w) for w in (parsed.hostname or '').split('.')]) if parsed.hostname else 0,
        'longest_word_path': max([len(w) for w in parsed.path.split('/')]) if parsed.path else 0,
        'avg_words_raw': np.mean([len(w) for w in re.findall(r'\w+', url)]) if re.findall(r'\w+', url) else 0,
        'avg_word_host': np.mean([len(w) for w in (parsed.hostname or '').split('.')]) if parsed.hostname else 0,
        'avg_word_path': np.mean([len(w) for w in parsed.path.split('/')]) if parsed.path else 0
    }

    aligned_features = [features[col] if col in features else 0 for col in data_features.columns]
    scaled = scaler1.transform([aligned_features])
    return scaled

def extract_features_model2(url):
    ext = tldextract.extract(url)
    domain = ext.domain + "." + ext.suffix if ext.suffix else ext.domain
    subdomain = ext.subdomain

    url_len = len(url)
    domain_len = len(domain)
    tld_len = len(ext.suffix)
    no_of_subdomains = subdomain.count('.') + 1 if subdomain else 0
    no_of_letters = len(re.findall(r'[a-zA-Z]', url))
    no_of_digits = len(re.findall(r'\d', url))
    no_of_special = len(re.findall(r'[^a-zA-Z0-9]', url))

    features = {
        "URLLength": url_len,
        "DomainLength": domain_len,
        "IsDomainIP": int(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) is not None),
        "URLSimilarityIndex": 100,
        "CharContinuationRate": 1,
        "TLDLegitimateProb": 0.5,
        "URLCharProb": 0.05,
        "TLDLength": tld_len,
        "NoOfSubDomain": no_of_subdomains,
        "HasObfuscation": 0,
        "NoOfObfuscatedChar": 0,
        "ObfuscationRatio": 0,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": no_of_letters / url_len if url_len else 0,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": no_of_digits / url_len if url_len else 0,
        "NoOfEqualsInURL": url.count('='),
        "NoOfQMarkInURL": url.count('?'),
        "NoOfAmpersandInURL": url.count('&'),
        "NoOfOtherSpecialCharsInURL": len(re.findall(r'[^\w\d\-._~:/?#\[\]@!$&\'()*+,;=]', url)),
        "SpacialCharRatioInURL": no_of_special / url_len if url_len else 0,
        "IsHTTPS": int(url.startswith("https")),
        "LineOfCode": 1000,
        "LargestLineLength": 1000,
        "HasTitle": 1,
        "DomainTitleMatchScore": 0.5,
        "URLTitleMatchScore": 0.5,
        "HasFavicon": 1,
        "Robots": 1,
        "IsResponsive": 1,
        "NoOfURLRedirect": 0,
        "NoOfSelfRedirect": 0,
        "HasDescription": 1,
        "NoOfPopup": 0,
        "NoOfiFrame": 0,
        "HasExternalFormSubmit": 0,
        "HasSocialNet": 0,
        "HasSubmitButton": 1,
        "HasHiddenFields": 0,
        "HasPasswordField": 1,
        "Bank": 0,
        "Pay": 0,
        "Crypto": 0,
        "HasCopyrightInfo": 1,
        "NoOfImage": 10,
        "NoOfCSS": 5,
        "NoOfJS": 10,
        "NoOfSelfRef": 20,
        "NoOfEmptyRef": 0,
        "NoOfExternalRef": 10
    }

    df = pd.DataFrame([features])
    scaler = StandardScaler()
    dummy_input = pd.DataFrame(np.zeros((1, df.shape[1])), columns=df.columns)
    scaler.fit(dummy_input)
    scaled_input = scaler.transform(df)
    return scaled_input.astype(np.float32)

def extract_features_model3(url):
    url = url.strip()
    url_length = len(url)
    url_digits = sum(c.isdigit() for c in url)
    url_letters = sum(c.isalpha() for c in url)
    url_specials = sum(not c.isalnum() for c in url)

    extra_features = [0] * 6  # 6 additional features
    features = [url_length, url_digits, url_letters, url_specials] + extra_features
    return np.array([features], dtype=np.float32)

def predict_url():
    url = url_entry.get().strip()
    if not url:
        messagebox.showwarning("Input Error", "Please enter a URL.")
        return

    try:
        selected_model = model_var.get()
        
        # Single model prediction
        if selected_model == "model1":
            features = extract_features_model1(url).astype(np.float32)
            interpreter = interpreter1
            input_details = input_details1
            output_details = output_details1
        elif selected_model == "model2":
            features = extract_features_model2(url).astype(np.float32)
            interpreter = interpreter2
            input_details = input_details2
            output_details = output_details2
        else:  # model3
            features = extract_features_model3(url).astype(np.float32)
            interpreter = interpreter3
            input_details = input_details3
            output_details = output_details3

        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        result_text = f"Result: {'Phishing ⚠️' if prediction > 0.5 else 'Legitimate ✅'}\n"
        result_text += f"Confidence: {prediction:.2%}"
        result_color = 'red' if prediction > 0.5 else 'green'
        
        result_label.config(text=result_text, fg=result_color)
        status_bar.config(text="Analysis complete")
        
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")
        status_bar.config(text="Error occurred")

def reset_interface():
    url_entry.delete(0, tk.END)
    result_label.config(text="")
    status_bar.config(text="Ready")

# GUI Setup
root = tk.Tk()
root.title("Phishing URL Detector")
root.geometry("700x500")
root.configure(bg="#f0f0f0")

# Style configuration
style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 11))
style.configure("Title.TLabel", font=("Helvetica", 24, "bold"), padding=20)
style.configure("TRadiobutton", background="#f0f0f0", font=("Helvetica", 11))

# Main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Title
title_label = ttk.Label(main_frame, text="Phishing URL Detector", style="Title.TLabel")
title_label.pack(pady=(0, 20))

# Model Selection Frame
model_frame = ttk.LabelFrame(main_frame, text="Select Model", padding="10")
model_frame.pack(fill=tk.X, pady=(0, 20))

model_var = tk.StringVar(value="model1")
models = [
    ("Model 1", "model1"),
    ("Model 2", "model2"),
    ("Model 3", "model3")
]

for i, (text, value) in enumerate(models):
    ttk.Radiobutton(
        model_frame, 
        text=text,
        value=value,
        variable=model_var
    ).pack(anchor="w", pady=2)

# URL Input Frame
input_frame = ttk.LabelFrame(main_frame, text="URL Analysis", padding="10")
input_frame.pack(fill=tk.X, pady=(0, 20))

url_label = ttk.Label(input_frame, text="Enter URL to analyze:")
url_label.pack(anchor="w", pady=(0, 5))

url_entry = ttk.Entry(input_frame, width=60, font=("Helvetica", 11))
url_entry.pack(fill=tk.X, pady=5)

# Buttons Frame
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=20)

check_button = ttk.Button(
    button_frame, 
    text="Analyze URL",
    command=predict_url,
    style="Accent.TButton"
)
check_button.pack(side=tk.LEFT, padx=5)

reset_button = ttk.Button(
    button_frame,
    text="Reset",
    command=reset_interface
)
reset_button.pack(side=tk.LEFT, padx=5)

# Result Frame
result_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

result_label = tk.Label(
    result_frame,
    text="",
    font=("Helvetica", 12),
    wraplength=600,
    justify="center"
)
result_label.pack(pady=10)

# Status bar
status_bar = ttk.Label(
    root,
    text="Ready",
    relief=tk.SUNKEN,
    anchor=tk.W,
    padding=(5, 2)
)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Bind Enter key to predict_url
url_entry.bind('<Return>', lambda event: predict_url())

# Start the application
root.mainloop()
