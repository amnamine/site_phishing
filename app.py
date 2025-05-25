from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tldextract

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('dnn2.h5')

# Dummy dataset to re-fit the scaler structure
data = pd.read_csv('dataset_phishing.csv')
data['status'] = data['status'].map({'legitimate': 0, 'phishing': 1})
data_features = data.drop(['url', 'status'], axis=1)
scaler = StandardScaler()
scaler.fit(data_features)

def extract_features(url):
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
    scaled = scaler.transform([aligned_features])
    return scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.json['url']
        if not url:
            return jsonify({'error': 'Please enter a URL'}), 400
        
        features = extract_features(url)
        prediction = model.predict(features)[0][0]
        
        result = {
            'is_phishing': bool(prediction > 0.5),
            'confidence': float(prediction),
            'message': 'Phishing ⚠️' if prediction > 0.5 else 'Legitimate ✅'
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 