from flask import Flask, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
import nltk
nltk.download('stopwords')

app = Flask(__name__)

# Load vectorizer and model
with open('vectorizer2.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('classifier2.pkl', 'rb') as f:
    model = pickle.load(f)

# Text preprocessing
port_stem = PorterStemmer()
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# API Route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    processed_text = preprocess(data['text'])
    vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized)

    result = 'Real' if prediction[0] == 1 else 'Fake'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
