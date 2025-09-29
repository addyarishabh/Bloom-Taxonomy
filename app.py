from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.models import load_model
import joblib

# Download NLTK data
nltk.download('punkt')

app = Flask(__name__)

# Load model and resources
model = load_model(r'D:\BERT_V1\elmo_verb_bloom_model.h5')
elmo = hub.load("https://tfhub.dev/google/elmo/3")
scaler = joblib.load(r'D:\BERT_V1\scaler.pkl')  # ✅ Updated line

# Bloom verb dictionary


bloom_verbs = {
    1: ["Arrange", "Define", "Describe", "Duplicate", "Identify", "Label", "List", "Match", "Memorize", "Name", "Order", "Outline", "Recognize", "Relate", "Recall", "Repeat", "Reproduce", "Select", "State"],
    2: ["Classify", "Convert", "Defend", "Describe", "Discuss", "Distinguish", "Estimate", "Explain", "Express", "Extend", "Generalized", "Give example(s)", "Identify", "Indicate", "Infer", "Locate", "Paraphrase", "Predict", "Recognize", "Rewrite", "Review", "Select", "Summarize", "Translate"],
    3: ["Apply", "Change", "Choose", "Compute", "Demonstrate", "Discover", "Dramatize", "Employ", "Illustrate", "Interpret", "Manipulate", "Modify", "Operate", "Practice", "Predict", "Prepare", "Produce", "Relate", "Schedule", "Sketch", "Solve", "Use", "Write"],
    4: ["Analyze", "Appraise", "Breakdown", "Calculate", "Categorize", "Compare", "Contrast", "Criticize", "Diagram", "Differentiate", "Discriminate", "Distinguish", "Examine", "Experiment", "Identify", "Illustrate", "Infer", "Model", "Outline", "Point out", "Question", "Relate", "Select", "Separate", "Subdivide", "Test"],
    5: ["Appraise", "Argue", "Assess", "Attach", "Choose", "Compare", "Conclude", "Contrast", "Defend", "Describe", "Discriminate", "Estimate", "Evaluate", "Explain", "Judge", "Justify", "Interpret", "Relate", "Predict", "Rate", "Select", "Summarize", "Support", "Value"],
    6: ["Arrange", "Assemble", "Categorize", "Collect", "Combine", "Comply", "Compose", "Construct", "Create", "Design", "Develop", "Devise", "Explain", "Formulate", "Generate", "Plan", "Prepare", "Rearrange", "Reconstruct", "Relate", "Reorganize", "Revise", "Rewrite", "Set up", "Summarize", "Synthesize", "Tell", "Write"]
}

def extract_verb_features(question):
    tokens = word_tokenize(question.lower())
    features = [0]*6
    for i in range(6):
        verbs = set(v.lower() for v in bloom_verbs[i+1])
        if any(t in verbs for t in tokens):
            features[i] = 1
    return features

def elmo_embed(texts):
    embeddings = elmo.signatures["default"](tf.constant(texts))["default"]
    return embeddings.numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        
        # Extract and scale features
        verb_feat = extract_verb_features(question)
        verb_feat_scaled = scaler.transform([verb_feat])  # ✅ Updated line
        
        elmo_vec = elmo_embed([question])
        elmo_vec_reshaped = np.expand_dims(elmo_vec, axis=1)
        
        # Predict
        pred = model.predict([elmo_vec_reshaped, verb_feat_scaled])
        predicted_class = np.argmax(pred) + 1
        confidence = float(np.max(pred)) * 100
        
        level_descriptions = {
            1: "Knowledge - Remembering facts",
            2: "Comprehension - Understanding meaning",
            3: "Application - Using concepts in new situations",
            4: "Analysis - Breaking down information",
            5: "Synthesis - Combining ideas to form new concepts",
            6: "Evaluation - Making judgments based on criteria"
        }
        
        return render_template('index.html', 
                               question=question,
                               level=predicted_class,
                               description=level_descriptions[predicted_class],
                               confidence=f"{confidence:.2f}%")
    
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    question = data['question']
    
    verb_feat = extract_verb_features(question)
    verb_feat_scaled = scaler.transform([verb_feat])  # ✅ Updated line
    elmo_vec = elmo_embed([question])
    elmo_vec_reshaped = np.expand_dims(elmo_vec, axis=1)
    
    pred = model.predict([elmo_vec_reshaped, verb_feat_scaled])
    predicted_class = int(np.argmax(pred) + 1)
    confidence = float(np.max(pred))
    
    return jsonify({
        'question': question,
        'predicted_level': predicted_class,
        'confidence': confidence,
        'level_description': {
            1: "Remember – Retrieving relevant knowledge from memory",
            2: "Understand – Constructing meaning from messages, including interpreting, exemplifying, classifying, summarizing, inferring, comparing, and explaining",
            3: "Apply – Carrying out or using a procedure in a given situation",
            4: "Analyze – Breaking material into constituent parts and detecting relationships and organizational structure",
            5: "Evaluate – Making judgments based on criteria and standards",
            6: "Create – Putting elements together to form a novel, coherent whole or make an original product"
        }[predicted_class]
    })

if __name__ == '__main__':
    app.run(debug=True)
