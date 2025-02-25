from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import spacy
import pdfplumber
from docx import Document
import pandas as pd
import ast
from scipy.stats import percentileofscore
import os
import re

app = Flask(_name_)
CORS(app)  # Enable CORS for all routes

nlp = spacy.load("en_core_web_sm")

# Load the analyzed CSV dataset if available
csv_path = r"C:\aianalyzer.py\ai-resume-analyzer\AnalyzedResumes.csv"
if os.path.exists(csv_path):
    df_dataset = pd.read_csv(csv_path)
    df_dataset['Analysis'] = df_dataset['Analysis'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    dataset_scores = df_dataset['Analysis'].apply(lambda a: a.get('resume_score', None)).dropna().tolist()
    average_dataset_score = sum(dataset_scores) / len(dataset_scores) if dataset_scores else None
else:
    dataset_scores = []
    average_dataset_score = None

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file):
    document = Document(file)
    text = "\n".join([para.text for para in document.paragraphs])
    return text

def extract_skills(text):
    tech_keywords = {
        "python", "java", "c++", "c", "c#", "javascript", "typescript", "html", "html5", "css", "css3", 
        "tailwind css", "bootstrap", "react", "react.js", "angular", "vue.js", "node.js", "express.js", 
        "next.js", "nuxt.js", "svelte", "django", "flask", "spring", "fastapi", "sql", "nosql", "mongodb", 
        "postgresql", "mysql", "sqlite", "redis", "aws", "azure", "gcp", "firebase", "docker", "kubernetes", 
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "r", 
        "scala", "git", "github", "gitlab", "bitbucket", "power bi", "tableau", "hadoop", "spark", 
        "airflow", "bash", "linux", "google cloud", "jenkins", "terraform", "ansible", "opencv", "mediapipe", 
        "llms", "chatgpt", "gpt-4", "openai api", "langchain", "prompt engineering", "homomorphic encryption", 
        "cybersecurity", "ethical hacking", "graphql", "rest api", "jwt", "oauth", "rabbitmq", "kafka",
        "mern", "mern stack", "three.js", "ar.js", "razorpay", "canva"
    }
    
    found_skills = set()
    lower_text = text.lower()
    
    for keyword in tech_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, lower_text):
            found_skills.add(keyword.capitalize())
    
    return list(found_skills)

def analyze_resume(text):
    word_count = len(text.split())
    bullet_count = text.count('\u2022') + text.count('-') + text.count('*')
    action_verbs = [
        "developed", "led", "managed", "created", "designed", 
        "implemented", "improved", "optimized", "built", 
        "initiated", "launched"
    ]
    action_count = sum(text.lower().count(verb) for verb in action_verbs)
    numbers = re.findall(r'\b\d+\b', text)
    quantification_count = len(numbers)
    skills = extract_skills(text)
    
    if word_count < 250:
        word_score = 0
    elif 250 <= word_count <= 1000:
        word_score = 20
    else:
        word_score = 15
    
    bullet_score = min(bullet_count * 1, 20)
    action_score = min(action_count * 2, 20)
    quantification_score = min(quantification_count * 1, 10)
    skills_score = min(len(skills) * 5, 20)
    
    total_score = word_score + bullet_score + action_score + quantification_score + skills_score
    total_score = min(total_score, 100)
    
    suggestions = []
    if word_count < 250:
        suggestions.append("Increase the word count to provide more context.")
    if bullet_count == 0:
        suggestions.append("Consider using bullet points to enhance readability.")
    if action_count == 0:
        suggestions.append("Incorporate action verbs to better showcase your achievements.")
    if quantification_count == 0:
        suggestions.append("Include quantifiable metrics to highlight your impact.")
    if len(skills) == 0:
        suggestions.append("Mention relevant technical skills and tools.")
    
    return {
        "word_count": word_count,
        "bullet_count": bullet_count,
        "action_count": action_count,
        "number_count": quantification_count,
        "extracted_skills": skills,
        "word_score": word_score,
        "bullet_score": bullet_score,
        "action_score": action_score,
        "quantification_score": quantification_score,
        "skills_score": skills_score,
        "resume_score": total_score,
        "suggestions": suggestions,
        "ats_friendly": "Yes" if word_count >= 250 else "No",
        "selection_likelihood": "High" if total_score > 70 else "Medium" if total_score > 50 else "Low"
    }

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    file = request.files['resume']
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        return jsonify({"error": "Unsupported file type. Please upload a PDF or DOCX file."}), 400

    if not text:
        return jsonify({"error": "Failed to extract text from the provided file."}), 500

    analysis = analyze_resume(text)

    if dataset_scores:
        score = analysis.get('resume_score', 0)
        percentile = percentileofscore(dataset_scores, score)
        analysis['score_percentile'] = percentile
        analysis['average_dataset_score'] = average_dataset_score

    return jsonify(analysis)

if _name_ == '_main_':
    app.run(debug=True)
