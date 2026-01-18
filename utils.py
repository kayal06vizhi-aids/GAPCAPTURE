# utils.py
import PyPDF2
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------- Load skills ---------------------
def load_skills(skills_file="skill_dataset.csv"):
    return pd.read_csv(skills_file)["Skills"].tolist()

# --------------------- Load job descriptions ---------------------
def load_job_descriptions(folder="job_descriptions"):
    job_dict = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            role = os.path.splitext(file)[0].strip().lower()
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                job_dict[role] = f.read()
    return job_dict

# --------------------- Extract text from PDF ---------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --------------------- Extract skills from text ---------------------
def extract_skills_from_text(text, skills_list):
    return [skill for skill in skills_list if skill.lower() in text.lower()]

# --------------------- Convert skills to binary vector ---------------------
def skills_to_vector(user_skills, skills_columns):
    return [1 if skill in user_skills else 0 for skill in skills_columns]

# --------------------- Compute missing skills and fit score ---------------------
import math

def compute_gap(user_vector, role_vector, skills_columns):
    # ✅ Clean NaN/None in role_vector
    cleaned_role_vector = []
    for v in role_vector:
        if v is None:
            cleaned_role_vector.append(0)
        elif isinstance(v, float) and math.isnan(v):
            cleaned_role_vector.append(0)
        else:
            cleaned_role_vector.append(int(v))

    # Only consider skills required by the role
    missing_skills = [
        skills_columns[i]
        for i in range(len(cleaned_role_vector))
        if cleaned_role_vector[i] == 1 and user_vector[i] == 0
    ]

    total_required = sum(cleaned_role_vector)

    if total_required == 0:
        fit_score = 0
    else:
        matched_required = sum(
            user_vector[i]
            for i in range(len(cleaned_role_vector))
            if cleaned_role_vector[i] == 1
        )
        fit_score = (matched_required / total_required) * 100

    return missing_skills, fit_score


# --------------------- Compute semantic similarity (NLP) ---------------------
def compute_semantic_similarity(user_skills, job_description):
    """
    NLP-based semantic similarity between user skills and job description
    """
    user_text = " ".join(user_skills).lower()
    job_text = job_description.lower()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([user_text, job_text])

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)

# --------------------- Final hybrid score ---------------------
def compute_final_score(user_skills, role_vector, job_description, skills_columns):
    user_vector = skills_to_vector(user_skills, skills_columns)

    missing_skills, fit_score = compute_gap(
        user_vector, role_vector, skills_columns
    )

    semantic_score = compute_semantic_similarity(
        user_skills, job_description
    )

    final_score = round((0.6 * fit_score) + (0.4 * semantic_score), 2)

    return {
        "missing_skills": missing_skills,
        "rule_based_score": round(fit_score, 2),
        "semantic_score": semantic_score,
        "final_match_score": final_score
    }


