# utils.py
import PyPDF2
import os
import pandas as pd

# --------------------- Load skills ---------------------
def load_skills(skills_file="skill_dataset.csv"):
    return pd.read_csv(skills_file)["Skills"].tolist()

# --------------------- Load job descriptions ---------------------
def load_job_descriptions(folder="job_descriptions"):
    job_dict = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            role = file.replace(".txt", "").replace("_", " ").title()
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
def compute_gap(user_vector, role_vector, skills_columns):
    # Only consider skills required by the role
    missing_skills = [skills_columns[i] for i in range(len(role_vector)) if role_vector[i] == 1 and user_vector[i] == 0]

    # Fit score: how many required skills user already has
    total_required = sum(role_vector)  # total required skills for the role
    if total_required == 0:
        fit_score = 0
    else:
        fit_score = sum([user_vector[i] for i in range(len(role_vector)) if role_vector[i] == 1]) / total_required * 100

    return missing_skills, fit_score

