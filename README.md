📌 GapCapture – Smart Job Skill Gap Analyzer

GapCapture is an AI-powered skill gap analysis system that compares a user's skills 
with job requirements and identifies missing skills. It helps users understand what 
they need to learn to match specific job roles, and also suggests alternative roles 
based on the user's skill profile.

🚀 Features
✅ Skill extraction from resume (PDF) or manual input, with synonym normalization
✅ Job skill requirement matching (rule-based)
✅ NLP-based semantic similarity matching (TF-IDF + cosine similarity)
✅ Hybrid scoring: weighted combination of rule-based and semantic scores
✅ Missing skills identification (Skill Gap Detection)
✅ Suggests relevant job roles based on skill similarity across all roles
✅ Dataset-driven skill mapping, built from real job postings

🧠 How it Works
1. User uploads a resume (PDF) or manually enters skills
2. System extracts skills via keyword matching + a synonym dictionary
3. Skills are compared against the selected role's requirements two ways:
   - Rule-based score: % of required skills present
   - Semantic score: TF-IDF/cosine similarity between skills and the job description
4. A hybrid score (60% rule-based + 40% semantic) is computed
5. Output includes matching skills, missing skills, hybrid fit score, 
   and suggested alternative roles

🛠️ Tech Stack
Python
Streamlit (UI)
scikit-learn (TF-IDF, cosine similarity)
Pandas / NumPy
PyPDF2 (resume parsing)
CSV datasets (custom-built from real job postings)

📂 Project Structure
GapCapture/
│
├── app.py               # Streamlit main application
├── utils.py              # Skill extraction, rule-based scoring, TF-IDF/NLP scoring
├── model_code.py          # Earlier prototype: ML classifier (Logistic Regression) 
│                           to predict role from skills — superseded by the 
│                           hybrid rule-based + NLP approach in utils.py
├── model.pkl              # Trained model from the earlier ML prototype
├── skills_columns.pkl     # Skill columns used by the earlier ML prototype
├── prepare_db.py          # Dataset preparation script
├── requirements.txt       # Required libraries
├── skill_dataset.csv      # Role-to-skill mapping dataset
├── training_dataset.csv   # Dataset used to train the earlier ML prototype
└── job_descriptions/       # Job description text files used for semantic matching

📝 Note
This project originally used a supervised ML classifier (see model_code.py) to 
predict job role from a skill vector. After learning more NLP techniques, I 
rewrote the matching logic using TF-IDF and cosine similarity combined with 
rule-based scoring, since skill-to-role fit is better modeled as a similarity/
scoring problem than a fixed-label classification problem. The earlier files 
are kept in the repo to show that iteration.
