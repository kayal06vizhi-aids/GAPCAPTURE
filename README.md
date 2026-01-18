📌 GapCapture – Smart Job Skill Gap Analyzer

GapCapture is an AI-powered skill gap analysis system that compares a user’s skills with job requirements and identifies missing skills. It helps users understand what they need to learn to match specific job roles, and also suggests alternative roles based on the user’s skill profile.

🚀 Features

✅ Skill extraction from user input
✅ Job skill requirement matching
✅ Missing skills identification (Skill Gap Detection)
✅ Suggests relevant job roles based on skills
✅ Dataset-driven skill mapping
✅ Simple interface using Flask (Python)

🧠 How it Works

User enters skill details manually (or pastes skill description)
System processes and extracts skills
Skills are compared with job description requirements
Output includes:
Matching skills
Missing skills (gap)
Suggested roles based on skill similarity

🛠️ Tech Stack

Python
Flask
Machine Learning
Pandas / NumPy
Pickle models (.pkl)
CSV datasets
HTML/CSS (UI)

📂 Project Structure
GapCapture/
│
├── app.py                 # Flask main application
├── model_code.py          # Skill gap logic / ML processing
├── model.pkl              # Trained model file
├── skills_columns.pkl     # Skill mapping columns
├── prepare_db.py          # Dataset preparation script
├── requirements.txt       # Required libraries
├── skill_dataset.csv      # Skills dataset
└── training_dataset.csv   # Training dataset used for model
