# app.py
import re
import streamlit as st
import pandas as pd
import pickle
from utils import (
    load_skills,
    load_job_descriptions,
    extract_text_from_pdf,
    extract_skills_from_text,
    skills_to_vector,
    compute_gap,              
    compute_final_score
)
SKILL_SYNONYMS = {
    "rag": "rag",
    "retrieval augmented generation": "rag",

    "chromadb": "chromadb",
    "chroma db": "chromadb",

    "pinecone": "pinecone",

    "langchain": "langchain",
    "lang chain": "langchain",

    "crewai": "crewai",
    "crew ai": "crewai",

    "llm": "llm",
    "large language model": "llm",
    "large language models": "llm",

    "api": "api",
    "apis": "api",

    "web crawling": "web crawling",
    "web scraping": "web crawling",

    "prompt engineering": "prompt engineering",
    "vector database": "vector database",
    "vector databases": "vector database",
}
def extract_synonym_skills(text: str):
    text = text.lower()
    found = set()

    for key, normalized_skill in SKILL_SYNONYMS.items():
        if key in text:
            found.add(normalized_skill)

    return list(found)

def role_vector_from_skillset(preferred_role, skills_columns):
    required = set(get_job_skills(preferred_role))
    return [1 if s in required else 0 for s in skills_columns]


# --- Page config ---
st.set_page_config(page_title="GapCapture - Skill Gap Analyzer", layout="wide", page_icon="🧠")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E6E6FA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load training dataset ---
df_roles = pd.read_csv("training_dataset.csv")
df_roles["Job Role"] = df_roles["Job Role"].astype(str).apply(
    lambda x: re.sub(r"\s*\d+$", "", x).strip()
)
df_roles.iloc[:, 1:] = df_roles.iloc[:, 1:].fillna(0).astype(int)

# --- Load role-specific skill list (unique per role) ---
df_skillset = pd.read_csv("skill_dataset.csv", encoding="utf-8-sig")
df_skillset.columns = df_skillset.columns.str.strip().str.lower()

# --- Build GLOBAL skills list (training_dataset + skill_dataset) ---
dataset_skills = set([c.strip().lower() for c in df_roles.columns[1:]])

csv_skills = set()
for s in df_skillset["skills"].dropna().astype(str):
    for skill in s.split(","):
        csv_skills.add(skill.strip().lower())

skills_columns = sorted(list(dataset_skills.union(csv_skills)))

# --- Function: get required skills for one job role ---
def get_job_skills(job_title: str):
    row = df_skillset[df_skillset["job_title"].astype(str).str.strip().str.lower()
                      == job_title.strip().lower()]
    if row.empty:
        return []
    return [s.strip().lower() for s in str(row.iloc[0]["skills"]).split(",")]



# --- Sidebar input ---
st.title("🧠 GapCapture - Smart Skill Gap Analyzer")

input_option = st.radio("Choose input type:", ["Upload Resume (PDF)", "Manual Skill Entry"])

if input_option == "Upload Resume (PDF)":
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    user_skills = []
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        user_skills = extract_skills_from_text(text, skills_columns)
elif input_option == "Manual Skill Entry":
    manual_text = st.text_area(
        "Enter your skills / experience (comma / space / newline separated):",
        height=180,
        placeholder="Example: Python, SQL, RAG, Pinecone, LangChain, Git"
    )

    # Extract skill-like words from dataset columns
    matched_skills = extract_skills_from_text(manual_text, skills_columns)

    # Extract synonyms (RAG, Pinecone, etc.)
    synonym_skills = extract_synonym_skills(manual_text)

    # Combine skills
    user_skills = list(set([s.lower() for s in matched_skills + synonym_skills]))



# --- Preferred role selection ---
preferred_role = st.selectbox("Select your preferred role:", df_roles["Job Role"].tolist())

# --- Analyze Button ---
if st.button("Analyze"):
    if not user_skills:
        st.warning("Please provide your skills via resume upload or manual entry.")
    else:
        # Convert skills to vector
        user_vector = skills_to_vector(user_skills, skills_columns)
        
        # Get role vector safely
        role_row = df_roles[df_roles["Job Role"].str.strip().str.lower() == preferred_role.strip().lower()]
        if role_row.empty:
            st.error("Selected role not found in dataset!")
            st.stop()
        role_vector = role_vector_from_skillset(preferred_role, skills_columns)

        # Compute missing skills and fit
        # Load job descriptions
        job_dict = load_job_descriptions()

        job_description = job_dict.get(preferred_role.strip().lower())
        if not job_description:
           st.error(f"Job description file not found for selected role: {preferred_role}")
           st.write("Available roles in JD folder:", list(job_dict.keys()))
           st.stop()


        # Compute final hybrid score (Rule-based + NLP)
        result = compute_final_score(
               user_skills,
               role_vector,
               job_description,
               skills_columns
            )

        # Use unique job-wise skills instead of global dataset skills
        role_required_skills = get_job_skills(preferred_role)

        if role_required_skills:
           missing_skills = [s for s in role_required_skills if s not in [u.lower() for u in user_skills]]
        else:
           # fallback: use original extraction if job not found in skill_dataset.csv
          missing_skills = result["missing_skills"]

        fit_score = result["final_match_score"]
        rule_score = result["rule_based_score"]
        

        # Display Dashboard
        st.subheader(f"📊 Dashboard - {preferred_role}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Overall Match Score (Hybrid AI):**")
            progress_value = int(max(0, min(100, rule_score)))
            st.progress(progress_value)
            st.markdown(f"**{rule_score:.2f}% overall match**")

            st.markdown(f"- 📘 Rule-based skill match: **{rule_score:.2f}%**")

            st.markdown("**Missing Skills:**")
            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("🎉 You have all required skills for this role!")

        with col2:
            st.markdown("**Role Fit Visualization:**")
            import matplotlib.pyplot as plt
            labels = ["Missing Skills","Matched Skills"]
            sizes = [100 - rule_score, rule_score]
            colors = ["#9370DB", "#D8BFD8"]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
            ax.axis("equal")
            st.pyplot(fig)

                # Suggest alternative roles
        st.subheader("💡 Other roles you might fit:")

        suggestions = []

        for role in df_skillset["job_title"].unique():
            if role.strip().lower() != preferred_role.strip().lower():
                role_vec = role_vector_from_skillset(role, skills_columns)
                _, score = compute_gap(user_vector, role_vec, skills_columns)

                if score >= 30:
                    suggestions.append((role, score))

        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)

        if suggestions:
            for role, score in suggestions[:5]:
                st.markdown(f"- {role} ({score:.0f}% fit)")
        else:
            st.write("No other role suggestions found.")
