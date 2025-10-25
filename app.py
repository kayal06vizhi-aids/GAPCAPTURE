# app.py
import streamlit as st
import pandas as pd
import pickle
from utils import extract_text_from_pdf, extract_skills_from_text, skills_to_vector, compute_gap

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

# --- Load training dataset and skills ---
df_roles = pd.read_csv("training_dataset.csv")
skills_columns = df_roles.columns[1:].tolist()  # all skill columns except "Job Role"

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
    user_skills = st.multiselect("Select your skills", skills_columns)

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
        role_vector = role_row.iloc[0, 1:].tolist()

        # Compute missing skills and fit
        missing_skills, fit_score = compute_gap(user_vector, role_vector, skills_columns)

        # Display Dashboard
        st.subheader(f"📊 Dashboard - {preferred_role}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fit Score:**")
            st.progress(int(fit_score))
            st.markdown(f"**{fit_score:.2f}% fit for this role**")

            st.markdown("**Missing Skills:**")
            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("🎉 You have all required skills for this role!")

        with col2:
            st.markdown("**Role Fit Visualization:**")
            import matplotlib.pyplot as plt
            labels = ["Missing Skills","Matched Skills"]
            sizes = [100 - fit_score, fit_score]
            colors = ["#9370DB", "#D8BFD8"]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
            ax.axis("equal")
            st.pyplot(fig)

        # Suggest alternative roles
        st.subheader("💡 Other roles you might fit:")
        suggestions = []
        for _, row in df_roles.iterrows():
           if row["Job Role"] != preferred_role:
              role_vec = row[1:].tolist()
              _,score = compute_gap(user_vector, role_vec, skills_columns)
              if score >= 50:  # threshold for suggestion
                    suggestions.append(f"{row['Job Role']} ({score:.0f}% fit)")

        if suggestions:
           for role in suggestions:
              st.markdown(f"- {role}")  # bullet point
        else:
           st.write("No other role suggestions found.")


