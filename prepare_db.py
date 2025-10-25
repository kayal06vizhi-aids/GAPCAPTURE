import os
import pandas as pd

# ---------- Step 1: Read the master skill dataset ----------
skills_df = pd.read_csv("skill_dataset.csv")
skills = [s.lower().strip() for s in skills_df["Skills"].dropna()]

# ---------- Step 2: Prepare to scan job descriptions ----------
job_folder = "job_descriptions"
job_files = [f for f in os.listdir(job_folder) if f.endswith(".txt")]

data = []  # to store combined info

# ---------- Step 3: Loop through each job description ----------
for file in job_files:
    job_role = file.replace(".txt", "")
    with open(os.path.join(job_folder, file), "r", encoding="utf-8") as f:
        text = f.read().lower()

    skill_presence = []
    for skill in skills:
        # check if skill appears in job description text
        if skill in text:
            skill_presence.append(1)
        else:
            skill_presence.append(0)

    data.append([job_role] + skill_presence)

# ---------- Step 4: Create DataFrame ----------
columns = ["Job Role"] + skills
combined_df = pd.DataFrame(data, columns=columns)

# ---------- Step 5: Save to CSV ----------
combined_df.to_csv("training_dataset.csv", index=False)

print("✅ training_dataset.csv created successfully!")