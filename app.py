import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write("Enter student details to estimate the final exam score and Pass/Fail status.")

DATA_PATHS = [Path("archive (1).zip"), Path("archive.zip"), Path("students.csv"), Path("student_data.csv")]
FEATURES = [
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "social_media_hours",
    "family_support",
]
TARGET = "final_exam_score"


@st.cache_data
def load_data():
    data_path = next((p for p in DATA_PATHS if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Dataset file not found. Upload the dataset used in Colab to the same GitHub repository. "
            "Expected archive (1).zip, archive.zip, students.csv, or student_data.csv."
        )

    if data_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(data_path) as zf:
            csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
            if not csv_files:
                raise ValueError("The ZIP file does not contain a CSV file.")
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(data_path)

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing) +
            ". The dataset must use the same column names as the Colab notebook."
        )

    df = df.drop_duplicates().copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df


try:
    df = load_data()
    if len(df) < 5:
        st.error("The dataset needs at least 5 usable rows to train a model.")
        st.stop()

    X = df[FEATURES]
    y_score = df[TARGET]
    y_passfail = (df[TARGET] > 40).astype(int)

    score_model = LinearRegression()
    score_model.fit(X, y_score)

    passfail_model = LogisticRegression(max_iter=1000, random_state=42)
    passfail_model.fit(X, y_passfail)

    st.caption(f"Model trained from {len(df):,} dataset rows.")

    st.subheader("Student details")
    with st.form("prediction_form"):
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0)
        assignment = st.slider("Assignment completion (%)", 0.0, 100.0, 75.0, 1.0)
        study_hours = st.slider("Study hours per day", 0.0, 10.0, 3.0, 0.5)
        social_hours = st.slider("Social media hours per day", 0.0, 10.0, 2.0, 0.5)
        family_support = st.slider("Family support (0–10)", 0.0, 10.0, 5.0, 0.5)
        submitted = st.form_submit_button("Predict performance", type="primary")

    if submitted:
        student = pd.DataFrame([[
            attendance, assignment, study_hours, social_hours, family_support
        ]], columns=FEATURES)

        score = float(score_model.predict(student)[0])
        passfail = "Pass" if int(passfail_model.predict(student)[0]) == 1 else "Fail"

        c1, c2 = st.columns(2)
        c1.metric("Predicted final exam score", f"{score:.1f}")
        c2.metric("Predicted status", passfail)

        st.info("These are estimates from a model trained on the uploaded dataset, not a guarantee of actual exam results.")

except Exception as e:
    st.error("The app could not load/train the model.")
    st.write(str(e))
    st.markdown(
        "**Fix:** Upload the correct dataset file to the same GitHub repository and confirm that its columns "
        "match the names used in your Colab notebook."
    )
