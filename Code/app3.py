import streamlit as st
import pickle
import re
import pandas as pd

# Load model
model = pickle.load(open("fake_job_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake Job Detector", page_icon="💼", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.big-title {
    font-size:40px;
    font-weight:bold;
}
.card {
    background-color:#1c1f26;
    padding:20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="big-title">💼 Fake Job Posting Detector</div>', unsafe_allow_html=True)
st.write("Detect fraudulent job postings using AI")

st.markdown("---")

# ---------- INPUT SECTION ----------
st.subheader("📄 Enter Job Details")

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input("Job Title")
    company = st.text_input("Company Name")
    location = st.text_input("Location")

with col2:
    salary = st.text_input("Salary")
    description = st.text_area("Job Description")

requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

# ---------- CLEAN FUNCTION ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

# ---------- BUTTON ----------
if st.button("🚀 Analyze Job"):

    job_text = job_title + " " + company + " " + location + " " + salary + " " + description + " " + requirements + " " + benefits

    if job_text.strip() == "":
        st.warning("⚠️ Please enter job details")
    else:
        cleaned = clean_text(job_text)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        prob = model.predict_proba(vector)

        fake_score = prob[0][1] * 100
        real_score = prob[0][0] * 100

        st.markdown("---")
        st.subheader("📊 Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fake Probability", f"{fake_score:.2f}%")

        with col2:
            st.metric("Real Probability", f"{real_score:.2f}%")

        if prediction[0] == 1:
            st.error("⚠️ This looks like a FAKE job")
        else:
            st.success("✅ This looks like a REAL job")

        # ---------- RISK ANALYSIS ----------
        st.subheader("⚠️ Risk Analysis")

        risks = []
        text_lower = job_text.lower()

        if "work from home" in text_lower:
            risks.append("Work from home scam pattern")

        if "no experience" in text_lower:
            risks.append("No experience required")

        if "earn" in text_lower:
            risks.append("Unrealistic earning promise")

        if risks:
            for r in risks:
                st.warning(r)
        else:
            st.success("No major risk detected")

# ---------- DASHBOARD ----------
st.markdown("---")
st.subheader("📊 System Overview")

data = pd.DataFrame({
    "Type": ["Real", "Fake"],
    "Count": [80, 20]
})

st.bar_chart(data.set_index("Type"))

# ---------- FOOTER ----------
st.markdown("---")
st.write("🚀 Developed using Machine Learning & NLP")