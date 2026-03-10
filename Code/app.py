import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("fake_job_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

# Prediction function
def predict_job(text):

    text = clean_text(text)

    vector = tfidf.transform([text])

    prediction = model.predict(vector)

    return prediction

# Streamlit UI
st.title("Fake Job Posting Detection System")

st.write("Enter job description to check whether it is Fake or Real.")

job_text = st.text_area("Job Description")

if st.button("Predict"):

    result = predict_job(job_text)

    if result[0] == 1:
        st.error("⚠️ This Job Posting is FAKE")
    else:
        st.success("✅ This Job Posting is REAL")