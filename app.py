import streamlit as st
import re
import joblib

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
model_loss = joblib.load("model_loss.pkl")
model_sev = joblib.load("model_sev.pkl")
model_ass = joblib.load("model_ass.pkl")

# -----------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_claim(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    loss_pred = model_loss.predict(vec)[0]
    sev_pred = model_sev.predict(vec)[0]
    ass_pred = model_ass.predict(vec)[0]

    return loss_pred, sev_pred, ass_pred

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Claims Description Normalizer", page_icon="🔍", layout="centered")

st.markdown("<h1 style='text-align:center;'>🔍 Claims Description Normalizer</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>NLP-based classification of Loss Type, Severity, and Affected Asset</p>", unsafe_allow_html=True)

user_input = st.text_area("✏️ Enter Claim Description:", height=120, placeholder="e.g. My car was hit from behind and bumper got damaged.")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a claim description.")
    else:
        loss, severity, asset = predict_claim(user_input)

        st.markdown("---")
        st.markdown("### ✅ Extracted Claim Information:")

        # Colored cards layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style='background:#1E90FF; padding:18px; border-radius:10px; text-align:center; color:white;'>
                <h4>Loss Type</h4>
                <h2>{}</h2>
            </div>
            """.format(loss), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background:#FFA500; padding:18px; border-radius:10px; text-align:center; color:white;'>
                <h4>Severity</h4>
                <h2>{}</h2>
            </div>
            """.format(severity), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='background:#32CD32; padding:18px; border-radius:10px; text-align:center; color:white;'>
                <h4>Affected Asset</h4>
                <h2>{}</h2>
            </div>
            """.format(asset), unsafe_allow_html=True)

        st.markdown("---")
        st.success("🎉 Prediction Completed Successfully!")
