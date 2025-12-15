import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
import time
import requests

from streamlit_lottie import st_lottie
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------- NLTK SETUP ----------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)


# ---------------- LOTTIE LOADER ----------------
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None


lottie_detect = load_lottie_url(
    "https://lottie.host/91a3439e-2a6a-44e4-97fd-d0383c1ac55d/5QkoYweKlm.json"
)
lottie_success = load_lottie_url(
    "https://lottie.host/7d084d07-3f18-4aa0-81ba-b96cf3a53d40/tfNjJSaI1K.json"
)
lottie_warning = load_lottie_url(
    "https://lottie.host/4e9d77fc-45d9-4301-9394-8ce57dd98b7d/xHjV0NZz2a.json"
)


# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    color: #e3f2fd;
}

.stTextArea textarea {
    background-color: #1e1e1e;
    color: #ffffff;
    border-radius: 12px;
    border: 1px solid #37474f;
    font-size: 16px;
}

textarea::placeholder {
    color: #9e9e9e;
}

.stButton > button {
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.6em;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1565c0, #1e88e5);
    transform: scale(1.05);
}

[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #263238;
}

[data-testid="stExpander"] {
    background-color: #1c1f26;
    border-radius: 12px;
    border: 1px solid #263238;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.title("📧 Email Spam Detection System")

if lottie_detect:
    st_lottie(lottie_detect, height=200)

st.markdown(
    "Detect whether a message is **SPAM** or **NOT SPAM** using **Machine Learning & NLP**."
)


# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]

    return " ".join(words)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "spam.csv",
            sep="\t",
            header=None,
            names=["label", "message"]
        )
    except FileNotFoundError:
        st.error("❌ spam.csv file not found")
        st.stop()

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(clean_text)
    return df


# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy


df = load_data()
model, vectorizer, accuracy = train_model(df)


# ---------------- DATA PREVIEW ----------------
with st.expander("📄 Dataset Preview"):
    st.dataframe(df.head())


# ---------------- DISTRIBUTION CHART ----------------
st.subheader("📊 Message Distribution")

fig, ax = plt.subplots()
fig.patch.set_facecolor("#1c1f26")
ax.set_facecolor("#1c1f26")

df["label"].value_counts().plot(kind="bar", ax=ax)

ax.set_title("Ham vs Spam", color="white")
ax.set_xlabel("Type", color="white")
ax.set_ylabel("Count", color="white")
ax.tick_params(colors="white")

st.pyplot(fig)


# ---------------- MODEL ACCURACY ----------------
st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")


# ---------------- USER INPUT ----------------
st.subheader("📝 Try It Yourself")

user_input = st.text_area(
    "Paste your message here:",
    height=150,
    placeholder="Congratulations! You have won a free prize..."
)

if st.button("🔍 Detect Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(1)

            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0][prediction]

        st.progress(float(probability))

        if prediction == 1:
            if lottie_warning:
                st_lottie(lottie_warning, height=180)
            st.error(f"🚨 **SPAM MESSAGE** (Confidence: {probability:.2%})")
        else:
            if lottie_success:
                st_lottie(lottie_success, height=180)
            st.success(f"✅ **NOT SPAM** (Confidence: {probability:.2%})")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "🌐 Made with ❤️ by **TEAM CLOUDCORE**  \n"
    "[GitHub](https://github.com/rayyan0r/spam-detector)"
)
