import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# Custom style
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button { background-color: #ff4b4b; color: white; font-weight: bold; }
    .stTextInput>div>input { font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("📧 Email Spam Detector")
st.write("This app uses a machine learning model to detect if a message is **spam** or **not spam**.")

# Clean text
def clean_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.strip().split()
    filtered = [stemmer.stem(word) for word in words if word not in stop_words]

    return ' '.join(filtered)

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_message'] = df['message'].apply(clean_text)
    return df

# Train model
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# Load data and train
df = load_data()
model, vectorizer, accuracy = train_model(df)

# Show data
with st.expander("📂 View Sample Data"):
    st.write(df.head())

# Chart
st.subheader("📊 Message Class Distribution")
fig, ax = plt.subplots()
df['label'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
ax.set_title("Number of Ham vs Spam Messages")
ax.set_xlabel("Message Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# Accuracy
st.write(f"✅ **Model Accuracy**: {accuracy:.2%}")

# User input
st.subheader("✉️ Try It Yourself")
user_input = st.text_area("Enter a message to check if it's spam:", height=150)

if st.button("🔎 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 This message is **SPAM** ({proba:.2%} confidence)")
        else:
            st.success(f"✅ This message is **NOT SPAM** ({proba:.2%} confidence)")

# Footer
st.markdown("""
---
TEAM CLOUDCORE
[View Source on GitHub](https://github.com/Rayyan0r/spam-detector)
""")
   


