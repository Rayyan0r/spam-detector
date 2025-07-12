import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])
    st.write(df.head())
    st.write(df['label'].value_counts())
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_message'] = df['message'].apply(clean_text)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Train model
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# Streamlit UI
st.title("📧 Email Spam Detector")

df = load_data()
model, vectorizer, accuracy = train_model(df)

st.write(f"Model accuracy: **{accuracy:.2%}**")

user_input = st.text_area("Enter a message to check if it's spam:")

if st.button("Check"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    if prediction == 1:
        st.error("🚨 This message is **SPAM**")
    else:
        st.success("✅ This message is **NOT SPAM**")
