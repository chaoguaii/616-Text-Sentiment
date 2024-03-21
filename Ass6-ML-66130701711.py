
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
model = pickle.load(open('naive_bayes_model.sav', 'rb'))

# Load TfidfVectorizer
vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

# Set title
st.title("Review Sentiment Prediction using Naive Bayes")

# Text input for user input
user_input = st.text_input("Enter your review:")

# Transform user input using TfidfVectorizer
user_input_vec = vectorizer.transform([user_input])

# Predict sentiment
pred = model.predict(user_input_vec)

# Display prediction result
st.write("## Prediction Result:")
st.write('Sentiment:', pred[0])
