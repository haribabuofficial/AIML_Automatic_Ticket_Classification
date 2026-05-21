import streamlit as st
import numpy as np
import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import google.generativeai as genai

# -----------------------------
# Gemini API Configuration
# -----------------------------

genai.configure(api_key="AIzaSyBaO_jDOoqpmZT6C_tqoL2AHeaRwL_cfm0")

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# Load Saved Files
# -----------------------------

model = load_model("file/bidir_lstm_model.keras")

with open("file/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("file/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Customer Support Ticket Classifier")

st.write("Predict support queue and generate auto reply")

ticket = st.text_area("Enter Customer Ticket")

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Queue"):

    if ticket.strip() != "":

        # Clean text
        cleaned = clean_text(ticket)

        # Tokenize
        seq = tokenizer.texts_to_sequences([cleaned])

        # Padding
        padded = pad_sequences(seq, maxlen=100)

        # Prediction
        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        predicted_queue = le.inverse_transform([predicted_class])[0]

        st.subheader("Predicted Queue")
        st.success(predicted_queue)

        # -----------------------------
        # Gemini Prompt
        # -----------------------------

        prompt = f"""
        Customer Ticket:
        {ticket}

        Predicted Department:
        {predicted_queue}

        Generate a polite professional support reply.
        """

        response = gemini_model.generate_content(prompt)

        st.subheader("Generated Reply")

        st.write(response.text)

    else:
        st.warning("Please enter a ticket")