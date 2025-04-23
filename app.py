import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle
@st.cache_resource
def load_model():
    with open("model_nlp.pkl", "rb") as file:
        model = joblib.load(file)
    return model

model = load_model()
vectorizer = joblib.load("vectorizer_nlp.pkl")

# Interface utilisateur
st.title("Analyse de Sentiment d'un Tweet")

# Entrée des caractéristiques 
text = st.text_area("Colle ici ton texte :", height=500)

if text:
    st.write("Texte reçu :")

if text:
    st.subheader("Texte prétraité :")
    clean_text = vectorizer.transform([text])
    st.write(clean_text)

# Prédiction

label_to_emotion = {
    0: "Tristesse",
    1: "Joie",
    2: "Amour",
    3: "Colère",
    4: "Peur",
    5: "Surprise"
}

if st.sidebar.button("Prédire le sentiment"):
    try:
        feel = model.predict(clean_text) 
        label = feel[0]  # ex: 3
        emotion = label_to_emotion.get(label, "Inconnu")

        st.success(f"Sentiment détecté : **{emotion}** (label : {label})")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")