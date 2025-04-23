from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle pré-entrainé
model = joblib.load('model_nlp.pkl')

# Créer une instance de FastAPI
app = FastAPI()

# Définir un modèle Pydantic pour la validation des entrées
class Feelings(BaseModel):
    text: str

# Route de prédiction
@app.post("/predict/")
async def predict_price(feel: Feelings):
    # Extraire les caractéristiques 
    features = np.array([[feel.text]])
    
    # Prédire le sentiment
    predicted_feel = model.predict(features)[0]
    
    # Retourner la prédiction
    return {"Prédiction du sentiment": predicted_feel}