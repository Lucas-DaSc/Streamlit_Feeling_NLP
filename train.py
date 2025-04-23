# Train

# Dépendances
import pandas as pd
import joblib
import pyarrow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath):
    data = pd.read_parquet(filepath, engine='pyarrow')

    return data

def preprocess_data(data):
    # Vectorisation 
    vectorizer = TfidfVectorizer(
    max_features=5000,  
    ngram_range=(1,2), 
    stop_words='english')

    #
    X = vectorizer.fit_transform(data['text'])
    y = data['label']  

    # Save vecto
    joblib.dump(vectorizer, "vectorizer_nlp.pkl")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train):
    # Model & parameters
    model = MultinomialNB()

    params = {'alpha' : [0.1, 0.5, 1.0,3.0, 10.0, 30.0, 50.0, 100.0],
          'force_alpha' : [True, False],
          'fit_prior' : [True, False], 
          'class_prior': [None]}

    grid = RandomizedSearchCV(model, params, cv=5, n_iter=10,random_state=42)
    grid.fit(X_train, y_train)
    
    print(grid.best_params_)

    y_pred = grid.predict(X_test)

    return grid, y_pred

def evaluation_model(y_test, y_pred, data):
    # Emotion
    emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    data['emotion'] = data['label'].map(emotion_map)
    # Metrique
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=emotion_map.values())

    return acc, report 

def save_model(grid):
    joblib.dump(grid, "model_nlp.pkl")
    return grid  

def run_pipeline(filepath):
    # Étape 1 : Chargement des données
    data = load_data(filepath)
    
    # Étape 2 : Preprocess des données
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Étape 3 : Entraînement du modèle
    grid, y_pred = train_model(X_train, X_test, y_train)

    # Étape 4 : Évaluation
    acc, report = evaluation_model(y_test, y_pred, data)

    # Étape 5 : Sauvegarde
    model = save_model(grid)

    # Afficher les résultats
    print(f'Accuracy: {acc}')
    print(f'Classification Report: {report}')

run_pipeline("nlp.parquet")