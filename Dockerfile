#Version Python
FROM python:3.12.3

# Créer le dossier de travail
WORKDIR /app

# Copier les fichiers du projet
COPY nlp.parquet /app/
COPY . .

# Installer les dépendances Python
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Lancer train.py 
CMD bash -c "python train.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
