import pandas as pd
import re
from transformers import BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_data(train_file, test_file):
    """
    Charge les données d'entraînement et de test depuis des fichiers.
    """
    train_data = pd.read_csv(train_file, sep=' ::: ', engine='python', header=None, names=["id", "title", "genre", "description"])
    test_data = pd.read_csv(test_file, sep=' ::: ', engine='python', header=None, names=["id", "title", "description"])
    return train_data, test_data

def preprocess_text(data):
    """
    Nettoie les descriptions en supprimant les caractères inutiles.
    """
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return text

    data['cleaned_description'] = data['description'].apply(clean_text)
    return data

def vectorize_text_bert(train_data, test_data):
    """
    Vectorise les descriptions avec BERT et encode les genres.
    """
    def encode_text(texts):
        encoded = tokenizer(texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
        return encoded['input_ids']

    train_encoded = encode_text(train_data['cleaned_description'])
    test_encoded = encode_text(test_data['cleaned_description'])

    # Encodage des genres
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['genre'])

    return train_encoded, y_train, test_encoded
