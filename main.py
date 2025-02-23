import pandas as pd
from src.preprocess import load_data, preprocess_text, vectorize_text_bert
from src.model import train_lightgbm, train_neural_network, evaluate_model, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib

# Chargement des données
print("Chargement des données...")
train_data, test_data = load_data("data/train_data.txt", "data/test_data.txt")
print("Données chargées avec succès.")

# Prétraitement des textes
print("Prétraitement des textes...")
train_data = preprocess_text(train_data)
test_data = preprocess_text(test_data)
print("Prétraitement terminé.")

# Vectorisation des descriptions avec BERT
print("Vectorisation des textes avec BERT...")
X_train, y_train, X_test = vectorize_text_bert(train_data, test_data)
print("Vectorisation terminée.")

# Gestion du déséquilibre des classes avec SMOTE
print("Application de SMOTE pour équilibrer les classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE terminé.")

# Validation croisée
print("Entraînement avec validation croisée...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in skf.split(X_train_resampled, y_train_resampled):
    X_train_split, X_val = X_train_resampled[train_index], X_train_resampled[val_index]
    y_train_split, y_val = y_train_resampled[train_index], y_train_resampled[val_index]

    # Entraînement LightGBM
    print("Entraînement du modèle LightGBM...")
    lgbm_model = train_lightgbm(X_train_split, y_train_split)
    print("Évaluation du modèle LightGBM...")
    evaluate_model(lgbm_model, X_val, y_val)
    plot_confusion_matrix(lgbm_model, X_val, y_val)

    # Entraînement du réseau neuronal
    print("Entraînement du réseau neuronal...")
    nn_model = train_neural_network(X_train_split, y_train_split, X_val, y_val)
    print("Évaluation du réseau neuronal...")
    evaluate_model(nn_model, X_val, y_val, is_nn=True)

# Sauvegarde des modèles
joblib.dump(lgbm_model, 'lightgbm_model.pkl')
print("Modèle LightGBM sauvegardé.")
