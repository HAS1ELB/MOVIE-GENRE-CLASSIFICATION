from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim

def train_lightgbm(X_train, y_train):
    """
    Entraîne un modèle LightGBM avec des classes pondérées.
    """
    model = LGBMClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train, X_val, y_val):
    """
    Entraîne un réseau neuronal simple.
    """
    class SimpleNN(nn.Module):
        def __init__(self, input_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, output_size)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    input_size = X_train.shape[1]
    output_size = len(set(y_train))
    model = SimpleNN(input_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, X_test, y_test, is_nn=False):
    """
    Évalue les performances d'un modèle avec un rapport de classification.
    :param model: Modèle entraîné (scikit-learn ou PyTorch)
    :param X_test: Données de test
    :param y_test: Labels de test
    :param is_nn: Si le modèle est un réseau neuronal PyTorch
    """
    if is_nn:
        # Convertir les données en tenseurs PyTorch
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        model.eval()  # Mode évaluation
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, axis=1).numpy()
    else:
        predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))


def plot_confusion_matrix(model, X_test, y_test):
    """
    Affiche une matrice de confusion.
    """
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test).plot()
