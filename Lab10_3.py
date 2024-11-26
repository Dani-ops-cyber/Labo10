# Importar bibliotecas necesarias
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Cargar los datos
data = pd.read_csv('/home/mario/Documentos/Embebidos/spam text data.csv')

# Modificar las etiquetas de la columna de categoría a valores binarios (spam = 1, no spam = 0)
data['Category'] = data['Category'].map({'spam': 1, 'ham': 0})

# Preparar las características y las etiquetas
X = data['Message']  # Usar la columna 'Message' en lugar de 'text'
y = data['Category']

# Vectorizar el texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X).toarray()  # Convierte el texto en datos numéricos

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Mensajes de prueba
test_messages = [
    "Win a free iPhone now!",  # Esto debería ser spam
    "Meeting tomorrow at 10am",  # Esto debería ser no spam
    "Limited time offer, claim your prize",  # Esto debería ser spam
    "Let's catch up this weekend!",  # Esto debería ser no spam
    "Congratulations! You have won a prize!", 
    "Can we schedule a meeting for tomorrow?",
    "Important meeting on Monday"
]

# Vectorizar los mensajes de prueba utilizando el mismo vectorizador
test_messages_vectorized = vectorizer.transform(test_messages).toarray()

# Realizar predicciones
predictions = model.predict(test_messages_vectorized)

# Mostrar resultados
for message, prediction in zip(test_messages, predictions):
    label = "Spam" if prediction == 1 else "Not Spam"
    print(f"Message: '{message}' - Prediction: {label}")
