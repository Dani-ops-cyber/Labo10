import tensorflow as tf
import numpy as np

# Datos de entrada (X) y salida (Y) basados en la ecuación Y = 3X + 2
X_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
Y_train = 3 * X_train + 2

# Crear un modelo secuencial con más capas y neuronas
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=[1]),  # Capa oculta con 10 neuronas
    tf.keras.layers.Dense(units=1)  # Capa de salida con 1 neurona
])

# Usar el optimizador Adam y la función de pérdida de error cuadrático medio
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, Y_train, epochs=500)

# Convertir los valores de entrada a arrays de NumPy para predecir
Y_pred_5 = model.predict(np.array([5.0]))
Y_pred_3_3 = model.predict(np.array([3.3]))

print("Predicción para X = 5:", Y_pred_5)
print("Predicción para X = 3.3:", Y_pred_3_3)
