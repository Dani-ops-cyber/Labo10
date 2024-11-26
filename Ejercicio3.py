#Ejercicio 2 modificacion del ejercicio 1
## 
#Con el modelo generado en el Ejercicio 1 modifique el número de capas, neuronas y la tipo 
# de funciones de activación para comprobar si los resultados se pueden mejorar.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

xs = np.array([3.3, 5], dtype=float)
ys = np.array([11.9, 17], dtype=float)

model = keras.Sequential()

# Modelo con 5 capas y 4 neuronas 
model.add(layers.Dense(4, input_dim=1))  # Primera capa, 4 neuronas
model.add(layers.Dense(4))               # Segunda capa, 4 neuronas
model.add(layers.Dense(4))               # Tercera capa, 4 neuronas
model.add(layers.Dense(4))               # Cuarta capa, 4 neuronas
model.add(layers.Dense(1))               # Quinta capa (capa de salida), 1 neurona para salida

model.compile(keras.optimizers.Adam(0.1), loss="mean_squared_error", metrics=['accuracy'])
model.fit(xs, ys, epochs=300)
predictions = model.predict(xs)
rounded_predictions = np.round(predictions)
_, accuracy = model.evaluate(xs, ys)
print("The accuracy is {}".format(accuracy))
resultado=model.predict(np.array([10.0])) # valor aleaotorio 
print("Resultado Y: " + str(resultado[0][0]))