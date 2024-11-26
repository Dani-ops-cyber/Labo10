# EJercicio 1
# entrenar un modelo
# Given the following equation: Y = 3X + 2 Generate an ANN to obtain an 
# approximate result for the following values: X = 5, X = 3.3.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

xs = np.array([3.3, 5], dtype=float)
ys = np.array([11.9, 17], dtype=float)

model = keras.Sequential()
model.add(layers.Dense(3, input_dim=1)) #First 1 layer, then 2 (5-3) and finally 2 (3-3)
model.add(layers.Dense(3))
model.add(layers.Dense(1))

model.compile(keras.optimizers.Adam(0.02), loss="mean_squared_error", metrics=['accuracy'])
model.fit(xs, ys, epochs=10)

predictions = model.predict(xs)
rounded_predictions = np.round(predictions)
_, accuracy = model.evaluate(xs, ys)
print("The accuracy is {}".format(accuracy))


resultado=model.predict(np.array([5]))
print("Resultado Y: " + str(resultado[0][0]))