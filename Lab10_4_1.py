import pandas as pd

# Crear un diccionario de datos
data = {
    "Student Name": ["Alice", "Bob", "Charlie", "David"],
    "Student Age": [20, 21, 19, 22],
    "No. of Lab completed": [10, 9, 8, 7],
    "Average score": [85, 90, 78, 88]
}

# Crear un DataFrame y guardar en un archivo Excel
df = pd.DataFrame(data)
df.to_excel("student_data.xlsx", index=False)
print("Hoja de cálculo 'student_data.xlsx' creada exitosamente.")
