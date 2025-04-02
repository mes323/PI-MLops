import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

movies_sample = pd.read_csv(r"C:\Users\Usuario\Desktop\Henry- DS\LABS\PI -MLOPS\MLOPS\app\movies_final.csv")

# Distribución de las películas por año
plt.figure(figsize=(10, 6))
sns.countplot(data=movies_sample, x="release_year")
plt.title("Distribución de Películas por Año")
plt.xlabel("Año de Estreno")
plt.ylabel("Cantidad de Películas")
plt.xticks(rotation=45)
plt.show()

# Distribución de los puntajes de las películas
plt.figure(figsize=(10, 6))
sns.histplot(movies_sample["vote_average"], bins=30, kde=True)
plt.title("Distribución de Puntajes de Películas")
plt.xlabel("Puntaje")
plt.ylabel("Frecuencia")
plt.show()

# Relación entre presupuesto y ganancias
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movies_sample, x="budget", y="revenue")
plt.title("Relación entre Presupuesto y Ganancias")
plt.xlabel("Presupuesto")
plt.ylabel("Ganancias")
plt.show()

# 4. Distribución de Géneros
# Convertir la columna 'genres' en una serie con todos los géneros por separado
genre_series = movies_sample["genres"].dropna().str.split(", ").explode()
plt.figure(figsize=(10, 6))
sns.countplot(y=genre_series, order=genre_series.value_counts().index)
plt.title("Distribución de Géneros")
plt.xlabel("Cantidad de Películas")
plt.ylabel("Género")
plt.show()
