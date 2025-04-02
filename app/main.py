from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI()

# -------------------- Cargar y Preparar el Dataset --------------------
# Carga el dataset completo (la muestra extraida en ETL)
movies_sample = pd.read_csv("app/movies_final.csv")


# Convertir la columna release_date a datetime y extraer año y mes
movies_sample["release_date"] = pd.to_datetime(movies_sample["release_date"], errors="coerce")
movies_sample["release_year"] = movies_sample["release_date"].dt.year
movies_sample["release_month"] = movies_sample["release_date"].dt.month

# Mapear los días de la semana a español
weekday_mapping = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miércoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sábado",
    "Sunday": "domingo"
}
movies_sample["day_name_es"] = movies_sample["release_date"].dt.day_name().map(weekday_mapping)

# Diccionario para meses en español
spanish_month_mapping = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12
}

# -------------------- Endpoints de la API --------------------

@app.get("/")
def root():
    return {"message": "API de películas funcionando con movies_sample!"}

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    mes_lower = mes.lower()
    if mes_lower not in spanish_month_mapping:
        raise HTTPException(status_code=400, detail="Mes inválido. Usa, por ejemplo, 'enero', 'febrero', etc.")
    mes_num = spanish_month_mapping[mes_lower]
    count = movies_sample[movies_sample["release_month"] == mes_num].shape[0]
    return {"message": f"{count} películas fueron estrenadas en el mes de {mes}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dia_lower = dia.lower()
    count = movies_sample[movies_sample["day_name_es"] == dia_lower].shape[0]
    return {"message": f"{count} películas fueron estrenadas en el día {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    movie = movies_sample[movies_sample["title"].str.lower() == titulo.lower()]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    movie = movie.iloc[0]
    return {"message": f"La película {movie['title']} fue estrenada en {int(movie['release_year'])} con un score de {movie['vote_average']}"}

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    movie = movies_sample[movies_sample["title"].str.lower() == titulo.lower()]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    movie = movie.iloc[0]
    if movie["vote_count"] < 2000:
        return {"message": f"La película {movie['title']} no cumple con la condición de tener al menos 2000 valoraciones."}
    return {"message": f"La película {movie['title']} tiene {int(movie['vote_count'])} votos con un promedio de {movie['vote_average']}"}

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    # Para recomendaciones, combinamos algunas características. Asegúrate que existan las columnas 'genres', 'actors' y 'director'
    # Si estas columnas no están presentes en tu dataset, deberás integrarlas en tu ETL.
    movies_sample["features"] = (
        movies_sample["genres"].fillna('') + " " +
        movies_sample["actors"].fillna('') + " " +
        movies_sample["director"].fillna('')
    )
    # Crear la matriz TF-IDF y la similitud de coseno
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_sample["features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(movies_sample.index, index=movies_sample['title'].str.lower())
    title_lower = titulo.lower()
    if title_lower not in indices:
        raise HTTPException(status_code=404, detail="Película no encontrada para recomendaciones")
    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Excluye la misma película
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_sample["title"].iloc[movie_indices].tolist()
    return {"recommended_movies": recommendations}

# -------------------- Ejecutar la API --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

