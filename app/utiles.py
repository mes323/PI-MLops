import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Funciones de Extracción -----------------

def extract_names(json_str):
    """
    Extrae y une los nombres de una lista de diccionarios (por ejemplo, para géneros o productoras).
    """
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            return ", ".join([item.get("name", "") for item in data if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

def extract_collection(json_str):
    """
    Extrae el nombre de la colección a la que pertenece la película.
    """
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, dict) and "name" in data:
            return data["name"]
        elif isinstance(data, list):
            return ", ".join([item.get("name", "") for item in data if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

def extract_production_countries(json_str):
    """
    Extrae los nombres de los países de producción.
    """
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            return ", ".join([item.get("name", "") for item in data if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

def extract_actors(json_str, top_n=5):
    """
    Extrae los nombres de los actores (limitando a los primeros top_n).
    """
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            return ", ".join([item.get("name", "") for item in data[:top_n] if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

def extract_director(json_str):
    """
    Extrae el nombre del director desde el campo crew.
    """
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            for item in data:
                if item.get("job", "").lower() == "director":
                    return item.get("name", None)
    except (ValueError, SyntaxError):
        return None
    return None

# ----------------- Funciones de Carga/Guardado -----------------

def load_csv(filepath):
    """
    Carga un archivo CSV en un DataFrame de pandas.
    """
    return pd.read_csv(filepath)

def save_csv(df, filepath):
    """
    Guarda un DataFrame en un archivo CSV.
    """
    df.to_csv(filepath, index=False)

# ----------------- Utilidades para el Sistema de Recomendación -----------------

def build_tfidf_matrix(features):
    """
    Crea la matriz TF-IDF a partir de una serie de características.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf, tfidf.fit_transform(features.fillna(''))

def compute_cosine_similarity(tfidf_matrix):
    """
    Calcula la similitud de coseno entre las filas de la matriz TF-IDF.
    """
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movie(title, movies_df, cosine_sim):
    """
    Recomendación de películas basándose en la similitud de coseno.
    """
    indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower())
    title_lower = title.lower()
    if title_lower not in indices:
        return []
    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# ----------------- Constantes y Mapeos -----------------

WEEKDAY_MAPPING = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miércoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sábado",
    "Sunday": "domingo"
}

SPANISH_MONTH_MAPPING = {
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
