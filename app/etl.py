# Cargar los datasets
credits_path = "credits.csv"
movies_path = "movies_dataset.csv"

credits_df = pd.read_csv(credits_path)
movies_df = pd.read_csv(movies_path)

# ----------------- Transformaciones en movies_dataset.csv -----------------

# Convertir 'budget' y 'revenue' a numérico, reemplazando nulos por 0
movies_df["budget"] = pd.to_numeric(movies_df["budget"], errors="coerce").fillna(0)
movies_df["revenue"] = movies_df["revenue"].fillna(0)

# Convertir 'release_date' a formato AAAA-mm-dd y extraer 'release_year'
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors="coerce")
movies_df["release_year"] = movies_df["release_date"].dt.year

# Calcular retorno de inversión (return = revenue / budget)
movies_df["return"] = movies_df.apply(
    lambda row: row["revenue"] / row["budget"] if row["budget"] > 0 else 0, axis=1
)

# Eliminar columnas innecesarias
columns_to_drop = ["video", "imdb_id", "adult", "original_title", "poster_path", "homepage"]
movies_df.drop(columns=columns_to_drop, inplace=True)

# Funciones para desanidar campos anidados

# Extraer nombres de listas de diccionarios (para genres, production_companies, spoken_languages)
def extract_names(json_str):
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            return ", ".join([item["name"] for item in data if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

# Extraer el nombre de la colección (belongs_to_collection)
def extract_collection(json_str):
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, dict) and "name" in data:
            return data["name"]
        elif isinstance(data, list):
            return ", ".join([item["name"] for item in data if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

# Extraer países de producción
def extract_production_countries(json_str):
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            return ", ".join([item["name"] for item in data if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

# Desanidar las columnas anidadas
movies_df["genres"] = movies_df["genres"].astype(str).apply(extract_names)
movies_df["production_companies"] = movies_df["production_companies"].astype(str).apply(extract_names)
movies_df["spoken_languages"] = movies_df["spoken_languages"].astype(str).apply(extract_names)
movies_df["belongs_to_collection"] = movies_df["belongs_to_collection"].astype(str).apply(extract_collection)
movies_df["production_countries"] = movies_df["production_countries"].astype(str).apply(extract_production_countries)

# ----------------- Transformaciones en credits.csv -----------------

# Extraer nombres de actores (cast) y director (crew)
def extract_actors(json_str):
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            # Extrae los primeros 5 actores
            return ", ".join([item["name"] for item in data[:5] if "name" in item])
    except (ValueError, SyntaxError):
        return None
    return None

def extract_director(json_str):
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            for item in data:
                if item.get("job") == "Director":
                    return item["name"]
    except (ValueError, SyntaxError):
        return None
    return None

credits_df["actors"] = credits_df["cast"].astype(str).apply(extract_actors)
credits_df["director"] = credits_df["crew"].astype(str).apply(extract_director)

# Limpieza de la columna 'id' para asegurar que solo tenga valores numéricos
movies_df = movies_df[pd.to_numeric(movies_df['id'], errors='coerce').notnull()]
credits_df = credits_df[pd.to_numeric(credits_df['id'], errors='coerce').notnull()]

# Convertir la columna 'id' a tipo entero en ambos DataFrames
movies_df['id'] = movies_df['id'].astype(int)
credits_df['id'] = credits_df['id'].astype(int)

# Seleccionar solo columnas necesarias de credits_df y unir con movies_df
credits_df = credits_df[["id", "actors", "director"]]
movies_sample = movies_df.merge(credits_df, on="id", how="left")

# Verificar resultados
movies_sample.head()

# Para poder trabajar con un Dataset mas corto extraemos una muestra de manera aleatoria.
#movies_final = movies_df.sample(n=4000, random_state=42)
# Usamos una muestra del 10% para poder correr el modelo en Render.
#movies_final.to_csv("movies_final.csv")