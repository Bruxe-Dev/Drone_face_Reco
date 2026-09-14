import os 
import pickle as pc 
import numpy as np

VECTOR_DB_PATH = "encodings/faces.pkl"

def load_vectors():
    if not os.path.exists(VECTOR_DB_PATH):
        return []

    with open (VECTOR_DB_PATH, 'rb') as file:
        return pc.load(file)


def save_vectors(vectors):
    os.makedirs('encodings',exist_ok=True)

    with open (VECTOR_DB_PATH,'wb') as file:
        pc.dump(vectors,file)

def add_vectors(name,embedding):

    vectors = load_vectors();

    vectors.append({
        "name":name,
        "embedding":embedding
    })

    save_vectors(vectors)

def cosine_similarity(a,b):
    a = a.flatten()
    b = b.flatten()

    return np.dot(a,b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def search_face(query_embedding):
    vectors = load_vectors()

    if not vectors:
        return "Unknown", -1

    best_name = "Unknown"
    best_score = -1

    for vector in vectors:
        score = cosine_similarity(
            query_embedding,
            vector["embedding"]
        )

        if score > best_score:
            best_score = score
            best_name = vector["name"]

    return best_name, best_score