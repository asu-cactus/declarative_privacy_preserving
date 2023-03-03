import pickle
import random
from sklearn.preprocessing import StandardScaler
from typings import *

# Set random seed
SEED = 1433 
random.seed(SEED)

# Global settings
frequency_range = 10
date_range = 365
location_range = 20

# Functions
def get_embeddings(
        path: str = 'dataset/embeddings/10K_encodings.pkl', 
        size: str = 1000
) -> list[Embedding]:
    embeddings = pickle.load(open(path, 'rb'))
    if size > len(embeddings):
        raise ValueError("Size exceed the embeddings size!")
    return random.sample(embeddings, size)

def transform_embedding(embedding: Embedding) -> Embedding:
    return embedding

def synthesize_database(embed_original) -> tuple[list[Embedding], list[int], list[int], list[int]]:
    embed_data, id_data = [], []
    date_data, location_data = [], []
    for i, embedding in enumerate(embed_original):
        fequency = random.randint(1, frequency_range)
        # fequency = 1
        for _ in range(fequency):
            embed_data.append(transform_embedding(embedding))
            id_data.append(i)
            date_data.append(random.randint(1, date_range))
            location_data.append(random.randint(1, location_range))
    
    return (embed_data, id_data, date_data, location_data)


def standardize_input(X_train: Embeddings) -> Embeddings:
    sc = StandardScaler()
    return sc.fit_transform(X_train)

def create_multi_output_trainset(
    embed_data: list[Embedding], 
    id_data: list[int], 
    date_data: list[int], 
    location_data: list[int]
) -> tuple[Embeddings, np.array, np.array]:
    labels = np.zeros((len(id_data), date_range * location_range), dtype=np.int32)
    for id, date, location in zip(id_data, date_data, location_data):
        labels[id, date * location - 1] = 1

    X_train = standardize_input(np.stack(embed_data))
    return (X_train, np.array(id_data), labels)
    

def create_multi_input_trainset(    
    embed_data: list[Embedding], 
    id_data: list[int], 
    date_data: list[int], 
    location_data: list[int]
):
    date_data = np.expand_dims(np.array(date_data), axis=1)
    X_train = np.concatenate((np.stack(embed_data), date_data), axis=1)
    labels = np.array(location_data)

    return (X_train, np.array(id_data), labels)
