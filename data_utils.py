import pickle
import random
from sklearn.preprocessing import StandardScaler
from typings import *
from global_variables import frequency_range, date_range, location_range, training_size, countries
import pandas as pd
import os
from datetime import date, timedelta

# Set random seed
SEED = 1433 
random.seed(SEED)


# Functions
def get_passenger_database(
    embed_original: list[Embedding],
    indices: list[int], 
    data_path: str = 'dataset/passenger_data/passenger.csv',
    names_path: str = 'dataset/names/query-names.txt',
) -> pd.DataFrame:
    if os.path.exists(data_path):
        return pd.read_csv(data_path)

    with open(names_path, 'r') as f:
        names = [name.strip() for name in f.readlines()]

    test_date1, test_date2 = date(1980, 1, 1), date(2000, 1, 1)
    dates_bet = test_date2 - test_date1
    total_days = dates_bet.days

    rows = []
    # Synthesize passenger database
    for j, index in enumerate(indices):
        row = {"id": index}
        row["name"] = names[j] # assume names are distinct
        row["dob"] = str(test_date1 + timedelta(days=random.randrange(total_days)))  
        row["country"] = random.choice(countries)
        # row["embed"] = embed_original[j]
        rows.append(row)
    database = pd.DataFrame(rows)
    database.to_csv(data_path, index=False)
    print(database)
    return database

def check_passenger_exist(
    passenger_data,
    name_query,
    datebirth,
    country,
    picture_id,
):
    for row in passenger_data.iter_rows():
        if row['id'] == picture_id and row["name"] == name_query and row['dob'] == datebirth and row['country'] == country:
            return picture_id
    return -1

def get_embeddings(
    path: str = 'dataset/embeddings/10K_encodings.pkl', 
) -> tuple[list[Embedding], list[int]]:
    embeddings = pickle.load(open(path, 'rb'))
    size = training_size // frequency_range
    if size > len(embeddings):
        raise ValueError("Size exceed the embeddings size!")
    if size == len(embeddings):
        return embeddings

    indices = random.sample(range(len(embeddings)), size)
    embed_original = [embeddings[i] for i in indices]
    return (embed_original, indices)

def transform_embedding(embedding: Embedding) -> Embedding:
    return embedding

def synthesize_database(
    embed_original: list[Embedding],
    is_transform: bool = True,  
    is_fix_freq: bool = True,
) -> tuple[list[Embedding], list[int], list[int], list[int]]:
    
    if is_fix_freq:
        embed_data, id_data = [], []
        date_data, location_data = [], []
        for i, embedding in enumerate(embed_original):
            embed_data.extend([embedding] * frequency_range)
            id_data.extend([i] * frequency_range)
            dates = random.sample(range(date_range), frequency_range)
            date_data.extend([date + 1 for date in dates])
            locations = random.sample(range(location_range), frequency_range)
            location_data.extend([location + 1 for location in locations])
    else:
        if is_transform:
            embed_data, id_data = [], []
            date_data, location_data = [], []
            for i, embedding in enumerate(embed_original):
                for _ in range(random.randint(1, frequency_range)):
                    embed_data.append(transform_embedding(embedding))
                    id_data.append(i)
                    date_data.append(random.randint(1, date_range))
                    location_data.append(random.randint(1, location_range))
            
        else:
            embed_data = embed_original
            length = len(embed_data)
            id_data = [i for i in range(length)]
            date_data = [random.randint(1, date_range) for _ in range(length)]
            location_data = [random.randint(1, location_range) for _ in range(length)]

    return (embed_data, id_data, date_data, location_data)


def standardize_input(X_train: Embeddings) -> tuple[Embeddings, StandardScaler]:
    sc = StandardScaler()
    sc.fit(X_train)
    return (sc.transform(X_train), sc)

def create_multi_output_trainset(
    embed_data: list[Embedding], 
    id_data: list[int], 
    date_data: list[int], 
    location_data: list[int],
) -> tuple[Embeddings, np.array, np.array]:

    labels = np.zeros((len(id_data), date_range * location_range), dtype=np.int32)
    for id, date, location in zip(id_data, date_data, location_data):
        labels[id, date * location - 1] = 1
    X_train, _ = standardize_input(np.stack(embed_data))
    return (X_train, np.array(id_data), labels)
    

def create_multi_input_trainset(    
    embed_data: list[Embedding], 
    id_data: list[int], 
    date_data: list[int], 
    location_data: list[int],
    is_onehot_encode: bool = True,
):
    # Create labels
    location_data = np.array(location_data)
    labels = np.zeros((len(location_data), location_range), dtype=np.int32)
    labels[np.arange(len(location_data)), location_data - 1] = 1

    # Create X_train
    date_data = np.array(date_data)
    if is_onehot_encode:
        date_data_array = np.zeros((len(date_data), date_range), dtype=np.int32)
        date_data_array[np.arange(len(date_data)), date_data - 1] = 1
    else:
        date_data_array = np.expand_dims(date_data, axis=1)
    X_train = np.concatenate((np.stack(embed_data), date_data_array), axis=1)
    X_train, _ = standardize_input(X_train)
    return (X_train, np.array(id_data), labels)