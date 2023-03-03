from train import train
from data_utils import (
    synthesize_database, 
    create_multi_output_trainset,
    create_multi_input_trainset,
    get_embeddings
)

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import numpy as np
from typings import *

# def train_model():
#     embed_data, id_data, date_data, location_data = synthesize_database()
#     y_enc = to_categorical(id_data)       
#     X_train = standardize_input(np.stack(embed_data))
#     model = MySequentialModel(out_size=len(set(id_data)))
#     train(model, X_train,  y_enc)


# def standardize_input(X_train: Embeddings) -> Embeddings:
#     sc = StandardScaler()
#     return sc.fit_transform(X_train)


def train_multioutput_model(embed_original):
    
    X_train, ids, labels = create_multi_output_trainset(*synthesize_database(embed_original))
    
    # Just to make TFDP happy, the length of X_train must be evenly divisible by batch size
    # X_train = X_train[:5000]
    # labels = labels[:5000]

    print(f'X_train: {X_train.shape}, labels: {labels.shape}')
    model = train(X_train, labels, loss_func='BinaryCrossentropy')
    return model

def train_multiinput_model(embed_original):
    X_train, ids, labels =  create_multi_input_trainset(*synthesize_database(embed_original))
    print(f'X_train: {X_train.shape}, labels: {labels.shape}')
    model = train(X_train, labels, loss_func='CategoricalCrossentropy')
    return model



if __name__ == '__main__':
    user_selection = 'multi-output'
    embed_original = get_embeddings() # Considered the query pictures
    embed_queries = embed_original[:100]
    
    if user_selection == 'multi-output':
        model = train_multioutput_model(embed_original)
        predictions = model.predict(embed_queries)
        for i in range(20):
            idx = (-predictions[i]).argsort()[:]
            print(idx)
    else:
        model = train_multiinput_model(embed_original)