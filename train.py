import tensorflow as tf
from modeling import MySequentialModel, CrossProductOutputModel
from global_variables import date_range, location_range, training_size, learning_rate, epochs
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from data_utils import (
    synthesize_database, 
    create_multi_output_trainset,
    create_multi_input_trainset,
    create_simple_trainset,
    get_embeddings
)
from typings import *
from global_variables import date_range, location_range


# def train_model():
#     embed_data, id_data, date_data, location_data = synthesize_database()
#     y_enc = to_categorical(id_data)       
#     X_train = standardize_input(np.stack(embed_data))
#     model = MySequentialModel(out_size=len(set(id_data)))
#     train(model, X_train,  y_enc)


# def standardize_input(X_train: Embeddings) -> Embeddings:
#     sc = StandardScaler()
#     return sc.fit_transform(X_train)


# def train_multi_output_model(embed_original):
#     X_train, ids, labels = create_multi_output_trainset(*synthesize_database(embed_original))
#     print(f'X_train: {X_train.shape}, labels: {labels.shape}')
#     model = train(X_train, labels, loss_func='BinaryCrossentropy', is_privacy_preserve=True)
#     return model

def train(
    X_train, 
    y_train,
    out_size: int,
    loss_func: str = 'CategoricalCrossentropy',
    batch_size: int = 100,
    num_microbatches: int = 1,
    l2_norm_clip: float = 1,
    noise_multiplier: float = 0.03,
    learning_rate: float = 0.02,
    epochs: int = 200,
    is_privacy_preserve: bool = True,
):
    if batch_size % num_microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')
    
    if is_privacy_preserve:
        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define model and loss function
    if loss_func == 'CategoricalCrossentropy':
        model = MySequentialModel(out_size=out_size)
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    elif loss_func == 'BinaryCrossentropy':
        model = CrossProductOutputModel(out_size=out_size)
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    else:
        raise ValueError
    
    # Fit model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    print(f'n={X_train.shape[0]}, batch_size={batch_size}')
    eps = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=X_train.shape[0],
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=1e-3)
    
    return (model, eps, X_train, y_train)


def train_multi_output_model(embed_data, id_data, date_data, location_data):
    X_train, ids, labels, sc = create_multi_output_trainset(embed_data, id_data, date_data, location_data)
    print(f'X_train: {X_train.shape}, labels: {labels.shape}')
    model, eps, X_train, y_train = train(
        X_train, labels, 
        out_size=location_range*date_range, 
        loss_func='CategoricalCrossentropy',
        is_privacy_preserve=True, 
        learning_rate=learning_rate, 
        epochs=epochs)
    return (model, eps, X_train, y_train, sc)

def train_multi_input_model(embed_data, id_data, date_data, location_data):
    X_train, ids, labels, sc =  create_multi_input_trainset(embed_data, id_data, date_data, location_data)
    print(f'X_train: {X_train.shape}, labels: {labels.shape}')
    model, eps, X_train, y_train =  train(X_train, labels, out_size=location_range , is_privacy_preserve=True, learning_rate=learning_rate, epochs=epochs)
    return (model, eps, X_train, y_train, sc)
    # return train(X_train, labels, out_size=location_range, is_privacy_preserve=True, learning_rate=0.002, epochs=100)


def train_simple(embed_data, id_data, location_data):
    X_train, ids, labels, sc = create_simple_trainset(embed_data, id_data, location_data)
    print(f'X_train: {X_train.shape}, labels: {labels.shape}')
    model, eps, X_train, y_train =  train(X_train, labels, out_size=location_range , is_privacy_preserve=True, learning_rate=learning_rate, epochs=epochs)
    return (model, eps, X_train, y_train, sc)

def train_model(embed_data, id_data, date_data, location_data, user_selection: str = 'multi-input'):
    if date_data is None:
        return train_simple(embed_data, id_data, location_data)
    else:
        if user_selection == 'multi-output':
            return train_multi_output_model(embed_data, id_data, date_data, location_data)
        else:
            return train_multi_input_model(embed_data, id_data, date_data, location_data)


def estimate_cost() -> tuple[dict]:
    batch_size = 200
    noise_multiplier = 0.03
    delta = 1e-3
    eps1 = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=training_size,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta)
    
    eps2 = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=training_size,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta)
    
    estimated_acc1 = 0.25
    estimated_acc2 = 0.45
    return (
        {"eps": eps1, "acc": estimated_acc1},
        {"eps": eps2, "acc": estimated_acc2})
    
