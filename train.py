import tensorflow as tf
from modeling import MySequentialModel, CrossProductOutputModel
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def train(
    X_train, 
    y_enc,
    batch_size: int = 500,
    num_microbatches: int = 1,
    l2_norm_clip: float = 1,
    noise_multiplier: float = 2,
    learning_rate: float = 2,
    epochs: int = 200,
    loss_func: str = 'CategoricalCrossentropy'
):
    if batch_size % num_microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')
    
    # optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    #     l2_norm_clip=l2_norm_clip,
    #     noise_multiplier=noise_multiplier,
    #     num_microbatches=num_microbatches,
    #     learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Define model and loss function
    if loss_func == 'CategoricalCrossentropy':
        model = MySequentialModel(out_size=1)
        loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    elif loss_func == 'BinaryCrossentropy':
        model = CrossProductOutputModel(out_size=365 * 20)
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    else:
        raise ValueError
    
    # Fit model
    model.fit(X_train, y_enc, epochs=epochs, batch_size=batch_size)

    compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=X_train.shape[0],
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=1e-3)
    
    return model   
