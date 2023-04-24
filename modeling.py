import tensorflow as tf


class MySequentialModel(tf.keras.Model):
    def __init__(
        self,
        out_size: int,
        units: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(
            units=out_size, activation='softmax')

    def call(self, x):
        x = self.dense_1(x)
        # x = self.dense_2(x)
        return self.output_layer(x)


class CrossProductOutputModel(tf.keras.Model):
    def __init__(
        self,
        out_size: int,
        units: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(
            units=out_size, activation='sigmoid')

    def call(self, x):
        return self.dense_2(self.dense_1(x))
