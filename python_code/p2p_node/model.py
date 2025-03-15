import tensorflow as tf
import numpy as np

def build_and_train_model():
    input1 = np.array([[0], [1], [0], [1]], dtype=np.int32)
    input2 = np.array([[0], [1], [1], [0]], dtype=np.int32)
    labels = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [0, 0, 1]], dtype=np.float32)

    input_a = tf.keras.Input(shape=(1,), dtype='int32', name='fruit1')
    input_b = tf.keras.Input(shape=(1,), dtype='int32', name='fruit2')

    embedding_layer = tf.keras.layers.Embedding(input_dim=3, output_dim=4)
    emb_a = embedding_layer(input_a)
    emb_b = embedding_layer(input_b)

    flat_a = tf.keras.layers.Flatten()(emb_a)
    flat_b = tf.keras.layers.Flatten()(emb_b)

    combined = tf.keras.layers.Concatenate()([flat_a, flat_b])

    hidden = tf.keras.layers.Dense(8, activation='relu')(combined)
    output = tf.keras.layers.Dense(3, activation='softmax')(hidden)

    model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit({'fruit1': input1, 'fruit2': input2}, labels, epochs=30, verbose=0)

    return model
