import tensorflow as tf


def load_baseline_model():
    return tf.keras.models.load_model(
        "/data/2024_csvs/_model_two_layers.keras", compile=False
    )


def neural_network_model():

    l2 = tf.keras.regularizers.L2(0.0001)
    loss_func = tf.keras.losses.MeanSquaredError(name="MSE")
    """"
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  # Track validation loss
    patience=10,
    min_delta=0.001,     # Require at least 0.001 improvement        
    restore_best_weights=True  # Revert to best model weights
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_mae',       
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
    )
    """

    metrics = {
        "mae": "mae",
        "MSE": tf.keras.metrics.MeanSquaredError(name="MSE"),
        "RMSE": tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    }

    model = load_baseline_model()

    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=[metrics["mae"], metrics["RMSE"], tf.keras.metrics.R2Score()],
    )

    return model


def train_the_model(model, X_val, y_val, input, output, local_epochs):

    model.fit(
        input,
        output,
        validation_data=(X_val, y_val),
        epochs=local_epochs,
        batch_size=16,
        verbose=0,
    )


@tf.function(experimental_relax_shapes=True, reduce_retracing=True)
def predict_fn(model, x):
    return model(x, training=False)
