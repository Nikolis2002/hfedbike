"""
Baseline training + cross-validation worker.

Despite the filename, this script does more than preprocessing: it is the
inner loop of the baseline grid search. `fine_tuning.py` invokes it as a
subprocess with different --optimizer, --lr, --hidden_layers, --r (L2),
etc. Each invocation:

  1. Loads the pre-cleaned weather CSV and the per-station bike-usage
     CSVs, merges and engineers features (data_processor()).
  2. Builds a feedforward network with the passed hyperparameters
     (neural_network_model()).
  3. Runs month-stratified 5-fold CV (K_fold()) or a chronological month
     split (month_split()).
  4. Writes the fold-average MAE / MSE / RMSE / R2 plus metadata to the
     MongoDB collection `citibike.resultsv2`, keyed by `run_id`.

The winning configuration across all runs is later retrained in
best_baseline_model.py and saved as data/2024/_model_two_layers.keras.

External dependencies: TensorFlow, scikit-learn, MongoDB at
mongodb://localhost:27017/, numba (GPU reset between folds).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import argparse
from pymongo import MongoClient
import os, pprint
from datetime import datetime
from tensorflow.keras import backend as K
from collections import Counter

# ---------------------------------------------------------------------
# Paths are resolved relative to this file, not cwd, so the script runs
# from anywhere. Repo root is three levels up: pre_processing.py ->
# model_search -> baseline -> repo root.
# ---------------------------------------------------------------------
REPO_ROOT       = Path(__file__).resolve().parents[2]
SCRIPT_DIR      = Path(__file__).resolve().parent
WEATHER_CSV     = REPO_ROOT / "data" / "processed" / "clean_weather.csv"
MERGED_CACHE    = SCRIPT_DIR / "merged_data.csv"          # feature-engineered cache
CHECKPOINT_PATH = SCRIPT_DIR / "_model_two_layers.keras"  # ModelCheckpoint output
SCREENSHOTS_DIR = SCRIPT_DIR / "screenshotsv2"            # run-plot outputs

client = MongoClient("mongodb://localhost:27017/")
database = client["citibike"]
average_results = database["resultsv2"]


gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def z_score(panda, columns):
    scaler = StandardScaler()

    zscored_data = scaler.fit_transform(panda[columns])
    sigma = scaler.scale_[0]
    print(f"This is the sigma: {sigma}")

    return zscored_data, sigma


def one_hot_encoding(panda, columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoder.fit(panda[columns])
    encoded_data = encoder.transform(panda[columns])

    return encoded_data


def _build_merged_frame():
    """Merge weather + hourly bike usage, add cyclic time features.

    Runs the MongoDB aggregation against ``citibike.bike_data_enriched``
    (see ``baseline/data_preparation/make_neighbs_2023/data_enrich.py``
    for how that collection is populated), joins with the cleaned
    OpenWeatherMap table at ``data/processed/clean_weather.csv``, adds
    cyclic/categorical time features, and shifts bike_usage by -1 to
    create the ``bike_usage_next`` target column.

    The result is cached to ``MERGED_CACHE`` so subsequent grid-search
    invocations skip this step.
    """
    # --- 1) cleaned hourly weather --------------------------------------
    weather_df = pd.read_csv(WEATHER_CSV)
    default_values = {
        "visibility": 10000, "wind_gust": 0,
        "rain_1h": 0, "rain_3h": 0,
        "snow_1h": 0, "snow_3h": 0,
    }
    weather_cols = list(default_values.keys())
    weather_df[weather_cols] = weather_df[weather_cols].fillna(default_values)
    weather_df["hour"] = pd.to_datetime(weather_df["hour"], errors="coerce")

    # --- 2) hourly trip counts across all of NYC ------------------------
    col = client["citibike"]["bike_data_enriched"]
    pipeline = [
        {"$group": {"_id": "$hour", "bike_usage": {"$sum": 1}}},
        {"$project": {"_id": 0, "hour": "$_id", "bike_usage": 1}},
        {"$sort": {"hour": 1}},
    ]
    bike_df = pd.DataFrame(list(col.aggregate(pipeline)))
    bike_df["hour"] = pd.to_datetime(bike_df["hour"])

    # --- 3) weather ⋈ usage, shift-1 target -----------------------------
    merged_df = pd.merge(weather_df, bike_df, on="hour", how="inner")
    merged_df["hour"] = pd.to_datetime(merged_df["hour"], errors="coerce")
    merged_df["bike_usage_next"] = merged_df["bike_usage"].shift(-1)
    # Last row has no "next" target, drop it.
    merged_df = merged_df.iloc[:-1].copy()

    # --- 4) time features ----------------------------------------------
    merged_df["hour_of_datetime"] = merged_df["hour"].dt.hour
    merged_df["day_of_week"]      = merged_df["hour"].dt.dayofweek
    merged_df["week_of_year"]     = merged_df["hour"].dt.isocalendar().week
    merged_df["month"]            = merged_df["hour"].dt.month

    weeks  = merged_df["week_of_year"].to_numpy()
    angles = 2 * np.pi * (weeks - 1) / 52
    merged_df["week_sin"] = np.sin(angles)
    merged_df["week_cos"] = np.cos(angles)

    merged_df["hour_sin"] = np.sin(2 * np.pi * merged_df["hour_of_datetime"] / 24)
    merged_df["hour_cos"] = np.cos(2 * np.pi * merged_df["hour_of_datetime"] / 24)

    merged_df.to_csv(MERGED_CACHE, index=False)
    return merged_df


def data_processor():
    # Regenerating the merged frame requires MongoDB and is slow, so we
    # cache it on disk after the first build.  Delete MERGED_CACHE to
    # force a rebuild (e.g. after re-ingesting raw trips).
    if MERGED_CACHE.exists():
        merged_df = pd.read_csv(MERGED_CACHE)
    else:
        merged_df = _build_merged_frame()

    months = merged_df["month"].values

    columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "month",
        "temp",
        "visibility",
        "dew_point",
        "feels_like",
        "temp_min",
        "temp_max",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_deg",
        "wind_gust",
        "rain_1h",
        "rain_3h",
        "snow_1h",
        "snow_3h",
        "clouds_all",
        "weather_main",
    ]
    input = merged_df[columns]

    # Cyclic hour features go in directly (no normalization).
    time_feats = input[["hour_sin", "hour_cos"]].to_numpy(dtype=np.float32)

    # Split the rest into numeric (z-scored) and categorical (one-hot).
    exclude_cols    = ["hour_sin", "hour_cos", "day_of_week", "month", "weather_main"]
    numeric_columns = [c for c in input.select_dtypes(include=["number"]).columns
                       if c not in exclude_cols]
    categorical_columns = ["day_of_week", "month", "weather_main"]
    print("Numeric columns: ", numeric_columns)

    z_scored_data, _ = z_score(input, numeric_columns)
    output, sigma    = z_score(merged_df, ["bike_usage_next"])
    categ_data       = one_hot_encoding(input, categorical_columns).astype(np.float32)

    filtered_input = np.concatenate(
        [time_feats, z_scored_data, categ_data], axis=1
    ).astype(np.float32)

    return filtered_input, output, sigma, months


def create_parser():
    # argument parser to be able to test the training process with diffrent variables
    parser = argparse.ArgumentParser(description="options for Neural Network")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Type the optimizer for weights you want to use options:adam,SGD",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The learning rate for training"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.0, help="THe momentum for the SGD"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train"
    )
    parser.add_argument(
        "--num_of_layers",
        type=str,
        default="double",
        help="The number of hidden layers options: I/2,2*I/3,I,2*I",
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        default="MSE",
        help="The loss function options: cross entropy,MSE",
    )
    parser.add_argument(
        "--hid_layer_func",
        type=str,
        default="Relu",
        help="Activation function for hidden layers options:Relu,Tanh,Silu",
    )
    parser.add_argument("--r", type=float, default=None, help="Regulazation factor")
    parser.add_argument(
        "--normal",
        type=bool,
        default=True,
        help="Normal training you pass ALL the paramaters",
    )
    parser.add_argument(
        "--compare_losses",
        type=bool,
        default=False,
        help="Toggle it to true if you want to see the evaluation losses seperatly",
    )
    parser.add_argument(
        "--more_layers",
        type=bool,
        default=False,
        help="Test the network with more layers",
    )
    parser.add_argument("--use_l2", type=bool, default=False, help="Use L2")
    parser.add_argument("--use_l1", type=bool, default=False, help="Use L1")
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="",
        help="Comma-separated list of hidden layer sizes, e.g., '64,32' or '128,64,32'",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0, help="The dropout_rate"
    )

    parser.add_argument(
        "--run_id",
        type=int,
        default=0,
        help="Unique identifier for this experiment run",
    )
    parser.add_argument("--b_size", type=int, default=32, help="batch size")

    args = parser.parse_args()

    return args


def create_folder(id):
    """Create a timestamped output folder under SCREENSHOTS_DIR."""
    date_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    folder = SCREENSHOTS_DIR / f"ID_{id}_DATE_{date_str}"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


# plot the taining-loss/validation loss
def plot(args, loss_table, val_loss_table, folder, max_epochs):
    if args.compare_losses == True:
        for i, history in enumerate(val_loss_table):
            epochs = range(1, len(history) + 1)
            plt.plot(epochs, history, label=f"Fold {i + 1}")

        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss per Fold with Stopping Epochs")
        plt.legend()
        filename = os.path.join(folder, f"Plot.png")
        plt.savefig(filename, format="png")
        plt.show()
    else:
        # Determine the maximum number of epochs (or use args.epochs)

        print(f"This is the max epochs:{max_epochs}")

        # Pad each fold's validation loss history if it stopped early
        padded_val_histories = []
        padded_train_histories = []
        for train_history, val_history in zip(loss_table, val_loss_table):
            if len(train_history) < max_epochs:
                pad_length = max_epochs - len(train_history)
                padded_train = train_history + [train_history[-1]] * pad_length
            else:
                padded_train = train_history[:max_epochs]

            if len(val_history) < max_epochs:
                pad_length = max_epochs - len(val_history)
                padded_val = val_history + [val_history[-1]] * pad_length
            else:
                padded_val = val_history[:max_epochs]

            padded_train_histories.append(padded_train)
            padded_val_histories.append(padded_val)

        # Compute the average validation loss at each epoch across folds
        avg_loss = np.mean(padded_train_histories, axis=0)
        avg_val_loss = np.mean(padded_val_histories, axis=0)
        epochs = range(1, len(avg_loss) + 1)

        # Plot the average validation loss curve and save it in the created folder
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, avg_val_loss, label="Average Validation Loss", color="green")
        plt.plot(epochs, avg_loss, label="Average Training Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Average Validation and Training Loss Over 5-Fold CV")
        plt.legend()
        filename = os.path.join(folder, f"Plot.png")
        plt.savefig(filename, format="png")


def neural_network_model(
    input_shape,
    optimizer,
    momentum,
    lr,
    num_of_layers,
    hid_layer_func,
    loss_func,
    use_l2,
    use_l1,
    r=0.001,
    deep=False,
    deep_layers=None,
    dropout_rate_rate=0,
):

    hidden_layers = {
        "half": math.ceil(
            input_shape / 2
        ),  # diffrent choices for the neuron of the hidden layers all viable
        "two thirds": math.ceil((2 * input_shape) / 3),
        "same": input_shape,
        "double": 2 * input_shape,
    }

    activation_options = {
        "Relu": "relu",  # activation options of hidden layer
        "Tanh": "tanh",
        "Silu": tf.nn.silu,
    }
    l = None
    if use_l2 == False and use_l1 == False:
        l = None
    elif use_l2 == True:
        l = tf.keras.regularizers.L2(r)  # L2 regulazation if you want
    elif use_l1 == True:
        l = tf.keras.regularizers.L1(r)
    else:
        raise ValueError("Unsupported option")

    options_loss = {
        "cross entropy": tf.keras.losses.BinaryCrossentropy(),  # the loss functions
        "MSE": tf.keras.losses.MeanSquaredError(name="MSE"),
    }

    metrics = {
        "mae": "mae",
        "MSE": tf.keras.metrics.MeanSquaredError(name="MSE"),
        "RMSE": tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    }  # the metrics of the nn

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",  # Track validation loss
        patience=10,
        min_delta=0.001,  # Require at least 0.001 improvement
        restore_best_weights=True,  # Revert to best model weights
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",  # or 'val_loss'
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_PATH),  # absolute path under baseline/model_search/
        monitor="val_mae",
        save_best_only=True,
        mode="min",
        verbose=1,
    )
    print(f"Running the model for:{hidden_layers[num_of_layers]} neurons")
    # the model itself, the number of output neurons is 1 because the patient has either alzheimers or not and using sigmoid as the activation champion we achieve the
    # the binary clissification
    if deep == False:
        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(input_shape,)),
                tf.keras.layers.Dense(
                    128,
                    activation=activation_options[hid_layer_func],
                    kernel_regularizer=l,
                ),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )
    else:
        print(f"Running for layers:{deep_layers}")
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_shape,)))

        for layer in deep_layers[1:]:
            model.add(
                tf.keras.layers.Dense(
                    layer,
                    activation=activation_options[hid_layer_func],
                    kernel_regularizer=l,
                )
            )

            if dropout_rate_rate > 0.0:
                model.add(tf.keras.layers.Dropout(dropout_rate_rate))

        model.add(tf.keras.layers.Dense(1, activation="linear"))

    # optimizer options
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum, nesterov=True
        )
    else:
        raise ValueError("Unsupported option")

    model.compile(
        optimizer=optimizer,
        loss=options_loss[loss_func],
        metrics=[metrics["mae"], metrics["RMSE"], tf.keras.metrics.R2Score()],
    )

    return model, early_stopping, reduce_lr, checkpoint


def month_split(filtered_input, output, months, args, folder, hidden_layers, sigma):
    """
    Train with 5‑folds stratified by month so each validation set
    contains a slice of every month.
    """
    # set up stratified 80/20 split by month label
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=44)

    round = 1
    evals = []
    val_loss_table = []
    loss_table = []
    early_stop_epochs = []
    batch_size = args.b_size
    print(f"This is the input shape:{filtered_input.shape[1]}")

    for train_idx, val_idx in sss.split(filtered_input, months):
        val_months = months[val_idx]
        print(
            f"Fold {round} — val month distribution:",
            {m: c for m, c in Counter(val_months).items()},
        )
        # slice into train/validation
        X_train, X_val = filtered_input[train_idx], filtered_input[val_idx]
        y_train, y_val = output[train_idx], output[val_idx]

        # build the model
        model, early_stop, reduce_lr, checkpoint = neural_network_model(
            filtered_input.shape[1],
            args.optimizer,
            args.momentum,
            args.lr,
            args.num_of_layers,
            args.hid_layer_func,
            args.loss_func,
            args.use_l2,
            args.use_l1,
            args.r,
            args.more_layers,
            hidden_layers,
            args.dropout_rate,
        )

        # train
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop, reduce_lr, checkpoint],
        )

        # record epochs and losses
        early_stop_epochs.append(len(history.history["loss"]))
        loss_table.append(history.history["loss"])
        val_loss_table.append(history.history["val_loss"])

        evaluation = model.evaluate(X_val, y_val, verbose=0)
        loss, mae, rmse, r2 = evaluation  # unpack the three values

        print(
            f"Round {round}: "
            f"Loss = {loss:.4f}, "
            f"MAE = {mae:.4f}, "
            f"Actual MAE = {mae * sigma:.4f}, "
            f"RMSE = {rmse:.4f}, "
            f"R2 = {r2:.4f}"
        )
        evals.append(evaluation)

        del model
        del history
        K.clear_session()
        round = round + 1

    print("Saved the best model")

    # plot and return
    max_epochs = max(early_stop_epochs)
    plot(args, loss_table, val_loss_table, folder, max_epochs)

    # write the results to mongodb for further analysis
    evals_np = np.array(evals)
    evals_json = {
        "_id": args.run_id,
        "use L2": args.use_l2,
        "use L1": args.use_l1,
        "splitting": "month",
        "sigma": sigma,
        "multiple layers": args.more_layers,  # ignore for 1 layer
        "choosen architecture": args.hidden_layers,  # ignore for 1 layer
        "chosen_weight": "double",
        "params": {
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "learning rate": args.lr,
            "epochs": args.epochs,
            "run_epochs": max_epochs,  # the epochs of the training, it can be less thean epochs because i have early stop
            "number of hidden layers": args.num_of_layers,
            "hidden layer activation function": args.hid_layer_func,
            "regulazation rate": args.r,
            "loss function": args.loss_func,
            "dropout_rate": args.dropout_rate,
        },
        "Average MSE": np.mean(evals_np[:, 0]),
        "Average MAE": np.mean(evals_np[:, 1]),
        "Average RMSE": np.mean(evals_np[:, 2]),
        "Average R2": np.mean(evals_np[:, 3]),
    }

    printer = pprint.PrettyPrinter(indent=4)
    print("\n")
    print("|--------FINAL RESULTS----------|")
    printer.pprint(evals_json)

    average_results.insert_one(evals_json)


def K_fold(filtered_input, output, args, folder, hidden_layers, sigma):
    file_fold_split = KFold(
        n_splits=5, shuffle=True, random_state=44
    )  # 5-cv fold with balanced output class data(StatifiedKFold does that)
    round = 1
    evals = []
    val_loss_table = []
    loss_table = []
    early_stop_epochs = []
    batch_size = args.b_size

    # for every split train the nn and evaluate it
    for training_idx, val_idx in file_fold_split.split(filtered_input, output):
        input_train, input_val = filtered_input[training_idx], filtered_input[val_idx]
        output_train, output_val = output[training_idx], output[val_idx]

        # neural_network_model returns 4 values (model, early_stop,
        # reduce_lr, checkpoint); the K_fold path just ignores the
        # checkpoint since we don't persist the best fold anywhere.
        model, early_stop, reduce_lr, _checkpoint = neural_network_model(
            filtered_input.shape[1],
            args.optimizer,
            args.momentum,
            args.lr,
            args.num_of_layers,
            args.hid_layer_func,
            args.loss_func,
            args.use_l2,
            args.use_l1,
            args.r,
            args.more_layers,
            hidden_layers,
            args.dropout_rate,
        )
        training = model.fit(
            input_train,
            output_train,
            validation_data=(input_val, output_val),
            epochs=args.epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop, reduce_lr],
        )

        stop_epoch = len(training.history["loss"])
        early_stop_epochs.append(stop_epoch)

        val_loss_table.append(training.history["val_loss"])
        loss_table.append(training.history["loss"])

        evaluation = model.evaluate(input_val, output_val, verbose=0)
        loss, mae, rmse, r2 = evaluation  # unpack the three values

        print(
            f"Round {round}: "
            f"Loss = {loss:.4f}, "
            f"MAE = {mae:.4f}, "
            f"RMSE = {rmse:.4f}, "
            f"R2 = {r2:.4f}"
        )
        evals.append(evaluation)

        del model
        del training
        K.clear_session()

    max_epochs = max(early_stop_epochs)
    plot(args, loss_table, val_loss_table, folder, max_epochs)

    # write the results to mongodb for further analysis
    evals_np = np.array(evals)
    evals_json = {
        "_id": args.run_id,
        "use L2": args.use_l2,
        "use L1": args.use_l1,
        "splitting": "Kfold",
        "sigma": sigma,
        "multiple layers": args.more_layers,  # ignore for 1 layer
        "choosen architecture": args.hidden_layers,  # ignore for 1 layer
        "chosen_weight": "double",
        "params": {
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "learning rate": args.lr,
            "epochs": args.epochs,
            "run_epochs": max_epochs,  # the epochs of the training, it can be less thean epochs because i have early stop
            "number of hidden layers": args.num_of_layers,
            "hidden layer activation function": args.hid_layer_func,
            "regulazation rate": args.r,
            "loss function": args.loss_func,
            "dropout_rate": args.dropout_rate,
        },
        "Average MSE": np.mean(evals_np[:, 0]),
        "Average MAE": np.mean(evals_np[:, 1]),
        "Average RMSE": np.mean(evals_np[:, 2]),
        "Average R2": np.mean(evals_np[:, 3]),
    }

    printer = pprint.PrettyPrinter(indent=4)
    print("\n")
    print("|--------FINAL RESULTS----------|")
    printer.pprint(evals_json)

    average_results.insert_one(evals_json)


if __name__ == "__main__":
    args = create_parser()
    input, output, sigma, months = data_processor()

    hidden_layers = []
    if (
        args.hidden_layers.strip()
    ):  # Check for non-empty string after stripping whitespace
        try:
            hidden_layers = [
                int(n.strip()) for n in args.hidden_layers.split(",") if n.strip()
            ]
            if not hidden_layers:
                print(
                    "Warning: --hidden_layers contained only commas/whitespace, using no hidden layers"
                )
            else:
                print(f"Using hidden layers: {hidden_layers}")
        except ValueError as e:
            print(
                f"Error: Invalid layer sizes in --hidden_layers '{args.hidden_layers}'. Use comma-separated integers."
            )
            raise
    else:
        print("ℹ No hidden layers specified - using default architecture")

    folder = create_folder(args.run_id)

    # input=input_list[0]
    # output=output_list[0]
    split = 1

    if split == 0:
        K_fold(input, output, args, folder, hidden_layers, sigma)
    else:
        month_split(input, output, months, args, folder, hidden_layers, sigma)
