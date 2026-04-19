import os, logging, sys 
import time
import zmq
import json, pickle
import numpy as np
from model import neural_network_model, train_the_model, predict_fn, load_baseline_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd 
import csv
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from collections import deque
import math

# =========================
# Node Configuration
# =========================

# Determine neighborhood: "physical" for real devices, "docker" for docker nodes.
network="192.168.1"
NEIGHBORHOOD = os.getenv("NEIGHBORHOOD", "physical").strip().lower()
area=None
SAVE_PATH = "/models" if os.getenv("NODE_ID") == "X" else "./"

LOG_DIR  = os.environ.get("LOG_DIR", "/app/logs")
LOG_FILE = os.path.join(LOG_DIR, f"{os.environ.get('NODE_ID','node')}.log")

os.makedirs(LOG_DIR, exist_ok=True) 

logger = logging.getLogger(__name__)
logger.info("Logger initialized; writing to %s", LOG_FILE)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s %(name)s %(levelname)s ─ %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
)

if NEIGHBORHOOD == "physical":
    NODE_LIST = ["A", "B", "C"]
    ip_mapping = {
        "A": 99,
        "B": 101,
        "C": 102,
        "D":103
    }
elif NEIGHBORHOOD == "docker":
    area = "region0"
    NODE_LIST = ["X", "Y", "Z", "W"]
    ip_mapping = {
        "X": 199,
        "Y": 200,
        "Z": 201,
        "W": 202
    }
    subzone_node = {
        "X": "0",
        "Y": "1",
        "Z": "2",
        "W": "3"
    }
elif NEIGHBORHOOD == "docker2":
    area = "region1"
    NODE_LIST = ["P", "Q", "R", 'L']
    ip_mapping = {
        "P": 204,
        "Q": 205,
        "R": 206,
        "L": 207
    }
    subzone_node = {
        "P": "0",
        "Q": "1",
        "R": "2",
        "L": "3"
    }
elif NEIGHBORHOOD == "docker3":
    area = "region2"
    NODE_LIST = ["A", "B", "C", "D"]
    ip_mapping = {
        "A": 208,
        "B": 209,
        "C": 210,
        "D": 211
    }
    subzone_node = {
        "A": "0",
        "B": "1",
        "C": "2",
        "D": "3"
    }

elif NEIGHBORHOOD == "docker4":
    area = "region3"
    NODE_LIST = ["E", "F", "G","H"]
    ip_mapping = {
        "E": 212,
        "F": 213,
        "G": 214,
        "H": 215
    }
    subzone_node = {
        "E": "0",
        "F": "1",
        "G": "2",
        "H": "3"
    }



leaders = {
    "region0":"X",
    "region1":"P",
    "region2": "A",
    "region3": "E"
}

    
NODE_ID = os.getenv("NODE_ID", NODE_LIST[0]).strip().upper()


NODE_ID = NODE_ID.upper()
node_index = NODE_LIST.index(NODE_ID)
PREV_NODE = NODE_LIST[node_index - 1] if node_index > 0 else NODE_LIST[-1]
NEXT_NODE = NODE_LIST[(node_index + 1) % len(NODE_LIST)]

PREV_NODE_IP = f"{network}.{ip_mapping[PREV_NODE]}"
NEXT_NODE_IP = f"{network}.{ip_mapping[NEXT_NODE]}"

# Coordinator IP is constant.
COORDINATOR_IP = f"{network}.203"  # Replace with your actual coordinator IP

# Communication parameters.
REQ_REP_PORT = 5555    # Port for neighbor-to-neighbor REQ-REP exchange.    
STARTING_SOCKET=5561
PEERS=5562
GLOBAL_UPDATE_PORT = 5560
TOTAL_ROUNDS = len(NODE_LIST)-1

BASE_WEATHERS = [
    'Clear','Clouds','Fog','Haze','Mist','Rain','Snow'
]

results_file = f"/results/node_results/{NODE_ID}_results.csv"
print(f"Neighborhood: {NEIGHBORHOOD}")
print(f"Current Node: {NODE_ID}")
print(f"Previous Node: {PREV_NODE} at IP {PREV_NODE_IP}")
print(f"Next Node: {NEXT_NODE} at IP {NEXT_NODE_IP}")
print(f"Coordinator IP: {COORDINATOR_IP}")


context = zmq.Context()

router= None
neigh_dealer=None
glob_dealer=None


if NODE_ID in leaders.values():
    router = context.socket(zmq.ROUTER)
    router.bind(f"tcp://*:{PEERS}")

else:
    neigh_dealer = context.socket(zmq.DEALER)
    neigh_dealer.setsockopt_string(zmq.IDENTITY,NODE_ID)
    neigh_dealer.connect(f"tcp://192.168.1.{ip_mapping[leaders[area]]}:{PEERS}")

glob_dealer=context.socket(zmq.DEALER)
glob_dealer.setsockopt_string(zmq.IDENTITY,NODE_ID)
glob_dealer.connect(f"tcp://{COORDINATOR_IP}:{STARTING_SOCKET}")



recv_socket = context.socket(zmq.PULL)
recv_socket.bind(f"tcp://0.0.0.0:{REQ_REP_PORT}")

send_socket = context.socket(zmq.PUSH)
send_socket.connect(f"tcp://{NEXT_NODE_IP}:{REQ_REP_PORT}")


class Welford:
    def __init__(self, eps: float = 1e-6):

        self.eps   = eps
        self.n     = 0
        self.dim   = None
        self.means = None
        self.M2    = None

    def update(self, x):
        arr = np.array(x, dtype=np.float64).reshape(-1)
        
        if self.n == 0:
            self.dim   = arr.shape[0]
            self.means = np.zeros(self.dim, dtype=np.float64)
            self.M2    = np.zeros(self.dim, dtype=np.float64)
        
        if arr.shape[0] != self.dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim}, got {arr.shape[0]}"
            )

        # Increment count and update each feature’s mean and M2
        self.n += 1
        delta = arr - self.means              # vector of length dim
        self.means += delta / self.n
        delta2 = arr - self.means
        self.M2 += delta * delta2

    def std(self):
        if self.n < 2:
            return np.full(self.dim, self.eps, dtype=np.float64)
        
        var = self.M2 / (self.n - 1)  # sample variance per feature
        sigma = np.sqrt(var)
        # floor at eps
        sigma[sigma < self.eps] = self.eps
        return sigma

    def get(self):

        if self.n == 0:
            return None, None
        return self.means.copy(), self.std()

    def reset(self, mean=0.0, std=None, init_count=0, hard=False):
        if hard:
            # Full reset: forget all stats
            self.n = 0
            self.dim = None
            self.means = None
            self.M2 = None
            return

        self.n = init_count
        mean = np.array(mean, dtype=np.float64).reshape(-1)
        self.dim = mean.shape[0]
        self.means = mean.copy()

        if std is not None and init_count > 1:
            std = np.array(std, dtype=np.float64).reshape(-1)
            self.M2 = (std ** 2) * (init_count - 1)
        else:
            self.M2 = np.zeros(self.dim, dtype=np.float64)

class RollingWelford:

    def __init__(self, window_size: int, eps: float = 1e-6):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.eps = eps

    def update(self, x: float):
        self.buffer.append(float(x))

    def get(self) -> tuple[float, float]:
 
        n = len(self.buffer)
        if n == 0:
            return 0.0, self.eps

        arr = np.array(self.buffer, dtype=np.float64)
        mean = float(arr.mean())
        if n < 2:
            return mean, self.eps

        sigma = float(arr.std(ddof=0))
        if sigma < self.eps:
            sigma = self.eps
        return mean, sigma

    def reset(self):
        self.buffer.clear()

# =========================
# Helper Functions
# =========================

def get_numeric_cols(input):
    exclude_cols = ['week_of_year','hour_sin','hour_cos',"day_of_week",'month','weather_main','bike_usage_next']

    numeric_columns=input.select_dtypes(include=['number']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    print(numeric_columns)

    return numeric_columns


def online_pre_processing(
    row: pd.Series,
    input_cols: list[str],
    output_col: str,
    means_inputs: np.ndarray,
    sigmas_inputs: np.ndarray,
    mean_output: float,
    sigma_output: float,
) -> tuple[np.ndarray, float]:


    X_raw = row[input_cols].to_numpy(dtype=np.float32)   # shape = (n_inputs,)


    eps = 1e-6
    sig_in = sigmas_inputs.astype(np.float32).copy()
    sig_in[sig_in == 0] = eps
    means_in = means_inputs.astype(np.float32)
    X_z_scored = (X_raw - means_in) / sig_in   # shape = (n_inputs,)


    time_feats = row[["hour_sin", "hour_cos"]].to_numpy(dtype=np.float32)  


    categ_data = one_hot_encoding(row)          
    categ_data = categ_data.reshape(-1).astype(np.float32)  

 
    X_1d = np.concatenate([time_feats, X_z_scored, categ_data], axis=0).astype(np.float32)
    X_combined = X_1d.reshape(1, -1) 


    y_raw = np.float32(row[output_col])
    sig_out = np.float32(sigma_output if sigma_output != 0 else eps)
    y_z = np.float32((y_raw - mean_output) / sig_out)

    return X_combined, y_z


def data_init():
    CSV_DIR = "/data/2024_csvs"
    weather_df=pd.read_csv(f"{CSV_DIR}/fedJan_2024.csv")
    file=pd.read_csv(f"{CSV_DIR}/{area}_subzone{subzone_node[NODE_ID]}_bike_usage.csv")

    default_values = {
        'visibility': 10000,
        'wind_gust': 0,
        'rain_1h': 0,
        'rain_3h': 0,
        'snow_1h': 0,
        'snow_3h': 0,
    }

    weather_df[['visibility','wind_gust','rain_1h', 'rain_3h', 'snow_1h', 'snow_3h']] = weather_df[['visibility','wind_gust','rain_1h', 'rain_3h', 'snow_1h', 'snow_3h']].fillna(default_values)

    weather_df["hour"] = pd.to_datetime(weather_df["hour"], errors="coerce")
    file["hour"] = pd.to_datetime(weather_df["hour"], errors="coerce")

    merged_df=pd.merge(weather_df,file,on="hour",how="left")
    merged_df["bike_usage"] = merged_df["bike_usage"].fillna(0)
    
    merged_df["hour"] = pd.to_datetime(merged_df["hour"], errors="coerce")
    print(merged_df["hour"])
    merged_df['bike_usage_next'] = merged_df['bike_usage'].shift(-1)
    merged_df = merged_df.iloc[:-1].copy()

    merged_df["hour_of_datetime"]=merged_df["hour"].dt.hour
    print(merged_df["hour_of_datetime"])
    merged_df["day_of_week"] = merged_df["hour"].dt.dayofweek
    merged_df['week_of_year'] = merged_df['hour'].dt.isocalendar().week
    merged_df["month"] = merged_df["hour"].dt.month
    merged_df["date"] = merged_df["hour"].dt.date
    
    #print(months)

    merged_df["hour_sin"]=np.sin(2*np.pi*merged_df["hour_of_datetime"]/24)
    merged_df["hour_cos"]=np.cos(2*np.pi*merged_df["hour_of_datetime"]/24)

    columns=['week_of_year','date','hour','hour_sin','hour_cos','day_of_week','month','temp','visibility','dew_point','feels_like','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg',"wind_gust",'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h','clouds_all','weather_main',"bike_usage_next"]
    
    merged_df=merged_df[columns]
    return merged_df



def pre_processing(input,numeric_columns,mean,sigma):

    time_feats = input[['hour_sin','hour_cos']].to_numpy(dtype=np.float32)
     

    z_scored_data=z_score_normal(input,numeric_columns)
    print(f"The shape of the z_score:{z_scored_data.shape[1]}")
    
    output=z_score(mean,["bike_usage_next"],sigma,input)
    
    categ_data=one_hot_encoding(input).astype(np.float32)
    print(f"The shape of the categ:{categ_data.shape[1]}")

    filtered_input=np.concatenate([time_feats,z_scored_data,categ_data], axis=1).astype(np.float32)

    return filtered_input,output


def one_hot_encoding(data):

    if isinstance(data, pd.Series):
        df = data.to_frame().T.reset_index(drop=True)
    elif isinstance(data, pd.DataFrame):
        df = data.copy().reset_index(drop=True)
    else:
        raise ValueError("one_hot_encoding expects a pandas Series or DataFrame")

    # 1) Merge "Fog" into "Mist" so that weather_main levels match exactly
    df["weather_main"] = df["weather_main"].replace("Fog", "Mist")

    # 2) Define the fixed categories that the baseline used:
    ALL_DAYS     = list(range(7))                # 0..6
    ALL_MONTHS   = list(range(1, 13))            # 1..12
    BASE_WEATHERS = [
        "Clear", "Clouds", "Haze", "Mist",
        "Rain", "Snow", "Thunderstorm"
    ]

    # 3) Cast to pd.Categorical with those exact levels (even if some levels don't appear in df)
    df["day_of_week"]  = pd.Categorical(df["day_of_week"],  categories=ALL_DAYS)
    df["month"]        = pd.Categorical(df["month"],        categories=ALL_MONTHS)
    df["weather_main"] = pd.Categorical(df["weather_main"], categories=BASE_WEATHERS)

    # 4) Now one-hot encode exactly those three columns
    dummies = pd.get_dummies(
        df[["day_of_week", "month", "weather_main"]],
        prefix=["dow", "mon", "weather"],
        dtype=np.float32,
        drop_first=False
    )

    return dummies.to_numpy().astype(np.float32)


def z_score_normal(panda,columns):
    scaler = StandardScaler()  
    
    zscored_data = scaler.fit_transform(panda[columns])
 
    return zscored_data


def z_score(mean: float, column: str, sigma: float, df) -> np.ndarray:

    arr = df[column].to_numpy()        
    return (arr - mean) / sigma 

def compute_z_score_values(output: pd.Series) -> dict:
    # output is a pd.Series of shape (n_samples,)
    arr = output.to_numpy()          # 1-D array, shape (n_samples,)
    n   = arr.shape[0]

    mu      = float(np.mean(arr))
    variance= float(np.var(arr, ddof=0))  # population variance

    return {
        "number_of_samples": n,
        "mean":               mu,
        "variance":           variance
    }

def send_z_score(output, dealer):
    # 1) compute local stats
    stats = compute_z_score_values(output)
    print("→ computed stats:", stats)

    # 2) send them as JSON
    dealer.send_json(stats)

    # 3) receive the ACK
    ack = dealer.recv_string()           # blocks until b"ACK" arrives
    print("→ coordinator ACK’d stats:", ack)

    # 4) receive the μ/σ payload
    raw = dealer.recv()                  # blocks until the JSON frame arrives
    resp = json.loads(raw.decode("utf-8"))
    print("→ received z‐score response:", resp)

    return float(resp["mu"]), float(resp["sigma"])



def serialize_weights(weights):
    return pickle.dumps(weights)

def deserialize_weights(data):
    return pickle.loads(data)

def serialize_weights_json(weights):
    return json.dumps([arr.tolist() for arr in weights])

def deserialize_weights_json(msg):
    data = json.loads(msg)
    return [np.array(chunk) for chunk in data]


def federated_average(*weight_sets):

    if len(weight_sets) == 1 and isinstance(weight_sets[0], list) and \
       all(isinstance(ws, list) for ws in weight_sets[0]):
        weight_sets = tuple(weight_sets[0])

    num_clients = len(weight_sets)
    layers = zip(*weight_sets)
    return [sum(layer) / num_clients for layer in layers]


def send_weights(weights):
    print(f"[{NODE_ID}] Sending weights to {NEXT_NODE}")
    send_socket.send(pickle.dumps(weights))  # No need to wait for ACK

def receive_weights():
    print(f"[{NODE_ID}] Waiting for weights from {PREV_NODE}")
    weights_bytes = recv_socket.recv()  # Will block until message received
    print(f"[{NODE_ID}] Received weights from {PREV_NODE}")
    return pickle.loads(weights_bytes)

def send_update_and_wait_for_peers(weights):

    dealer = context.socket(zmq.DEALER)
    dealer.identity = NODE_ID.encode("utf-8")
    dealer.connect(f"tcp://{COORDINATOR_IP}:{GLOBAL_UPDATE_PORT}")

    # 1) Send our own update under our AREA
    payload = {
        "area": area,                      # make sure AREA is defined globally
        "msg":   serialize_weights_json(weights)
    }
    dealer.send_multipart([
        b"",   
        json.dumps(payload).encode("utf-8")
    ])
    print(f"[{NODE_ID}] Sent my weights for area={area!r} → coordinator")

    # 2) Wait for the coordinator’s bundled reply
    _, raw = dealer.recv_multipart()
    reply = json.loads(raw.decode("utf-8"))
    dealer.close()

    # 3) Unpack all the other areas’ updates
    other_updates = {}
    for peer_area, ser_w in reply.get("msgs", {}).items():
        other_updates[peer_area] = deserialize_weights_json(ser_w)
        print(f"[{NODE_ID}] Received update for peer area={peer_area!r}")

    return other_updates

def split_by_week_chrono(df, week_col='week_of_year', test_frac=0.2, time_col='timestamp'):

    train_parts = []
    test_parts = []
    
    for _, date_df in df.groupby(week_col):
        # make sure it’s sorted in time
        date_df = date_df.sort_values(time_col)
        n = len(date_df)
        split_idx = int(n * (1 - test_frac))
        
        train_parts.append(date_df.iloc[:split_idx])
        test_parts.append(date_df.iloc[split_idx:])
    
    train = pd.concat(train_parts).reset_index(drop=True)
    test  = pd.concat(test_parts).reset_index(drop=True)
    return train, test

def split_train_val_chrono(df, val_frac=0.2, time_col='timestamp'):

    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - val_frac))
    train = df.iloc[:split_idx].reset_index(drop=True)
    val   = df.iloc[split_idx:].reset_index(drop=True)
    return train, val


def ring_all_reduce(my_weights):
    # Start with own weights
    accumulated_weights = [np.copy(w) for w in my_weights]

    for _ in range(TOTAL_ROUNDS):
        # Send current weights to the next node
        send_weights(my_weights)

        # Receive neighbor's weights from the previous node
        neighbor_weights = receive_weights()

        # Accumulate weights
        for i in range(len(accumulated_weights)):
            accumulated_weights[i] += neighbor_weights[i]

        # Set my_weights to received neighbor weights to forward in next round
        my_weights = neighbor_weights

    # Average after full loop
    averaged_weights = [w / len(NODE_LIST) for w in accumulated_weights]
    return averaged_weights
    

# =========================
# Main Federated Learning Process
# =========================

def main():
    print("Started main")
    # 1) Load raw data
    merged_df = data_init()
    numeric_cols=get_numeric_cols(merged_df)
    input_mean=0
    input_sigma=0

    time.sleep(2)

    # 2) Load and compile baseline, then build a fresh local model
    baseline = load_baseline_model()
    baseline.compile(
        optimizer="nadam",
        loss="mse",
        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(name="mse"),
            tf.keras.metrics.R2Score(name="r2")
        ]
    )

    model, early_stopping, reduce_lr = neural_network_model()

    full_df = merged_df.sort_values("hour").reset_index(drop=True)

    input_stats=Welford()
    output_stats=Welford()

    #window_size = 504
    #rolling_trackers = {
       # col: RollingWelford(window_size=window_size, eps=1e-6)
        #for col in numeric_cols
    #}

    #target_tracker = RollingWelford(window_size=window_size, eps=1e-6)

    # 5) Buffers for each week
    current_date = None
    date_buffer = []  # collect raw rows for weekly training
    # buffers for predictions & truths:
    date_true = []         # actual bike_usage_next
    date_base_pred = []    # baseline predictions
    date_fed_pred = []     # federated model predictions

    # 6) Path for writing results
    results_file = f"/results/node_results/{NODE_ID}_results.csv"
    HEADER = [
        "Date",
        "base_mae", "base_mse",
        "fed_mae",  "fed_mse"
    ]
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

    # 7) MAIN LOOP: 
    for idx, row in full_df.iterrows():
        date = row["date"]

        # Initialize current_date on the very first row
        if current_date is None:
            current_date = date
            print(f"\n=== Starting date {current_date} ===\n")
        #elif current_date == 4:
            #for col in numeric_cols:
              # rolling_trackers[col].reset()
            #target_tracker.reset()


        if date != current_date:
            print(f"\n=== date {current_date} ended. Computing 24 hour metrics… ===")

            # 7a.1) Compute baseline vs. federated metrics for the past week
            y_true_arr = np.array(date_true)
            yb_arr = np.array(date_base_pred)
            yf_arr = np.array(date_fed_pred)

            if len(y_true_arr) > 0:
                base_mae  = mean_absolute_error(y_true_arr, yb_arr)
                base_mse = mean_squared_error(y_true_arr, yb_arr)

                fed_mae   = mean_absolute_error(y_true_arr, yf_arr)
                fed_mse  = mean_squared_error(y_true_arr, yf_arr)

                # Append to CSV
                with open(results_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_date,
                        base_mae, base_mse,
                        fed_mae,  fed_mse
                    ])
                print(f"[{NODE_ID}] Logged  metrics for date {current_date} → {results_file}")
            else:
                print(f"[{NODE_ID}] WARNING: No data collected for date {current_date}. Skipping metrics write.")


            date_df = pd.DataFrame(date_buffer)
            if not date_df.empty:
                print(f"[{NODE_ID}] Running 24 hour training/federation on date {current_date} data (n={len(date_df)})…")
                # 1) Compute “true” μ/σ via your aggregator
                output = date_df["bike_usage_next"]
                global_mean, global_sigma = send_z_score(output, glob_dealer)
                input_mean,input_sigma=input_stats.get()

                # 2) Chronological split → train & val
                tr_df, val_df = split_train_val_chrono(
                    date_df, val_frac=0.2, time_col='hour'
                )

                # 3) Preprocess using that week's μ/σ
                X_tr, y_tr = pre_processing(tr_df,numeric_cols, global_mean, global_sigma)
                X_val, y_val = pre_processing(val_df,numeric_cols, global_mean, global_sigma)

                # 4) Train your local model
                train_the_model(
                    model,
                    X_val, y_val,
                    early_stopping, reduce_lr,
                    X_tr, y_tr,
                    100
                )
                my_weights = model.get_weights()

                my_weights=ring_all_reduce(my_weights)

                # 6) Inter‐neighborhood exchange (only leaders drive coordinator)
                if NODE_ID in leaders.values():
                    received_weights= send_update_and_wait_for_peers(my_weights)
                    all_weights = [my_weights] + list(received_weights.values())
                    flat_list = np.concatenate([w.ravel() for w in all_weights[0]])
                    print(f"[DEBUG] local flatten before federated_average: ‖flat‖={np.linalg.norm(flat_list):.6f}")
                    my_weights = federated_average(all_weights)
                    
                    model.set_weights(my_weights)
                    print(f"[{NODE_ID}] Completed synchronous neighborhood exchange")

                    for peer in NODE_LIST:
                        if peer == NODE_ID:
                            continue

                        payload = {
                            "weights": [w.tolist() for w in my_weights]
                        }
                        # send [peer_identity, empty, json_payload]
                        router.send_multipart([
                            peer.encode("utf-8"),
                            b"",
                            json.dumps(payload).encode("utf-8")
                        ])
                        print(f"[{NODE_ID}] Sent global update to {peer}")

                        # 3) Wait for that peer’s ACK
                        _id, _empty, raw = router.recv_multipart()
                        ack = raw.decode("utf-8")
                        print(f"[{NODE_ID}] Received ACK from {_id.decode()}: {ack}")

                else:
                    print(f"[{NODE_ID}] Waiting for leader’s global update (date {current_date})…")
                    frames = neigh_dealer.recv_multipart()
                    raw = frames[-1]
                    msg = json.loads(raw.decode("utf-8"))
                    peer_weights = [np.array(w) for w in msg["weights"]]
                    model.set_weights(peer_weights)
                    print(f"[{NODE_ID}] Received global update from leader for date {current_date}.")
                    ack = json.dumps({"status": "ACK"}).encode("utf-8")
                    neigh_dealer.send_multipart([b"", ack])
                    print(f"[{NODE_ID}] Sent ACK to leader.")

                date_buffer.clear()
                date_true.clear()
                date_base_pred.clear()
                date_fed_pred.clear()

                init_count = len(date_df)
                input_stats.reset(hard=True)
                #input_stats.reset(input_mean,input_sigma,init_count)
                output_stats.reset(global_mean,global_sigma,init_count)
                #for col in numeric_cols:
                # rolling_trackers[col].reset()
                #target_tracker.reset()
                date_buffer.clear()

            else:
                print(f"[{NODE_ID}] WARNING: date_buffer is empty for date {current_date}. Skipping training/federation.")



            current_date = date
            print(f"\n=== Starting date {current_date} ===\n")

        row_df = pd.DataFrame([row])
        input_array=row[numeric_cols].to_numpy(dtype=np.float32)
        output_array=np.float32(row[["bike_usage_next"]])

        input_stats.update(input_array)
        output_stats.update(output_array)

        input_means,input_sigmas=input_stats.get()
        output_mean,output_sigma=output_stats.get()

        input,out=online_pre_processing(row,numeric_cols,"bike_usage_next",input_means,input_sigmas,output_mean,output_sigma)


        yb = baseline.predict(input)[0][0]

        yf = model.predict(input)[0][0]

        date_true.append(out)
        date_base_pred.append(yb)
        date_fed_pred.append(yf)

        true_val = out.item()  # safe conversion
        print(f"[{NODE_ID}] Hour {row_df['hour']} Date:{date},True={true_val:.4f}, BasePred={yb:.2f}, FedPred={yf:.2f}")

        #  5) Buffer the row so it’s included in the date training
        date_buffer.append(row)

    if date_buffer:
        print(f"\n=== Final date {current_date} ended. Computing final date metrics… ===")
        y_true_arr = np.array(date_true)
        yb_arr = np.array(date_base_pred)
        yf_arr = np.array(date_fed_pred)

        if len(y_true_arr) > 0:
            base_mae  = mean_absolute_error(y_true_arr, yb_arr)
            base_mse = mean_squared_error(y_true_arr, yb_arr)
            base_r2   = r2_score(y_true_arr, yb_arr)

            fed_mae   = mean_absolute_error(y_true_arr, yf_arr)
            fed_mse  = mean_squared_error(y_true_arr, yf_arr)
            fed_r2    = r2_score(y_true_arr, yf_arr)

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_date,
                    base_mae, base_mse,
                    fed_mae,  fed_mse
                ])
            print(f"[{NODE_ID}] Logged final 24 hour metrics for date {current_date} → {results_file}")
        else:
            print(f"[{NODE_ID}] WARNING: No data collected for final date {current_date}, skipping metrics.")

        # Run the training & federation pipeline one last time
        date_df = pd.DataFrame(date_buffer)
        if not date_df.empty:
            print(f"[{NODE_ID}] Running final training/federation on date {current_date} (n={len(date_df)})…")
            output = date_df["bike_usage_next"]
            global_mean, global_sigma = send_z_score(output, glob_dealer)

            tr_df, val_df = split_train_val_chrono(
                date_df, val_frac=0.2, time_col='hour'
            )
            X_tr, y_tr = pre_processing(tr_df,numeric_cols,global_mean, global_sigma)
            X_val, y_val = pre_processing(val_df,numeric_cols, global_mean, global_sigma)

            train_the_model(
                model, X_val, y_val,
                early_stopping, reduce_lr,
                X_tr, y_tr,100
            )
            my_weights = model.get_weights()

            my_weights=ring_all_reduce(my_weights)

            if NODE_ID in leaders.values():
                received_weights= send_update_and_wait_for_peers(my_weights)
                all_weights = [my_weights] + list(received_weights.values())
                my_weights = federated_average(all_weights)
                
                model.set_weights(my_weights)
                print(f"[{NODE_ID}] Completed synchronous neighborhood exchange")

                for peer in NODE_LIST:
                    if peer == NODE_ID:
                        continue

                    payload = {
                        "weights": [w.tolist() for w in my_weights]
                    }
                    # send [peer_identity, empty, json_payload]
                    router.send_multipart([
                        peer.encode("utf-8"),
                        b"",
                        json.dumps(payload).encode("utf-8")
                    ])
                    print(f"[{NODE_ID}] Sent global update to {peer}")

                    # 3) Wait for that peer’s ACK
                    _id, _empty, raw = router.recv_multipart()
                    ack = raw.decode("utf-8")
                    print(f"[{NODE_ID}] Received ACK from {_id.decode()}: {ack}")

            else:
                print(f"[{NODE_ID}] Waiting for leader’s global update (final date)…")
                frames = neigh_dealer.recv_multipart()
                raw = frames[-1]
                msg = json.loads(raw.decode("utf-8"))
                peer_weights = [np.array(w) for w in msg["weights"]]
                model.set_weights(peer_weights)
                print(f"[{NODE_ID}] Received final global update from leader.")
                ack = json.dumps({"status": "ACK"}).encode("utf-8")
                neigh_dealer.send_multipart([b"", ack])
                print(f"[{NODE_ID}] Sent ACK to leader for final date {current_date}.")
        else:
            print(f"[{NODE_ID}] WARNING: final date_buffer empty, skipping final training/federation.")

    print("\n=== All done ===")



if __name__ == "__main__":
    main()
