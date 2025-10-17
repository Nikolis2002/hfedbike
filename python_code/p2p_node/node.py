import os, logging, sys 
import time
import zmq
import json
import numpy as np
from model import neural_network_model, train_the_model, predict_fn, load_baseline_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd 
import csv
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    area = "lower_west_manhattan"
    NODE_LIST = ["X", "Y", "Z", "W"]
    ip_mapping = {
        "X": 199,
        "Y": 200,
        "Z": 201,
        "W": 202
    }
    subzone_node = {
        "X": "NW",
        "Y": "NE",
        "Z": "SW",
        "W": "SE"
    }
elif NEIGHBORHOOD == "docker2":
    area = "upper_manhattan"
    NODE_LIST = ["P", "Q", "R", 'L']
    ip_mapping = {
        "P": 204,
        "Q": 205,
        "R": 206,
        "L": 207
    }
    subzone_node = {
        "P": "NW",
        "Q": "NE",
        "R": "SW",
        "L": "SW"
    }
elif NEIGHBORHOOD == "docker3":
    area = "east_of_manhattan"
    NODE_LIST = ["A", "B", "C", "D"]
    ip_mapping = {
        "A": 208,
        "B": 209,
        "C": 210,
        "D": 211
    }
    subzone_node = {
        "A": "NW",
        "B": "NE",
        "C": "SW",
        "D": "SE"
    }

elif NEIGHBORHOOD == "docker4":
    area = "south_brooklyn"
    NODE_LIST = ["E", "F", "G"]
    ip_mapping = {
        "E": 212,
        "F": 213,
        "G": 214
    }
    subzone_node = {
        "E": "NW",
        "F": "SW",
        "G": "SW"
    }



leaders = {
    "lower_west_manhattan":"X",
    "upper_manhattan":"P",
    "east_of_manhattan": "A",
    "south_brooklyn": "E"
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



# REP socket to receive weights from the previous node.
recv_socket = context.socket(zmq.REP)
recv_socket.bind(f"tcp://0.0.0.0:{REQ_REP_PORT}")

# REQ socket to send weights to the next node.
send_socket = context.socket(zmq.REQ)
send_socket.connect(f"tcp://{NEXT_NODE_IP}:{REQ_REP_PORT}")





# =========================
# Helper Functions
# =========================

def data_init():
    CSV_DIR = "/data/2024_csvs"
    weather_df=pd.read_csv(f"{CSV_DIR}/fedJan_2024.csv")
    file=pd.read_csv(f"{CSV_DIR}/{area}_{subzone_node[NODE_ID]}_bike_usage.csv")

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

    merged_df=pd.merge(weather_df,file,on="hour",how="inner")
    merged_df["hour"] = pd.to_datetime(merged_df["hour"], errors="coerce")
    merged_df['bike_usage_next'] = merged_df['bike_usage'].shift(-1)
    merged_df = merged_df.iloc[:-1].copy()

    merged_df["hour_of_datetime"]=merged_df["hour"].dt.hour
    merged_df["day_of_week"] = merged_df["hour"].dt.dayofweek
    merged_df['week_of_year'] = merged_df['hour'].dt.isocalendar().week
    merged_df["month"] = merged_df["hour"].dt.month
    #print(months)

    merged_df["hour_sin"]=np.sin(2*np.pi*merged_df["hour_of_datetime"]/24)
    merged_df["hour_cos"]=np.cos(2*np.pi*merged_df["hour_of_datetime"]/24)
    
    #months = merged_df["month"].values
    return merged_df



def pre_processing(merged_df,mean,sigma):

    columns=['hour_sin','hour_cos','day_of_week','month','temp','visibility','dew_point','feels_like','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg',"wind_gust",'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h','clouds_all','weather_main']

    input=merged_df[columns]

    time_feats = input[['hour_sin','hour_cos']].to_numpy(dtype=np.float32)
     

    exclude_cols = ['hour_sin','hour_cos',"day_of_week",'month','weather_main']

    numeric_columns=input.select_dtypes(include=['number']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    print("Numeric columns: ",numeric_columns)

    categorical_columns=['day_of_week','month','weather_main']

    z_scored_data=z_score_normal(input,numeric_columns)
    print(f"The shape of the z_score:{z_scored_data.shape[1]}")
    output=z_score(mean,["bike_usage_next"],sigma,merged_df)
    categ_data=one_hot_encoding(input).astype(np.float32)

    print("This is the categorical data:")
    print(categ_data)
    print(f"The shape of the categ:{categ_data.shape[1]}")

    filtered_input=np.concatenate([time_feats,z_scored_data,categ_data], axis=1).astype(np.float32)

    return filtered_input,output


def one_hot_encoding(df):
    # 1) First, merge “Fog” into “Mist” so it matches your baseline
    tmp = df.copy()
    tmp['weather_main'] = tmp['weather_main'].replace('Fog', 'Mist')

    # 2) Define the full set of categories your baseline used
    ALL_DAYS   = list(range(7))       
    ALL_MONTHS = list(range(1, 13))   
    BASE_WEATHERS = [
        'Clear', 'Clouds', 'Haze', 'Mist',
        'Rain', 'Snow', 'Thunderstorm'
    ]

    # 3) Cast to Categorical with those fixed levels
    tmp['day_of_week']  = pd.Categorical(tmp['day_of_week'],  categories=ALL_DAYS)
    tmp['month']        = pd.Categorical(tmp['month'],        categories=ALL_MONTHS)
    tmp['weather_main'] = pd.Categorical(tmp['weather_main'], categories=BASE_WEATHERS)

    # 4) One-hot encode exactly those three columns
    dummies = pd.get_dummies(
        tmp[['day_of_week','month','weather_main']],
        prefix=['dow','mon','weather'],
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
    """Convert model weights (NumPy arrays) into a JSON string."""
    return json.dumps([w.tolist() for w in weights])

def deserialize_weights(weights_json):
    """Convert JSON string back into a list of NumPy arrays."""
    return [np.array(w) for w in json.loads(weights_json)]

def federated_average(*weight_sets):

    if len(weight_sets) == 1 and isinstance(weight_sets[0], list) and \
       all(isinstance(ws, list) for ws in weight_sets[0]):
        weight_sets = tuple(weight_sets[0])

    num_clients = len(weight_sets)
    layers = zip(*weight_sets)
    return [sum(layer) / num_clients for layer in layers]


def send_weights(weights):
    """Send weights to the next node and wait for an ACK."""
    print(f"[{NODE_ID}] Sending weights to {NEXT_NODE} ({NEXT_NODE_IP}) on port {REQ_REP_PORT}...")
    send_socket.send_string(serialize_weights(weights))
    ack = send_socket.recv_string()
    if ack == "ACK":
        print(f"[{NODE_ID}] Weights successfully sent to {NEXT_NODE}")

def receive_weights():
    """Receive weights from the previous node and send an ACK."""
    print(f"[{NODE_ID}] Waiting for weights from {PREV_NODE} ({PREV_NODE_IP}) on port {REQ_REP_PORT}...")
    weights_json = recv_socket.recv_string()
    recv_socket.send_string("ACK")
    print(f"[{NODE_ID}] Received weights from {PREV_NODE}")
    return deserialize_weights(weights_json)

def send_update_and_wait_for_peers(weights):

    dealer = context.socket(zmq.DEALER)
    dealer.identity = NODE_ID.encode("utf-8")
    dealer.connect(f"tcp://{COORDINATOR_IP}:{GLOBAL_UPDATE_PORT}")

    # 1) Send our own update under our AREA
    payload = {
        "area": area,                      # make sure AREA is defined globally
        "msg":   serialize_weights(weights)
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
        other_updates[peer_area] = deserialize_weights(ser_w)
        print(f"[{NODE_ID}] Received update for peer area={peer_area!r}")

    return other_updates

def split_by_week_chrono(df, week_col='week_of_year', test_frac=0.2, time_col='timestamp'):
    """
    For each week, sort by time and take the *first* (1 - test_frac) as train,
    and the *last* test_frac as test.
    """
    train_parts = []
    test_parts = []
    
    for _, week_df in df.groupby(week_col):
        # make sure it’s sorted in time
        week_df = week_df.sort_values(time_col)
        n = len(week_df)
        split_idx = int(n * (1 - test_frac))
        
        train_parts.append(week_df.iloc[:split_idx])
        test_parts.append(week_df.iloc[split_idx:])
    
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

# =========================
# Main Federated Learning Process
# =========================

def main():
    print("Started main")
    # Give sockets time to connect.
    merged_df=data_init()
    
    #output=merged_df["bike_usage"]
    time.sleep(2)

    print(merged_df)
    baseline=load_baseline_model()

    baseline.compile(
    optimizer="nadam",
    loss="mse",
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.R2Score(name="r2")]
)
 
    model,early_stopping,reduce_lr= neural_network_model()

    train_df, test_df = split_by_week_chrono(merged_df,
                                         week_col='week_of_year',
                                         test_frac=0.2,
                                         time_col='hour')


    test_by_week = {
    week: df_week
    for week, df_week in test_df.groupby("week_of_year")
    }

    for week,week_df in train_df.groupby("week_of_year"):
        
        print(f"This is the week:{week}")

        if week == 1:
            print(f"Skipping week {week} (first week).")
            continue

        week_df = week_df.sort_values("hour").reset_index(drop=True)

        output=week_df["bike_usage_next"]
        global_mean, global_sigma=send_z_score(output,glob_dealer)

        tr_df, val_df = split_train_val_chrono(
        week_df,
        val_frac=0.2,
        time_col='hour'
    )
        X_tr, y_tr = pre_processing(tr_df,  global_mean, global_sigma)
        X_val, y_val = pre_processing(val_df, global_mean, global_sigma)

        #filtered_input,filtered_output=pre_processing(week_df,global_mean,global_sigma)
        #print(f"This is the filtered input shape:{filtered_input.shape[1]}")   

        """
        X_tr, X_val, y_tr, y_val = train_test_split(
        filtered_input, filtered_output, test_size=0.2, random_state=42
        )
        """
        
        # Build and train the local model.
        
        train_the_model(model,X_val,y_val,early_stopping,reduce_lr,X_tr,y_tr,100)
        my_weights = model.get_weights()

        # Perform neighbor-to-neighbor exchanges for a fixed number of rounds.
        for round_num in range(1, TOTAL_ROUNDS + 1):
            print(f"[{NODE_ID}] Round {round_num} starts.")
            # Alternate send/receive order based on (node_index + round_num) % 2.
            if (node_index + round_num) % 2 == 0:
                send_weights(my_weights)
                received_weights = receive_weights()
            else:
                received_weights = receive_weights()
                send_weights(my_weights)
        
            # Perform federated averaging with the neighbor's update.
            my_weights = federated_average(my_weights, received_weights)

        print("federation ended Starting neighbourhood aggregation")
        
        # After all rounds, if this node is the designated requester (e.g., NODE_ID == "A"),
        # fetch the neighborhood-wide aggregated update from the coordinator

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
            #wait for the leader’s broadcast
            frames = neigh_dealer.recv_multipart()
            # frames == [empty_frame, raw_payload]
            raw = frames[-1]
            msg = json.loads(raw.decode("utf-8"))
            peer_weights = [np.array(w) for w in msg["weights"]]
            model.set_weights(peer_weights)
            print(f"[{NODE_ID}] Received global update from leader")

            # ACK back to the leader with the same framing
            ack = json.dumps({"status": "ACK"}).encode("utf-8")
            neigh_dealer.send_multipart([b"", ack])
            print(f"[{NODE_ID}] Sent ACK back to leader")

        week_test = test_by_week.get(week, pd.DataFrame())

        X_test, y_test = pre_processing(week_test, global_mean, global_sigma)

        baseline_results = baseline.evaluate(X_test, y_test, verbose=0)
        base_loss, base_mae, base_rmse, base_r2 = baseline_results
        
        fed_results = model.evaluate(X_test, y_test, verbose=0)
        fed_loss, fed_mae, fed_rmse, fed_r2 = fed_results
        """""
        X_test_tensor = tf.constant(X_test, dtype=tf.float32)
        base_preds = predict_fn(baseline,    X_test_tensor).numpy().flatten()
        fed_preds  = predict_fn(model, X_test_tensor).numpy().flatten()

        base_mae = mean_absolute_error(y_test, base_preds)
        fed_mae  = mean_absolute_error(y_test, fed_preds)

        print(f"Baseline MAE:{base_mae}, Federated MAE:{fed_mae}")
        """""

        HEADER = [
        "week",
        "base_loss", "base_mae", "base_rmse", "base_r2",
        "fed_loss",  "fed_mae",  "fed_rmse",  "fed_r2"
        ]

        results_file = f"/results/node_results/{NODE_ID}_results.csv"

        # Create the file with header if it doesn’t exist
        if not os.path.exists(results_file):
            with open(results_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(HEADER)

        # … inside your weekly loop, after you have computed:
        #    base_loss, base_mae, base_rmse, base_r2
        #    fed_loss,  fed_mae,  fed_rmse,  fed_r2

        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                week,
                base_loss, base_mae, base_rmse, base_r2,
                fed_loss,  fed_mae,  fed_rmse,  fed_r2
            ])
        print(f"[{NODE_ID}] Appended week {week} → {results_file}")

if __name__ == "__main__":
    main()
