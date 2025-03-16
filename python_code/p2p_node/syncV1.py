import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import requests
import tensorflow as tf
import os
import time
from model import build_and_train_model

# Node Configuration
NODE_ID = "A"  # Change based on the node's ID
PREV_NODE_ID = "D"  # Node expected to send weights to this one
NEXT_NODE_IP = "192.168.1.101"
NEXT_NODE_URL = f"http://{NEXT_NODE_IP}:8000/send_weights"
MY_PORT = 8000
TOTAL_NODES = 2

# --- SERVER SETUP ---
app = FastAPI()

class WeightPacket(BaseModel):
    sender: str
    round: int
    weights: str

@app.post("/send_weights")
async def receive_weights(packet: WeightPacket):
    """ Receive weights from another node, process and save them. """
    try:
        weights = np.array(json.loads(packet.weights), dtype=object)
        np.save(f"weights_from_{packet.sender}_round_{packet.round}.npy", weights, allow_pickle=True)
        return {"status": "received", "sender": packet.sender}
    except Exception as e:
        return {"error": str(e)}

def start_server():
    """ Start the FastAPI server in a separate thread. """
    uvicorn.run(app, host="0.0.0.0", port=MY_PORT)

# --- Client Logic (Main Thread) ---

def serialize_weights(weights):
    """ Convert model weights into a JSON serializable format. """
    return json.dumps([np.array(w).tolist() for w in weights], allow_nan=False)

def deserialize_weights(weights_json):
    """ Convert JSON back into NumPy arrays for TensorFlow model weights. """
    return [np.array(w) for w in json.loads(weights_json)]

def federated_average(weights1, weights2):
    """ Perform federated averaging between two sets of weights. """
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def send_weights(weights, round_num):
    """ Send trained weights to the next node. """
    data = {
        "sender": NODE_ID,
        "round": round_num,
        "weights": serialize_weights(weights)
    }
    try:
        resp = requests.post(NEXT_NODE_URL, json=data, timeout=5)
        resp.raise_for_status()
        print(f"✅ Weights sent successfully: {resp.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send weights: {e}")

def wait_for_weights(round_num, sender_id):
    """ Wait for the previous node's weights to arrive. """
    filename = f"weights_from_{sender_id}_round_{round_num}.npy"
    timeout = 60
    waited = 0
    while not os.path.exists(filename):
        if waited > timeout:
            raise TimeoutError("❌ Did not receive weights in time.")
        time.sleep(1)
        waited += 1
    return np.load(filename, allow_pickle=True)

def main():
    """ Main logic for training and federated learning rounds. """
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Give server time to start

    # Train local model
    model = build_and_train_model()
    my_weights = model.get_weights()

    for round_num in range(1, TOTAL_NODES):
        print(f"🚀 Round {round_num} starts.")

        # Send current weights to the next node
        send_weights(my_weights, round_num)

        # Wait for previous node's weights and aggregate
        received_weights = wait_for_weights(round_num, PREV_NODE_ID)
        my_weights = federated_average(my_weights, received_weights)

    model.set_weights(my_weights)
    model.save(f"final_model_node_{NODE_ID}.h5")
    print("✅ Final model saved after federated learning.")

if __name__ == "__main__":
    main()
