import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import requests
import tensorflow as tf
import os, time
from model import build_and_train_model

NODE_ID = "A"
PREV_NODE_ID = "D"
NEXT_NODE_IP= '192.168.1.99'
NEXT_NODE_URL = f"http://{NEXT_NODE_IP}:8000/send_weights"
MY_PORT = 8000
TOTAL_NODES = 4

# --- SERVER SETUP ---
app = FastAPI()

class WeightPacket(BaseModel):
    sender: str
    round: int
    weights: str

@app.post("/send_weights")
async def receive_weights(packet: WeightPacket):
    weights = json.loads(packet.weights)
    np.save(f"weights_from_{packet.sender}_round_{packet.round}.npy", weights, allow_pickle=True)
    return {"status": "received", "sender": packet.sender}

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- Client Logic (Main Thread) ---

def serialize_weights(weights):
    return json.dumps([w.tolist() for w in weights])

def federated_average(weights1, weights2):
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def send_weights(weights, round_num):
    data = {
        "sender": NODE_ID,
        "round": round_num,
        "weights": serialize_weights(weights)
    }
    resp = requests.post(NEXT_NODE_URL, json=data)
    print("Weights sent to next node:", resp.json())

def wait_for_weights(round_num, sender_id):
    filename = f"weights_from_{sender_id}_round_{round_num}.npy"
    timeout = 60
    waited = 0
    while not os.path.exists(filename):
        if waited > timeout:
            raise TimeoutError("Did not receive weights.")
        time.sleep(1)
        waited += 1
    return np.load(filename, allow_pickle=True)

def main():
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # give server time to start

    # Local training
    model = build_and_train_model()
    my_weights = model.get_weights()

    TOTAL_NODES = 4
    for round_num in range(1, TOTAL_NODES):
        print(f"Round {round_num} starts.")

        send_weights(my_weights, round_num)

        # Wait and aggregate
        received_weights = wait_for_weights(round_num, PREV_NODE_ID)
        my_weights = federated_average(my_weights, received_weights)

    model.set_weights(my_weights)
    model.save(f"final_model_node_{NODE_ID}.h5")
    print("Final model saved after aggregation.")

if __name__ == "__main__":
    main()
