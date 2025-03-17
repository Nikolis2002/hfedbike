import os
import time
import zmq
import json
import numpy as np
import tensorflow as tf
from model import build_and_train_model

# =========================
# Node Configuration
# =========================

# Determine neighborhood: "physical" for real devices, "docker" for docker nodes.
NEIGHBORHOOD = os.getenv("NEIGHBORHOOD", "physical").strip().lower()

if NEIGHBORHOOD == "physical":
    NODE_LIST = ["A", "B", "C"]  # IDs for physical nodes
    ip_mapping = {
        "A": 99,   # 192.168.1.99
        "B": 101,  # 192.168.1.101
        "C": 102   # 192.168.1.102
    }
    # Get NODE_ID from environment or prompt the user.
    node_id = os.getenv("NODE_ID")
    if not node_id:
        node_id = input("Enter node ID (A, B, or C): ").strip().upper()
    NODE_ID = node_id
else:
    NODE_LIST = ["X", "Y", "Z"]  # IDs for Docker nodes
    ip_mapping = {
        "X": 200,  # 192.168.1.200
        "Y": 201,  # 192.168.1.201
        "Z": 202   # 192.168.1.202
    }
    NODE_ID = os.getenv("NODE_ID", NODE_LIST[0]).strip().upper()

NODE_ID = NODE_ID.upper()
node_index = NODE_LIST.index(NODE_ID)
PREV_NODE = NODE_LIST[node_index - 1] if node_index > 0 else NODE_LIST[-1]
NEXT_NODE = NODE_LIST[(node_index + 1) % len(NODE_LIST)]

PREV_NODE_IP = f"192.168.1.{ip_mapping[PREV_NODE]}"
NEXT_NODE_IP = f"192.168.1.{ip_mapping[NEXT_NODE]}"

# Coordinator IP is constant.
COORDINATOR_IP = "192.168.1.150"  # Replace with your actual coordinator IP

# Communication parameters.
REQ_REP_PORT = 5555    # Port for neighbor-to-neighbor REQ-REP exchange.
COORD_PORT = 5570      # Port for coordinator communication.
TOTAL_ROUNDS = 2       # Number of rounds for neighbor exchange.

print(f"Neighborhood: {NEIGHBORHOOD}")
print(f"Current Node: {NODE_ID}")
print(f"Previous Node: {PREV_NODE} at IP {PREV_NODE_IP}")
print(f"Next Node: {NEXT_NODE} at IP {NEXT_NODE_IP}")
print(f"Coordinator IP: {COORDINATOR_IP}")


context = zmq.Context()

# REP socket to receive weights from the previous node.
recv_socket = context.socket(zmq.REP)
recv_socket.bind(f"tcp://0.0.0.0:{REQ_REP_PORT}")

# REQ socket to send weights to the next node.
send_socket = context.socket(zmq.REQ)
send_socket.connect(f"tcp://{NEXT_NODE_IP}:{REQ_REP_PORT}")

# =========================
# Helper Functions
# =========================

def serialize_weights(weights):
    """Convert model weights (NumPy arrays) into a JSON string."""
    return json.dumps([w.tolist() for w in weights])

def deserialize_weights(weights_json):
    """Convert JSON string back into a list of NumPy arrays."""
    return [np.array(w) for w in json.loads(weights_json)]

def federated_average(weights1, weights2):
    """Average corresponding weights from two sets of weights."""
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

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

def fetch_global_update():
    """
    Contact the coordinator to fetch the neighborhood-wide aggregated weights.
    This function is called only after finishing the neighbor rounds.
    """
    coord_socket = context.socket(zmq.REQ)
    coord_socket.connect(f"tcp://{COORDINATOR_IP}:{COORD_PORT}")
    print(f"[{NODE_ID}] Requesting global update from coordinator at {COORDINATOR_IP}:{COORD_PORT}...")
    coord_socket.send_string("REQUEST_UPDATE")
    response = coord_socket.recv_string()
    coord_socket.close()
    if response != "No update available":
        print(f"[{NODE_ID}] Received global update from coordinator.")
        return deserialize_weights(response)
    else:
        print(f"[{NODE_ID}] No global update available.")
        return None

# =========================
# Main Federated Learning Process
# =========================

def main():
    # Give sockets time to connect.
    time.sleep(2)
    
    # Build and train the local model.
    model = build_and_train_model()
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
    
    # After all rounds, if this node is the designated requester (e.g., NODE_ID == "A"),
    # fetch the neighborhood-wide aggregated update from the coordinator.
    if NODE_ID == "A":
        global_update = fetch_global_update()
        if global_update is not None:
            my_weights = federated_average(my_weights, global_update)
        else:
            print(f"[{NODE_ID}] Skipping global update as none was received.")
    
    # Set the final weights and save the model.
    model.set_weights(my_weights)
    model.save(f"final_model_node_{NODE_ID}.h5")
    print(f"[{NODE_ID}] Final model saved after federated learning.")

if __name__ == "__main__":
    main()
