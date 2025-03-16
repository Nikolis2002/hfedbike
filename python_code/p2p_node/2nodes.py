import threading
import time
import zmq
import json
import numpy as np
import tensorflow as tf
from model import build_and_train_model

# --- Node Configuration ---
# For a 2-node system, configure as follows:
#
# For Node A (Leader, IP: 192.168.1.99):
#   NODE_ID = "A"
#   PREV_NODE_IP = "192.168.1.101"  # Node D (follower)
#   NEXT_NODE_IP = "192.168.1.101"  # Node D (follower)
#   Bind REP (to receive) on port 5555, connect REQ (to send) to Node D on port 5556
#
# For Node D (Follower, IP: 192.168.1.101):
#   NODE_ID = "D"
#   PREV_NODE_IP = "192.168.1.99"   # Node A (leader)
#   NEXT_NODE_IP = "192.168.1.99"   # Node A (leader)
#   Bind REP (to receive) on port 5556, connect REQ (to send) to Node A on port 5555

# --- Set these values per node:
NODE_ID = "A"  # Set "A" for Node A, "D" for Node D

if NODE_ID == "A":
    PREV_NODE_IP = "192.168.1.101"  # Node D
    NEXT_NODE_IP = "192.168.1.101"  # Node D
    RECV_PORT = 5555  # Leader binds REP here
    SEND_PORT = 5556  # Leader connects REQ here
else:
    PREV_NODE_IP = "192.168.1.99"   # Node A
    NEXT_NODE_IP = "192.168.1.99"   # Node A
    RECV_PORT = 5556  # Follower binds REP here
    SEND_PORT = 5555  # Follower connects REQ here

TOTAL_ROUNDS = 2

# --- ZMQ Context and Sockets ---
context = zmq.Context()

# Set up the receiving socket (REP) for weight exchange
recv_socket = context.socket(zmq.REP)
recv_socket.bind(f"tcp://0.0.0.0:{RECV_PORT}")

# Set up the sending socket (REQ) for weight exchange
send_socket = context.socket(zmq.REQ)
send_socket.connect(f"tcp://{NEXT_NODE_IP}:{SEND_PORT}")

# --- Helper Functions ---

def serialize_weights(weights):
    """Convert model weights to a JSON-serializable string."""
    return json.dumps([w.tolist() for w in weights])

def deserialize_weights(weights_json):
    """Convert JSON string back into list of NumPy arrays."""
    return [np.array(w) for w in json.loads(weights_json)]

def federated_average(weights1, weights2):
    """Average two sets of weights element-wise."""
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def send_weights(weights):
    """Send weights to the next node via REQ-REP and wait for ACK."""
    print(f"[{NODE_ID}] Sending weights to {NEXT_NODE_IP} on port {SEND_PORT}...")
    send_socket.send_string(serialize_weights(weights))
    ack = send_socket.recv_string()
    if ack == "ACK":
        print(f"[{NODE_ID}] Weights successfully sent to {NEXT_NODE_IP}.")

def receive_weights():
    """Receive weights from the previous node via REQ-REP."""
    print(f"[{NODE_ID}] Waiting for weights from previous node on port {RECV_PORT}...")
    weights_json = recv_socket.recv_string()
    recv_socket.send_string("ACK")
    print(f"[{NODE_ID}] Weights received from previous node.")
    return deserialize_weights(weights_json)

# --- Main Federated Learning Process ---
def main():
    # Give sockets time to settle
    time.sleep(2)

    # Train local model
    model = build_and_train_model()
    my_weights = model.get_weights()

    # Leader-Follower Synchronous Exchange:
    # - Leader (Node A) sends first, then waits to receive.
    # - Follower (Node D) waits to receive first, then sends.
    if NODE_ID == "A":
        # Node A: Leader sends first, then receives.
        for round_num in range(1, TOTAL_ROUNDS):
            print(f"[A] Round {round_num} starts.")
            send_weights(my_weights)
            received_weights = receive_weights()
            my_weights = federated_average(my_weights, received_weights)
    else:
        # Node D: Follower receives first, then sends.
        for round_num in range(1, TOTAL_ROUNDS):
            print(f"[D] Round {round_num} starts.")
            received_weights = receive_weights()
            send_weights(my_weights)
            my_weights = federated_average(my_weights, received_weights)

    model.set_weights(my_weights)
    model.save(f"final_model_node_{NODE_ID}.h5")
    print(f"[{NODE_ID}] Final model saved after federated learning.")

if __name__ == "__main__":
    main()
