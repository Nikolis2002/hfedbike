import threading
import time
import zmq
import json
import numpy as np
import tensorflow as tf
from model import build_and_train_model

# --- Node Configuration ---

# Define the list of nodes and explicit IP mapping.
NODE_LIST = ["A", "B", "C"]  # For a 3-node ring; extend as needed.
ip_mapping = {
    "A": 99,    # Node A: 192.168.1.99
    "B": 101,   # Node B: 192.168.1.101
    "C": 102    # Node C: 192.168.1.102
}

# Set the current node's ID.
# (Change this value on each device to "A", "B", or "C" accordingly.)
NODE_ID = "A"

# Determine this node's index in the ring.
node_index = NODE_LIST.index(NODE_ID)

# Determine previous and next nodes in the ring.
PREV_NODE = NODE_LIST[node_index - 1] if node_index > 0 else NODE_LIST[-1]
NEXT_NODE = NODE_LIST[(node_index + 1) % len(NODE_LIST)]

# Build the IP addresses from the mapping.
PREV_NODE_IP = f"192.168.1.{ip_mapping[PREV_NODE]}"
NEXT_NODE_IP = f"192.168.1.{ip_mapping[NEXT_NODE]}"

# Use a common port for weight exchange (REQ-REP).
REQ_REP_PORT = 5555

# Total rounds of communication for federated averaging.
TOTAL_ROUNDS = 2  # You can adjust this as needed.

print(f"Current Node: {NODE_ID}")
print(f"Previous Node: {PREV_NODE} at IP {PREV_NODE_IP}")
print(f"Next Node: {NEXT_NODE} at IP {NEXT_NODE_IP}")

# --- ZMQ Context and Sockets ---
context = zmq.Context()

# Each node binds its REP socket to receive weights on REQ_REP_PORT.
recv_socket = context.socket(zmq.REP)
recv_socket.bind(f"tcp://0.0.0.0:{REQ_REP_PORT}")

# Each node's REQ socket connects to the next node's receive port.
send_socket = context.socket(zmq.REQ)
send_socket.connect(f"tcp://{NEXT_NODE_IP}:{REQ_REP_PORT}")

# --- Helper Functions ---

def serialize_weights(weights):
    """Convert model weights (NumPy arrays) into a JSON string."""
    return json.dumps([w.tolist() for w in weights])

def deserialize_weights(weights_json):
    """Convert JSON string back into a list of NumPy arrays."""
    return [np.array(w) for w in json.loads(weights_json)]

def federated_average(weights1, weights2):
    """Perform federated averaging by averaging corresponding weights."""
    return [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

def send_weights(weights):
    """Send weights to the next node using REQ-REP and wait for an ACK."""
    print(f"[{NODE_ID}] 📤 Sending weights to {NEXT_NODE} ({NEXT_NODE_IP}) on port {REQ_REP_PORT}...")
    send_socket.send_string(serialize_weights(weights))
    ack = send_socket.recv_string()
    if ack == "ACK":
        print(f"[{NODE_ID}] ✅ Weights successfully sent to {NEXT_NODE}")

def receive_weights():
    """Wait for weights from the previous node using REQ-REP."""
    print(f"[{NODE_ID}] 📥 Waiting for weights from {PREV_NODE} ({PREV_NODE_IP}) on port {REQ_REP_PORT}...")
    weights_json = recv_socket.recv_string()
    recv_socket.send_string("ACK")
    print(f"[{NODE_ID}] ✅ Received weights from {PREV_NODE}")
    return deserialize_weights(weights_json)

# --- Alternating Communication in Rounds ---
def main():
    # Give sockets time to connect.
    time.sleep(2)
    
    # Train the local model.
    model = build_and_train_model()
    my_weights = model.get_weights()

    # Run multiple rounds of federated averaging.
    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"[{NODE_ID}] 🚀 Round {round_num} starts.")
        
        # Alternate send/receive order based on (node_index + round_num) modulo 2.
        if (node_index + round_num) % 2 == 0:
            # For this round, send first then receive.
            send_weights(my_weights)
            received_weights = receive_weights()
        else:
            # For this round, receive first then send.
            received_weights = receive_weights()
            send_weights(my_weights)
        
        # Perform federated averaging.
        my_weights = federated_average(my_weights, received_weights)

    # Update model weights and save the final model.
    model.set_weights(my_weights)
    model.save(f"final_model_node_{NODE_ID}.h5")
    print(f"[{NODE_ID}] ✅ Final model saved after federated learning.")

if __name__ == "__main__":
    main()
