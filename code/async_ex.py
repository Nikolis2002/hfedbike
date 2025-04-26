import zmq
import time
import pickle, base64, sys
import numpy as np
import tensorflow as tf

# -------------------------------
# Helper Functions for Serialization
# -------------------------------
def encode_weights(weights):
    """Serialize weights using pickle and encode with base64 to get a string."""
    return base64.b64encode(pickle.dumps(weights)).decode('utf-8')

def decode_weights(encoded_str):
    """Decode a base64 string and deserialize using pickle."""
    return pickle.loads(base64.b64decode(encoded_str))

# -------------------------------                                                                                                                                                                                                                             
# -------------------------------
def build_and_train_model():
    # Toy dataset and mapping:
    # (lemon, lemon) -> apple, (apple, apple) -> lemon, (lemon, apple)/(apple, lemon) -> banana
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
    model.fit({'fruit1': input1, 'fruit2': input2}, labels, epochs=200, verbose=0)
    
    return model.get_weights()

def fedavg(weights_list):
    """Element-wise averaging of a list of weight sets."""
    return [np.mean(np.array(group), axis=0) for group in zip(*weights_list)]

# -------------------------------
# ZeroMQ Setup for P2P Communication
# -------------------------------
context = zmq.Context()

# Publisher socket: will bind to a port.
pub_socket = context.socket(zmq.PUB)

# Subscriber socket: will connect to peers.
sub_socket = context.socket(zmq.SUB)
sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

# -------------------------------
# Command-Line Configuration
# -------------------------------
# Usage: python node.py <node_id> <bind_port> <peer_endpoint1> <peer_endpoint2> ...
if len(sys.argv) < 3:
    print("Usage: python node.py <node_id> <bind_port> [peer_endpoint1] ...")
    sys.exit(1)

node_id = sys.argv[1]          # e.g., "A", "B", "C", or "D"
bind_port = sys.argv[2]        # Port on which this node binds its PUB socket
peer_endpoints = sys.argv[3:]   # List of peer endpoints (e.g., "tcp://192.168.1.102:5556", etc.)

# Bind the PUB socket (listen on all interfaces on the given port)
pub_socket.bind(f"tcp://*:{bind_port}")
print(f"Node {node_id} is publishing on port {bind_port}")

# Connect SUB socket to each peer's PUB endpoint.
for endpoint in peer_endpoints:
    sub_socket.connect(endpoint)
    print(f"Node {node_id} connected to peer endpoint {endpoint}")

# Set up a poller for the SUB socket.
poller = zmq.Poller()
poller.register(sub_socket, zmq.POLLIN)

# -------------------------------
# Local Model Training and Weight Initialization
# -------------------------------
local_weights = build_and_train_model()
print(f"Node {node_id}: Initial weights:\n{local_weights}")

# -------------------------------
# Function to Publish Local Weights
# -------------------------------
def publish_weights():
    message = {
        'node': node_id,
        'weights': encode_weights(local_weights)
    }
    # Send the entire message as a pickled object.
    pub_socket.send(pickle.dumps(message))
    print(f"Node {node_id}: Published weights.")

# -------------------------------
# One Round of Weight Exchange
# -------------------------------
# Publish our local weights once.
publish_weights()

# Wait for a fixed duration to receive peer updates (e.g., 10 seconds).
end_time = time.time() + 10  # One round: 10 seconds for incoming messages.
received_messages = []

while time.time() < end_time:
    socks = dict(poller.poll(1000))  # Poll for 1 second.
    if sub_socket in socks and socks[sub_socket] == zmq.POLLIN:
        msg = sub_socket.recv()
        received_messages.append(msg)
        print(f"Node {node_id}: Received a message.")

# -------------------------------
# Process Received Messages and Perform FedAvg
# -------------------------------
for msg in received_messages:
    try:
        message = pickle.loads(msg)
        sender = message.get('node')
        if sender == node_id:
            continue  # Ignore our own published message.
        remote_weights = decode_weights(message.get('weights'))
        print(f"Node {node_id}: Received weights from {sender}:\n{remote_weights}")
        # Average current local weights with the received weights.
        local_weights = fedavg([local_weights, remote_weights])
        print(f"Node {node_id}: Updated local weights after averaging with {sender}:\n{local_weights}")
    except Exception as e:
        print(f"Node {node_id}: Error processing message: {e}")

print(f"Node {node_id}: Final aggregated weights after one round:\n{local_weights}")
