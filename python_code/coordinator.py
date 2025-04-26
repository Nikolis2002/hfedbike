import zmq
import json
import numpy as np
import time

# Configuration for the reverse proxy:
LOCAL_UPDATE_PORT = 5560  # Port where local aggregators publish updates 
REQUEST_PORT = 5570       # Port where neighborhoods request the latest global update

# Create a ZeroMQ context
context = zmq.Context()

# SUB socket: receives local updates

sub_socket = context.socket(zmq.SUB)
#sub_socket.connect("tcp://192.168.1.200:5560")  # X
sub_socket.connect("tcp://192.168.1.204:5560")  # P
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")


# REP socket: responds to requests for the latest update
rep_socket = context.socket(zmq.REP)
rep_socket.bind(f"tcp://*:{REQUEST_PORT}")

# Variable to store the most recent update (serialized as JSON)
stored_update = None

# Set up a poller to handle both sockets
poller = zmq.Poller()
poller.register(sub_socket, zmq.POLLIN)
poller.register(rep_socket, zmq.POLLIN)

print("i use the new codeA ")
print("Reverse Proxy started. Waiting for local updates and requests...")

while True:
    # Poll both sockets with a timeout (in milliseconds)
    events = dict(poller.poll(timeout=1000))
    
    # Check for a new local update
    if sub_socket in events and events[sub_socket] == zmq.POLLIN:
        try:
            msg = sub_socket.recv_string()
            stored_update = msg  # Cache the update
            print("Stored a new local update.")
        except Exception as e:
            print("Error receiving local update:", e)
    
    # Check for a request from a subscriber (e.g., a neighborhood aggregator)
    if rep_socket in events and events[rep_socket] == zmq.POLLIN:
        try:
            # The request content may be ignored; it's just a trigger.
            _ = rep_socket.recv_string()
            if stored_update is not None:
                rep_socket.send_string(stored_update)
                print("Sent stored update to a requester.")
            else:
                rep_socket.send_string("No update available")
                print("No update available to send.")
        except Exception as e:
            print("Error handling request:", e)
    
    #time.sleep(0.1)
