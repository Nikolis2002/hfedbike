from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import threading

app =Flask(__name__)
weights=None

@app.route('/receive_weights', methods=['POST'])
def receive_weights():
    global weights
    data=request.json

    