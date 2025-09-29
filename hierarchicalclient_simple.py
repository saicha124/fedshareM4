#!/usr/bin/env python3
import pickle
import sys
import threading
import time

import numpy as np
import requests
from flask import Flask, request
import tensorflow as tf

import flcommon
import mnistcommon
from config import HierarchicalClientConfig

# Set deterministic seeds for consistent initialization across all clients
np.random.seed(42)
tf.random.set_seed(42)

config = HierarchicalClientConfig(int(sys.argv[1]))

client_datasets = mnistcommon.load_train_dataset(config.number_of_clients, permute=True)

api = Flask(__name__)

round_weight = 0
training_round = 0
total_upload_cost = 0
total_download_cost = 0

def start_next_round(data):
    global training_round
    
    x_train, y_train = client_datasets[config.client_index][0], client_datasets[config.client_index][1]
    model = mnistcommon.get_model()

    if data:  # Only load weights if we received data from server
        round_weight = pickle.loads(data)
        model.set_weights(round_weight)
        print(f"Client {config.client_index} loaded weights from fog node (round {training_round + 1})")

    print(
        f"Model: Hierarchical FL, "
        f"Round: {training_round + 1}/{config.training_rounds}, "
        f"Client {config.client_index + 1}/{config.number_of_clients}, "
        f"Dataset Size: {len(x_train)}")
        
    model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size, verbose=config.verbose,
              validation_split=config.validation_split)

    # Evaluate local client performance on test data
    x_test, y_test = mnistcommon.load_test_dataset()
    local_results = model.evaluate(x_test, y_test, verbose=0)
    local_loss = local_results[0]
    local_accuracy = local_results[1]

    print(f"Client {config.client_index} - Round {training_round + 1} completed - Local Loss: {local_loss:.4f}, Local Accuracy: {local_accuracy:.4f}")

    # Send weights to fog node (simplified version)
    client_weights = model.get_weights()
    serialized_weights = pickle.dumps(client_weights)
    
    # Send to assigned fog node
    fog_node_index = config.client_index % config.num_servers
    fog_node_port = config.fog_base_port + fog_node_index
    
    try:
        response = requests.post(f'http://{config.server_address}:{fog_node_port}/recv', 
                               data=serialized_weights, timeout=30)
        print(f"Client {config.client_index} sent weights to fog node {fog_node_index}")
    except Exception as e:
        print(f"Client {config.client_index} failed to send weights: {e}")

    training_round += 1

@api.route('/recv', methods=['POST'])
def receive_global_weights():
    data = request.get_data()
    try:
        start_next_round(data)
        return "Round completed"
    except Exception as e:
        print(f"Error in receive_global_weights: {e}")
        return f"Error: {e}", 500

@api.route('/', methods=['GET'])
@api.route('/health', methods=['GET'])
def health_check():
    return f"Hierarchical Client {config.client_index} is running"

@api.route('/start', methods=['GET'])
def start_training():
    try:
        start_next_round(None)  # Start first round without initial weights
        return "Training started"
    except Exception as e:
        print(f"Error starting training: {e}")
        return f"Error: {e}", 500

if __name__ == "__main__":
    client_port = config.client_base_port + config.client_index
    print(f"Starting Hierarchical Client {config.client_index} on port {client_port}")
    print(f"Privacy Features: Simplified version (no DP/SS for debugging)")
    
    api.run(host='127.0.0.1', port=client_port, debug=False, use_reloader=False)