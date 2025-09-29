#!/usr/bin/env python3
import pickle
import sys
import threading
import time

import numpy as np
import requests
from flask import Flask, request
from requests_toolbelt.adapters import source
import tensorflow as tf

import flcommon
import mnistcommon
from config import HierarchicalClientConfig
from differential_privacy import DifferentialPrivacy
from shamir_secret_sharing import ShamirSecretSharing

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

# Initialize privacy mechanisms
dp_mechanism = DifferentialPrivacy(
    epsilon=config.dp_epsilon, 
    delta=config.dp_delta, 
    clip_norm=config.dp_clip_norm
)

sss_mechanism = ShamirSecretSharing(
    threshold=config.secret_threshold,
    total_shares=config.total_shares
)


def apply_differential_privacy(model_weights):
    """
    Apply differential privacy to model weights before sharing
    
    Args:
        model_weights: List of weight arrays from model
        
    Returns:
        List of differentially private weight arrays
    """
    print(f"[PRIVACY] Applying differential privacy (ε={config.dp_epsilon}, δ={config.dp_delta})")
    
    # Add calibrated noise to weights for differential privacy
    noisy_weights = dp_mechanism.add_noise_to_weights(model_weights)
    
    # Log privacy parameters
    privacy_params = dp_mechanism.get_privacy_parameters()
    print(f"[PRIVACY] Noise scale: {privacy_params['noise_scale']:.6f}")
    print(f"[PRIVACY] Gradient clipping norm: {privacy_params['clip_norm']}")
    
    return noisy_weights


def create_secret_shares(private_weights):
    """
    Create Shamir secret shares from differentially private weights
    
    Args:
        private_weights: Differentially private model weights
        
    Returns:
        Dictionary mapping fog_node_id to weight shares
    """
    print(f"[SECRET] Creating {config.total_shares} shares with threshold {config.secret_threshold}")
    
    # Generate secret shares for the entire model
    weight_shares = sss_mechanism.split_model_weights(private_weights)
    
    print(f"[SECRET] Successfully created shares for {len(private_weights)} weight layers")
    
    return weight_shares


def send_shares_to_fog_nodes(weight_shares):
    """
    Send secret shares to fog nodes for aggregation
    
    Args:
        weight_shares: Dictionary mapping fog_node_id to weight shares
    """
    global total_upload_cost
    
    print(f"[UPLOAD] Sending shares to {len(weight_shares)} fog nodes")
    
    # Send shares to fog nodes concurrently
    threads = []
    for fog_id, shares in weight_shares.items():
        thread = threading.Thread(
            target=send_to_fog_node, 
            args=(fog_id, shares, training_round)
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all uploads to complete
    for thread in threads:
        thread.join()


def send_to_fog_node(fog_id, shares, round_num):
    """
    Send shares to a specific fog node
    
    Args:
        fog_id: Fog node identifier
        shares: Weight shares for this fog node
        round_num: Current training round
    """
    global total_upload_cost
    
    try:
        # Serialize shares
        serialized_shares = pickle.dumps(shares)
        total_upload_cost += len(serialized_shares)
        
        # Send to fog node (map Shamir share ID to fog node index: 1,2,3 → 0,1,2)
        fog_node_index = fog_id - 1
        url = f'http://{config.server_address}:{config.fog_base_port + fog_node_index}/recv'
        print(f"[DEBUG] Mapping share {fog_id} → fog node {fog_node_index} (port {config.fog_base_port + fog_node_index})")
        s = requests.Session()
        new_source = source.SourceAddressAdapter(flcommon.get_ip(config))
        s.mount('http://', new_source)
        
        response = s.post(url, serialized_shares)
        print(f"[UPLOAD] Sent {len(serialized_shares)} bytes to fog node {fog_id}: {response.json()}")
        
    except Exception as e:
        print(f"[ERROR] Failed to send shares to fog node {fog_id}: {e}")


def start_next_round(data):
    """
    Execute one round of hierarchical federated learning
    
    Args:
        data: Global model weights from previous round (if any)
    """
    
    # Load local dataset
    x_train, y_train = client_datasets[config.client_index][0], client_datasets[config.client_index][1]
    
    # Initialize model
    model = mnistcommon.get_model()
    global training_round
    
    # Load global weights from previous round
    if training_round != 0:
        global round_weight
        round_weight = pickle.loads(data)
        model.set_weights(round_weight)
    
    print(
        f"Model: Hierarchical FL, "
        f"Round: {training_round + 1}/{config.training_rounds}, "
        f"Client {config.client_index + 1}/{config.number_of_clients}, "
        f"Dataset Size: {len(x_train)}")
    
    # Local training
    model.fit(
        x_train, y_train, 
        epochs=config.epochs, 
        batch_size=config.batch_size, 
        verbose=config.verbose,
        validation_split=config.validation_split
    )
    
    # Evaluate local client performance
    x_test, y_test = mnistcommon.load_test_dataset()
    local_results = model.evaluate(x_test, y_test, verbose=0)
    local_loss = local_results[0]
    local_accuracy = local_results[1]
    
    print(f"Client {config.client_index} Local Performance:")
    print(f"  loss: {local_loss:.6f}")
    print(f"  accuracy: {local_accuracy:.6f}")
    
    # Get updated model weights
    model_weights = model.get_weights()
    round_weight = model_weights
    
    # Apply differential privacy
    private_weights = apply_differential_privacy(model_weights)
    
    # Create secret shares
    weight_shares = create_secret_shares(private_weights)
    
    # Send shares to fog nodes
    send_shares_to_fog_nodes(weight_shares)
    
    # Log privacy spending with UI-compatible format
    epsilon_spent, delta_spent = dp_mechanism.compute_privacy_spent(training_round + 1)
    print(f"[PRIVACY] Privacy spent so far: ε={epsilon_spent:.4f}, δ={delta_spent:.6f}")
    print(f"Training Round: {training_round + 1}")  # For UI parsing
    
    global total_download_cost
    print(f"[DOWNLOAD] Total download cost so far: {total_download_cost}")
    print(f"[UPLOAD] Total upload cost so far: {total_upload_cost}")
    
    print(f"********************** Round {training_round} completed **********************")
    training_round += 1
    print("Waiting to receive response from leader fog node...")


@api.route('/recv', methods=['POST'])
def recv():
    """Receive global model from leader fog node"""
    my_thread = threading.Thread(target=recv_thread, args=(request.data,))
    my_thread.start()
    return {"response": "ok"}


def recv_thread(data):
    """Handle received global model"""
    global total_download_cost
    total_download_cost += len(data)
    
    global training_round
    if config.training_rounds == training_round:
        # Training completed - evaluate global performance
        final_weights = pickle.loads(data)
        flcommon.evaluate_global_performance("Hierarchical FL", final_weights, mnistcommon.get_model)
        
        # Final privacy accounting with UI-compatible format
        final_epsilon, final_delta = dp_mechanism.compute_privacy_spent(training_round)
        print("=" * 80)
        print(f"🔐 FINAL PRIVACY ACCOUNTING")
        print("=" * 80)
        print(f"📊 Total Privacy Spent: ε={final_epsilon:.4f}, δ={final_delta:.6f}")
        print(f"🛡️  Privacy Budget Used: {(final_epsilon/config.dp_epsilon)*100:.1f}%")
        print(f"Privacy spent so far: ε={final_epsilon:.4f}, δ={final_delta:.6f}")  # For UI parsing
        print("=" * 80)
        
        print(f"[DOWNLOAD] Total download cost so far: {total_download_cost}")
        print(f"[UPLOAD] Total upload cost so far: {total_upload_cost}")
        
        print("Training finished.")
        return
    
    # Continue with next round
    my_thread = threading.Thread(target=start_next_round, args=(data,))
    my_thread.start()


@api.route('/', methods=['GET'])
@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "client_id": config.client_index,
        "algorithm": "Hierarchical FL",
        "privacy_params": dp_mechanism.get_privacy_parameters(),
        "secret_sharing": {
            "threshold": config.secret_threshold,
            "total_shares": config.total_shares
        }
    }


@api.route('/start', methods=['GET'])
def start():
    """Start federated learning process"""
    print(f"[START] Hierarchical FL Client {config.client_index} starting...")
    print(f"[PRIVACY] Differential privacy enabled (ε={config.dp_epsilon}, δ={config.dp_delta})")
    print(f"[SECRET] Shamir secret sharing enabled (threshold={config.secret_threshold}/{config.total_shares})")
    
    my_thread = threading.Thread(target=start_next_round, args=(0,))
    my_thread.start()
    return {"response": "ok"}


if __name__ == "__main__":
    print(f"🚀 Starting Hierarchical FL Client {config.client_index}")
    print(f"🔐 Privacy: ε={config.dp_epsilon}, δ={config.dp_delta}")
    print(f"🔑 Secret Sharing: {config.secret_threshold}/{config.total_shares}")
    print(f"🌐 Listening on {config.client_address}:{config.client_base_port + int(sys.argv[1])}")
    
    api.run(
        host=flcommon.get_ip(config), 
        port=config.client_base_port + int(sys.argv[1]), 
        debug=False, 
        threaded=True, 
        use_reloader=False
    )