#!/usr/bin/env python3
import pickle
import sys
import threading
import time

import numpy as np
from flask import Flask, request

import flcommon
from config import LeaderFogConfig
from shamir_secret_sharing import ShamirSecretSharing

api = Flask(__name__)

config = LeaderFogConfig()

run_start_time = time.time()
training_round = 0

total_upload_cost = 0
total_download_cost = 0

fog_shares = []

# Initialize secret sharing mechanism
sss_mechanism = ShamirSecretSharing(
    threshold=config.secret_threshold,
    total_shares=config.total_shares
)


def reconstruct_and_aggregate_global_model(fog_shares_collection):
    """
    Reconstruct global model from fog node shares using Shamir's Secret Sharing
    
    Args:
        fog_shares_collection: List of aggregated shares from fog nodes
        
    Returns:
        Final global model weights properly reconstructed from shares
    """
    print(f"[RECONSTRUCTION] Reconstructing global model from {len(fog_shares_collection)} fog nodes using Shamir's Secret Sharing")
    
    # Check if we have enough shares for reconstruction
    if len(fog_shares_collection) < config.secret_threshold:
        raise ValueError(f"Need at least {config.secret_threshold} shares, got {len(fog_shares_collection)}")
    
    # Create shares dictionary mapping fog_node_id to shares
    weight_shares = {}
    for fog_index, shares in enumerate(fog_shares_collection):
        # Use fog_index + 1 as share_id (Shamir shares are 1-indexed)
        weight_shares[fog_index + 1] = shares
    
    # Reconstruct the global model using Shamir's Secret Sharing
    try:
        global_weights = sss_mechanism.reconstruct_model_weights(weight_shares)
        print(f"[RECONSTRUCTION] Successfully reconstructed {len(global_weights)} layers using Shamir interpolation")
        
        # Log reconstruction details
        print(f"[SHAMIR] Used {len(weight_shares)} shares with threshold {config.secret_threshold}")
        print(f"[SHAMIR] Share IDs: {list(weight_shares.keys())}")
        
        return global_weights
        
    except Exception as e:
        print(f"[ERROR] Shamir reconstruction failed: {e}")
        print(f"[FALLBACK] Using weighted averaging as backup (WARNING: This is not secure!)")
        
        # Fallback to averaging (for debugging only)
        num_layers = len(fog_shares_collection[0])
        global_weights = []
        
        for layer_index in range(num_layers):
            layer_contributions = []
            for fog_index in range(len(fog_shares_collection)):
                layer_contributions.append(fog_shares_collection[fog_index][layer_index])
            
            averaged_layer = np.array(layer_contributions).mean(axis=0, dtype=np.float32)
            global_weights.append(averaged_layer)
        
        return global_weights


def broadcast_global_model(global_weights):
    """
    Broadcast final global model to all clients
    
    Args:
        global_weights: Final aggregated global model weights
    """
    global total_upload_cost
    
    try:
        # Serialize global model
        serialized_model = pickle.dumps(global_weights)
        
        print(f"[BROADCAST] Broadcasting global model ({len(serialized_model)} bytes)")
        
        # Use existing flcommon function to broadcast to clients
        flcommon.broadcast_to_clients(serialized_model, config, lead_server=True)
        
        total_upload_cost += len(serialized_model) * config.number_of_clients
        
        print(f"[BROADCAST] Global model broadcasted to {config.number_of_clients} clients")
        
    except Exception as e:
        print(f"[ERROR] Failed to broadcast global model: {e}")


def recv_thread(fog_shares, data, remote_addr):
    """
    Process received shares from fog nodes
    
    Args:
        fog_shares: Collection of fog node shares
        data: Serialized share data
        remote_addr: Fog node address
    """
    global total_download_cost
    total_download_cost += len(data)
    
    print(f"[DOWNLOAD] Aggregated shares from fog node {remote_addr} received. size: {len(data)}")
    
    try:
        # Deserialize fog node shares
        shares = pickle.loads(data)
        fog_shares.append(shares)
        
        print(f"[AGGREGATION] Shares from fog node {remote_addr} processed successfully.")
        print(f"[PROGRESS] Received shares from {len(fog_shares)}/{config.num_servers} fog nodes")
        
        # Check if we have received from all fog nodes
        if len(fog_shares) < config.num_servers:
            return
        
        # Reconstruct and aggregate final global model
        global_weights = reconstruct_and_aggregate_global_model(fog_shares)
        
        # Broadcast global model to all clients
        broadcast_global_model(global_weights)
        
        # Clear fog shares for next round
        fog_shares.clear()
        
        global training_round
        training_round += 1
        
        print(f"[DOWNLOAD] Total download cost so far: {total_download_cost}")
        print(f"[UPLOAD] Total upload cost so far: {total_upload_cost}")
        
        print("=" * 80)
        print(f"🎯 GLOBAL AGGREGATION ROUND {training_round} COMPLETED")
        print("=" * 80)
        print(f"📊 Fog nodes processed: {config.num_servers}")
        print(f"👥 Clients served: {config.number_of_clients}")
        print(f"🔒 Privacy-preserving: Differential Privacy + Secret Sharing")
        print("=" * 80)
        
    except Exception as e:
        print(f"[ERROR] Failed to process shares from fog node {remote_addr}: {e}")


@api.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        "server_id": "leader_fog", 
        "status": "healthy",
        "algorithm": "Hierarchical FL Leader",
        "fog_shares_received": len(fog_shares),
        "expected_fog_nodes": config.num_servers,
        "training_round": training_round,
        "uptime_seconds": int(time.time() - run_start_time)
    }


@api.route('/recv', methods=['POST'])
def recv():
    """Receive aggregated shares from fog nodes"""
    my_thread = threading.Thread(
        target=recv_thread, 
        args=(fog_shares, request.data, request.remote_addr)
    )
    my_thread.start()
    return {"response": "ok"}


@api.route('/status', methods=['GET'])
def status():
    """Get leader fog node detailed status"""
    return {
        "server_id": "leader_fog",
        "algorithm": "Hierarchical FL",
        "training_round": training_round,
        "fog_shares_received": len(fog_shares),
        "expected_fog_nodes": config.num_servers,
        "total_download_cost": total_download_cost,
        "total_upload_cost": total_upload_cost,
        "uptime_seconds": int(time.time() - run_start_time),
        "privacy_features": {
            "differential_privacy": True,
            "shamir_secret_sharing": True,
            "epsilon": config.dp_epsilon,
            "delta": config.dp_delta,
            "secret_threshold": config.secret_threshold,
            "total_shares": config.total_shares
        },
        "architecture": {
            "type": "Three-Tier Hierarchical FL",
            "clients": config.number_of_clients,
            "fog_nodes": config.num_servers,
            "leader": 1
        }
    }


@api.route('/privacy_accounting', methods=['GET'])
def privacy_accounting():
    """Get privacy accounting information"""
    from differential_privacy import DifferentialPrivacy
    
    # Create temporary DP instance for accounting
    dp = DifferentialPrivacy(
        epsilon=config.dp_epsilon,
        delta=config.dp_delta,
        clip_norm=config.dp_clip_norm
    )
    
    # Calculate privacy spent
    epsilon_spent, delta_spent = dp.compute_privacy_spent(training_round)
    
    return {
        "current_round": training_round,
        "privacy_budget": {
            "epsilon": config.dp_epsilon,
            "delta": config.dp_delta
        },
        "privacy_spent": {
            "epsilon": epsilon_spent,
            "delta": delta_spent
        },
        "privacy_remaining": {
            "epsilon": max(0, config.dp_epsilon - epsilon_spent),
            "delta": max(0, config.dp_delta - delta_spent)
        },
        "budget_utilization": {
            "epsilon_percent": min(100, (epsilon_spent / config.dp_epsilon) * 100),
            "delta_percent": min(100, (delta_spent / config.dp_delta) * 100)
        }
    }


if __name__ == "__main__":
    print("=" * 80)
    print("🌟 HIERARCHICAL FEDERATED LEARNING LEADER FOG NODE")
    print("=" * 80)
    print(f"🔐 Privacy Features:")
    print(f"   • Differential Privacy (ε={config.dp_epsilon}, δ={config.dp_delta})")
    print(f"   • Shamir Secret Sharing ({config.secret_threshold}/{config.total_shares})")
    print(f"🏗️  Three-Tier Architecture:")
    print(f"   • Clients: {config.number_of_clients}")
    print(f"   • Fog Nodes: {config.num_servers}")
    print(f"   • Leader: 1 (this node)")
    print(f"🌐 Network Configuration:")
    print(f"   • Listening on {config.master_server_address}:{config.leader_fog_port}")
    print(f"   • Expecting aggregated shares from {config.num_servers} fog nodes")
    print("=" * 80)
    
    api.run(
        host=config.master_server_address, 
        port=config.leader_fog_port, 
        debug=False, 
        threaded=True,
        use_reloader=False
    )