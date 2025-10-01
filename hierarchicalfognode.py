#!/usr/bin/env python3
import pickle
import sys
import threading
import time

import numpy as np
import requests
from flask import Flask, request
from requests_toolbelt.adapters import source

from config import FogNodeConfig
from shamir_secret_sharing import ShamirSecretSharing

config = FogNodeConfig(int(sys.argv[1]))

api = Flask(__name__)

training_round = 0
client_shares = []
total_download_cost = 0
total_upload_cost = 0

# Initialize secret sharing mechanism
sss_mechanism = ShamirSecretSharing(
    threshold=config.secret_threshold,
    total_shares=config.total_shares
)


def aggregate_shares(shares_collection):
    """
    Simple averaging of shares from clients (no weighted aggregation on shares)
    
    CRITICAL: Shamir shares in this implementation are stored as floats.
    We use simple averaging (not weighted) to preserve numerical stability.
    Leader will handle weighted aggregation after reconstruction.
    
    Args:
        shares_collection: List of client shares
        
    Returns:
        Averaged shares to send to leader fog node
    """
    print(f"[AGGREGATION] Averaging shares from {len(shares_collection)} clients using SIMPLE MEAN")
    
    num_clients = len(shares_collection)
    num_layers = len(shares_collection[0])
    
    # Simple averaging: sum all shares and divide by number of clients
    averaged_shares = {}
    
    for layer_index in range(num_layers):
        layer_sum = None
        
        for client_share_list in shares_collection:
            client_share = client_share_list[layer_index]
            
            if layer_sum is None:
                layer_sum = client_share.astype(np.float64)
            else:
                layer_sum = layer_sum + client_share.astype(np.float64)
        
        # Simple average: divide by number of clients
        averaged_share = (layer_sum / num_clients).astype(np.float32)
        averaged_shares[layer_index] = averaged_share
    
    # Convert to list format
    aggregated_list = [averaged_shares[i] for i in range(len(averaged_shares))]
    
    print(f"[AGGREGATION] Successfully averaged {len(aggregated_list)} layers (simple mean)")
    print(f"[INFO] Weighted aggregation will be done at leader level after reconstruction")
    
    return aggregated_list


def send_to_leader_fog(aggregated_shares):
    """
    Send averaged shares to leader fog node
    
    Args:
        aggregated_shares: Averaged shares from clients
    """
    global total_upload_cost
    
    try:
        # Serialize shares
        serialized_shares = pickle.dumps(aggregated_shares)
        total_upload_cost += len(serialized_shares)
        
        # Send to leader fog node
        url = f'http://{config.server_address}:{config.leader_fog_port}/recv'
        s = requests.Session()
        new_source = source.SourceAddressAdapter(config.server_address)
        s.mount('http://', new_source)
        
        response = s.post(url, serialized_shares)
        
        print(f"[UPLOAD] Sent {len(serialized_shares)} bytes to leader fog node: {response.json()}")
        
    except Exception as e:
        print(f"[ERROR] Failed to send to leader fog node: {e}")


def recv_thread(client_shares, data, remote_addr):
    """
    Process received shares from clients
    
    Args:
        client_shares: Collection of client shares
        data: Serialized share data
        remote_addr: Client address
    """
    
    global total_download_cost
    total_download_cost += len(data)
    
    print(f"[DOWNLOAD] Share from {remote_addr} received. size: {len(data)}", flush=True)
    
    try:
        # Deserialize client shares
        share = pickle.loads(data)
        client_shares.append(share)
        
        print(f"[SECRET] Share from {remote_addr} processed successfully.", flush=True)
        print(f"[PROGRESS] Received {len(client_shares)}/{config.number_of_clients} shares", flush=True)
        
        # Check if we have enough shares to proceed
        if len(client_shares) < config.number_of_clients:
            return
        
        # Aggregate shares from all clients
        print(f"[AGGREGATION] Starting aggregation of {len(client_shares)} shares...", flush=True)
        aggregated_shares = aggregate_shares(client_shares)
        
        # Send aggregated result to leader fog node
        print(f"[UPLOAD] Sending aggregated shares to leader fog...", flush=True)
        send_to_leader_fog(aggregated_shares)
        
        # Clear shares for next round
        client_shares.clear()
        
        global training_round
        training_round += 1
        
        print(f"[DOWNLOAD] Total download cost so far: {total_download_cost}", flush=True)
        print(f"[UPLOAD] Total upload cost so far: {total_upload_cost}", flush=True)
        
        print(f"********************** [FOG] Round {training_round} completed **********************", flush=True)
        
    except Exception as e:
        print(f"[ERROR] Failed to process share from {remote_addr}: {e}", flush=True)
        import traceback
        traceback.print_exc()


@api.route('/', methods=['GET'])
@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        "fog_id": config.fog_index, 
        "status": "healthy",
        "algorithm": "Hierarchical FL Fog Node",
        "shares_received": len(client_shares),
        "expected_clients": config.number_of_clients,
        "secret_sharing": {
            "threshold": config.secret_threshold,
            "total_shares": config.total_shares
        }
    }


@api.route('/recv', methods=['POST'])
def recv():
    """Receive shares from clients"""
    my_thread = threading.Thread(
        target=recv_thread, 
        args=(client_shares, request.data, request.remote_addr)
    )
    my_thread.start()
    return {"response": "ok"}


@api.route('/status', methods=['GET'])
def status():
    """Get fog node status"""
    return {
        "fog_id": config.fog_index,
        "training_round": training_round,
        "shares_received": len(client_shares),
        "expected_clients": config.number_of_clients,
        "total_download_cost": total_download_cost,
        "total_upload_cost": total_upload_cost
    }


if __name__ == "__main__":
    print(f"🌫️  Starting Hierarchical FL Fog Node {config.fog_index}")
    print(f"🔑 Secret Sharing: {config.secret_threshold}/{config.total_shares}")
    print(f"🌐 Listening on {config.server_address}:{config.fog_base_port + int(sys.argv[1])}")
    print(f"👥 Expecting shares from {config.number_of_clients} clients")
    print(f"📡 Will forward aggregated shares to leader at port {config.leader_fog_port}")
    
    api.run(
        host=config.server_address, 
        port=config.fog_base_port + int(sys.argv[1]), 
        debug=False, 
        threaded=True,
        use_reloader=False
    )