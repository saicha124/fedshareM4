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
    Aggregate secret shares from multiple clients using integer-weighted sums
    
    CRITICAL: Shamir shares MUST remain integers in the finite field.
    We compute integer-weighted sums (using dataset sizes as weights) and send
    the total weight to the leader, which divides after reconstruction.
    
    Args:
        shares_collection: List of client shares
        
    Returns:
        Tuple of (aggregated_shares, total_weight) to send to leader fog node
    """
    print(f"[AGGREGATION] Aggregating shares from {len(shares_collection)} clients")
    
    # Use dataset sizes as integer weights (NO floating point operations on shares!)
    total_participating_clients = len(shares_collection)
    participating_dataset_sizes = config.clients_dataset_size[:total_participating_clients]
    total_weight = int(sum(participating_dataset_sizes))
    
    print(f"[AGGREGATION] Fog Node {config.fog_index} Aggregation Details:")
    print(f"  Participating clients: {total_participating_clients}")
    print(f"  Dataset sizes (integer weights): {participating_dataset_sizes}")
    print(f"  Total weight: {total_weight}")
    
    # Perform INTEGER-weighted sum of shares (preserves finite field properties)
    aggregated_shares = {}
    num_layers = len(shares_collection[0])
    
    for layer_index in range(num_layers):
        # Initialize sum with zeros in int64 to avoid overflow
        layer_sum = None
        
        for client_index in range(total_participating_clients):
            client_share = shares_collection[client_index][layer_index]
            # Integer weight (dataset size)
            weight = int(participating_dataset_sizes[client_index])
            
            # Multiply share by integer weight (stays in finite field)
            weighted_share = client_share * weight
            
            if layer_sum is None:
                layer_sum = weighted_share.astype(np.int64)
            else:
                layer_sum = layer_sum + weighted_share.astype(np.int64)
        
        aggregated_shares[layer_index] = layer_sum
    
    # Convert back to list format
    aggregated_list = [aggregated_shares[i] for i in range(len(aggregated_shares))]
    
    print(f"[AGGREGATION] Successfully aggregated {len(aggregated_list)} layers using integer weights")
    print(f"[AGGREGATION] Sending total_weight={total_weight} to leader for post-reconstruction division")
    
    return aggregated_list, total_weight


def send_to_leader_fog(aggregated_shares, total_weight):
    """
    Send aggregated shares and total weight to leader fog node
    
    Args:
        aggregated_shares: Aggregated weight shares (integer-weighted sum)
        total_weight: Total dataset size for post-reconstruction division
    """
    global total_upload_cost
    
    try:
        # Package both shares and weight together
        payload = {
            'shares': aggregated_shares,
            'total_weight': total_weight,
            'fog_id': config.fog_index
        }
        
        # Serialize payload
        serialized_payload = pickle.dumps(payload)
        total_upload_cost += len(serialized_payload)
        
        # Send to leader fog node
        url = f'http://{config.server_address}:{config.leader_fog_port}/recv'
        s = requests.Session()
        new_source = source.SourceAddressAdapter(config.server_address)
        s.mount('http://', new_source)
        
        response = s.post(url, serialized_payload)
        
        print(f"[UPLOAD] Sent {len(serialized_payload)} bytes to leader fog node: {response.json()}")
        print(f"[UPLOAD] Payload includes shares + total_weight={total_weight}")
        
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
        aggregated_shares, total_weight = aggregate_shares(client_shares)
        
        # Send aggregated result to leader fog node
        print(f"[UPLOAD] Sending aggregated shares to leader fog...", flush=True)
        send_to_leader_fog(aggregated_shares, total_weight)
        
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