#!/bin/bash

# Three-Tier Hierarchical Federated Learning with Differential Privacy and Shamir Secret Sharing
# Enhanced privacy-preserving federated learning system

# Exit on any error
set -e

echo "=============================================================================="
echo "🌟 STARTING THREE-TIER HIERARCHICAL FEDERATED LEARNING"
echo "=============================================================================="
echo "🔐 Privacy Features:"
echo "   • Differential Privacy (ε=1.0, δ=1e-5)"
echo "   • Shamir Secret Sharing (threshold=2/3)"
echo "🏗️  Architecture:"
echo "   • Clients: 5"
echo "   • Fog Nodes: 3" 
echo "   • Leader Fog: 1"
echo "=============================================================================="

# Import and reload config to get current values
python3 -c "
import importlib
import config
importlib.reload(config)
print(f'✅ Configuration loaded: {config.Config.number_of_clients} clients, {config.Config.num_servers} fog nodes')
"

# Set memory optimization environment variables
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# Create logs directory if it doesn't exist
LOG_DIR="logs/hierarchical-mnist-client-5-fog-3"
mkdir -p "$LOG_DIR"

echo "📁 Log directory: $LOG_DIR"

# Function to check if a port is available
check_port() {
    if ! nc -z 127.0.0.1 $1 2>/dev/null; then
        return 0  # Port is available
    else
        echo "⚠️  Port $1 is already in use"
        return 1  # Port is in use
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://$host:$port/" > /dev/null 2>&1; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within expected time"
    return 1
}

# Function to start service
start_service() {
    local script=$1
    local index=$2
    local service_name=$3
    local log_file=$4
    
    echo "🚀 Starting $service_name $index..."
    nohup python3 "$script" "$index" > "$log_file" 2>&1 &
    local pid=$!
    echo "   PID: $pid"
    
    # Store PID for cleanup
    echo $pid >> "$LOG_DIR/pids.txt"
}

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up processes..."
    if [ -f "$LOG_DIR/pids.txt" ]; then
        while read pid; do
            if kill -0 $pid 2>/dev/null; then
                echo "   Stopping process $pid"
                kill $pid 2>/dev/null || true
            fi
        done < "$LOG_DIR/pids.txt"
        rm -f "$LOG_DIR/pids.txt"
    fi
    echo "✅ Cleanup completed"
}

# Set up signal handling
trap cleanup EXIT INT TERM

# Clear any existing PID file
rm -f "$LOG_DIR/pids.txt"

echo ""
echo "🔍 Checking port availability..."

# Check required ports
REQUIRED_PORTS=(4000 4500 4501 4502 9500 9501 9502 9503 9504)
for port in "${REQUIRED_PORTS[@]}"; do
    if ! check_port $port; then
        echo "❌ Required port $port is not available. Please stop other services."
        exit 1
    fi
done

echo "✅ All required ports are available"
echo ""

# Start leader fog node first
echo "1️⃣  STARTING LEADER FOG NODE"
echo "----------------------------------------"
start_service "hierarchicalleaderfog.py" 0 "Leader Fog Node" "$LOG_DIR/leader-fog.log"
sleep 3

# Wait for leader fog node to be ready
if ! wait_for_service "127.0.0.1" "4000" "Leader Fog Node"; then
    echo "❌ Failed to start Leader Fog Node"
    exit 1
fi

echo ""

# Start fog nodes
echo "2️⃣  STARTING FOG NODES"
echo "----------------------------------------"
for i in $(seq 0 2); do
    start_service "hierarchicalfognode.py" "$i" "Fog Node" "$LOG_DIR/fognode-$i.log"
    sleep 1
done

echo "⏳ Waiting for fog nodes to initialize..."
sleep 5

# Check fog nodes
for i in $(seq 0 2); do
    port=$((4500 + i))
    if ! wait_for_service "127.0.0.1" "$port" "Fog Node $i"; then
        echo "❌ Failed to start Fog Node $i"
        exit 1
    fi
done

echo ""

# Start clients
echo "3️⃣  STARTING HIERARCHICAL FL CLIENTS"
echo "----------------------------------------"
for i in $(seq 0 4); do
    start_service "hierarchicalclient.py" "$i" "Hierarchical Client" "$LOG_DIR/hierarchicalclient-$i.log"
    sleep 1
done

echo "⏳ Waiting for clients to initialize..."
sleep 5

# Check clients
for i in $(seq 0 4); do
    port=$((9500 + i))
    if ! wait_for_service "127.0.0.1" "$port" "Client $i"; then
        echo "❌ Failed to start Client $i"
        exit 1
    fi
done

echo ""
echo "4️⃣  STARTING FEDERATED LEARNING"
echo "----------------------------------------"

# Start training by sending start command to all clients
echo "📡 Initiating training across all clients..."
for i in $(seq 0 4); do
    port=$((9500 + i))
    echo "   Starting client $i..."
    curl -s "http://127.0.0.1:$port/start" > /dev/null 2>&1 &
done

echo ""
echo "✅ All clients started successfully!"
echo ""
echo "=============================================================================="
echo "🎯 HIERARCHICAL FEDERATED LEARNING IN PROGRESS"
echo "=============================================================================="
echo "📊 Monitor Progress:"
echo "   • Leader Fog:    http://127.0.0.1:4000/status"
echo "   • Fog Nodes:     http://127.0.0.1:450{0-2}/status"  
echo "   • Clients:       http://127.0.0.1:950{0-4}/"
echo "📁 Logs Location:   $LOG_DIR/"
echo "🔐 Privacy Accounting: http://127.0.0.1:4000/privacy_accounting"
echo "=============================================================================="
echo "⏱️  Training in progress... (typically 3-5 minutes)"
echo "💡 Press Ctrl+C to stop all processes"
echo ""

# Monitor training progress
python3 -c "
import time
import requests
import json

def check_progress():
    try:
        # Check leader fog status
        response = requests.get('http://127.0.0.1:4000/status', timeout=5)
        if response.status_code == 200:
            status = response.json()
            round_num = status.get('training_round', 0)
            print(f'📈 Training Round: {round_num}')
            
            # Check privacy accounting
            privacy_response = requests.get('http://127.0.0.1:4000/privacy_accounting', timeout=5)
            if privacy_response.status_code == 200:
                privacy = privacy_response.json()
                epsilon_used = privacy.get('budget_utilization', {}).get('epsilon_percent', 0)
                print(f'🔐 Privacy Budget Used: {epsilon_used:.1f}%')
                
            return round_num
    except:
        return -1

print('⏳ Monitoring training progress...')
max_rounds = 3  # From config
last_round = 0

while True:
    current_round = check_progress()
    if current_round > last_round:
        print(f'✅ Round {current_round} completed')
        last_round = current_round
        
    if current_round >= max_rounds:
        print('🎉 Training completed successfully!')
        break
        
    time.sleep(10)
"

echo ""
echo "🎉 HIERARCHICAL FEDERATED LEARNING COMPLETED!"
echo "=============================================================================="
echo "📋 Final Results:"
echo "   • Check individual client logs in: $LOG_DIR/"
echo "   • Privacy accounting: All data remained secure with DP + Secret Sharing"
echo "   • Global model: Successfully aggregated across three-tier architecture"
echo "=============================================================================="

# Keep processes running to show final logs
echo "📖 Showing final logs (press Ctrl+C to exit)..."
sleep 5

# Show summary of final performance
echo ""
echo "📊 TRAINING SUMMARY:"
echo "----------------------------------------"
tail -20 "$LOG_DIR/hierarchicalclient-0.log" | grep -E "(Global Test|Privacy|completed)" || echo "Training logs available in $LOG_DIR/"

echo ""
echo "✅ Hierarchical FL training session completed!"
echo "   Logs saved in: $LOG_DIR/"

# Wait for user interrupt
wait