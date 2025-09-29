#!/bin/bash

set -euo pipefail

# Kill specific existing processes (not the script itself!)
pkill -f "scotchserver.py" || true
pkill -f "scotchclient.py" || true  
pkill -f "logger_server.py" || true
pkill -f "flask_starter.py" || true
sleep 2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTHONPATH="/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages:$PYTHONPATH"
PYTHON=/home/runner/workspace/.pythonlibs/bin/python3

# Read configuration from config.py
N=$($PYTHON -c "from config import Config; print(Config.num_servers)")
M=$($PYTHON -c "from config import Config; print(Config.number_of_clients)")

echo "Configuration: $M clients, $N servers"

DEST_DIRECTORY="scotch-mnist-client-${M}-server-${N}"
echo "$DEST_DIRECTORY"
mkdir -p logs/${DEST_DIRECTORY}

echo "Using Python: $PYTHON"

# Start logger server with PID tracking
nohup $PYTHON logger_server.py &>logs/${DEST_DIRECTORY}/logger_server.log &
echo $! > logs/${DEST_DIRECTORY}/logger_server.pid
sleep 3

# Start scotch servers with validation
for ((SERVER = 0; SERVER < N; SERVER++)); do
  echo "Starting server ${SERVER}"
  nohup $PYTHON scotchserver.py "${SERVER}" &>logs/${DEST_DIRECTORY}/scotchserver-${SERVER}.log &
  echo $! > logs/${DEST_DIRECTORY}/scotchserver-${SERVER}.pid
  sleep 3
  
  # Wait for server to be ready (up to 10 seconds)
  for i in {1..10}; do
    if grep -q "Running on\|Serving Flask app" logs/${DEST_DIRECTORY}/scotchserver-${SERVER}.log 2>/dev/null; then
      echo "Server ${SERVER} started successfully"
      break
    fi
    sleep 1
  done
done

# Start scotch clients after servers are ready
for ((CLIENT = 0; CLIENT < M; CLIENT++)); do
  echo "Starting client ${CLIENT}"
  nohup $PYTHON scotchclient.py "${CLIENT}" &>logs/${DEST_DIRECTORY}/scotchclient-${CLIENT}.log &
  echo $! > logs/${DEST_DIRECTORY}/scotchclient-${CLIENT}.pid
  sleep 3
  
  # Wait for client to be ready (up to 10 seconds)
  CLIENT_PORT=$((9500 + CLIENT))
  for i in {1..10}; do
    if grep -q "Running on\|Serving Flask app" logs/${DEST_DIRECTORY}/scotchclient-${CLIENT}.log 2>/dev/null; then
      echo "Client ${CLIENT} started successfully on port ${CLIENT_PORT}"
      break
    fi
    sleep 1
  done
done

# Trigger training on all clients after they're all ready
echo "Triggering SCOTCH training..."
sleep 2
for ((CLIENT = 0; CLIENT < M; CLIENT++)); do
  CLIENT_PORT=$((9500 + CLIENT))
  echo "Triggering training on client ${CLIENT} (port ${CLIENT_PORT})"
  curl -s "http://127.0.0.1:${CLIENT_PORT}/start" >> logs/${DEST_DIRECTORY}/training_triggers.log 2>&1 || echo "Failed to trigger client ${CLIENT}"
done

echo "SCOTCH processes started and training triggered successfully"
