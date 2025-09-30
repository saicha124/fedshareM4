#!/bin/bash

export PYTHONUNBUFFERED=1
PYTHON=python3

# Read configuration from config.py
N=$($PYTHON -c "from config import Config; print(Config.num_servers)")
M=$($PYTHON -c "from config import Config; print(Config.number_of_clients)")

echo "Configuration: $M clients, $N servers"

DEST_DIRECTORY="fedshare-mnist-client-${M}-server-${N}"
echo "$DEST_DIRECTORY"
mkdir -p logs/${DEST_DIRECTORY}

nohup $PYTHON logger_server.py &>logs/${DEST_DIRECTORY}/logger_server.log &

nohup $PYTHON fedshareleadserver.py &>logs/${DEST_DIRECTORY}/fedshareleadserver.log &

for ((SERVER = 0; SERVER < N; SERVER++)); do
  echo "Running server ${SERVER}"
  nohup $PYTHON fedshareserver.py "${SERVER}" &>"logs/${DEST_DIRECTORY}/fedshareserver-${SERVER}.log" &
done

echo "Waiting for servers to initialize..."
sleep 10

for ((CLIENT = 0; CLIENT < M; CLIENT++)); do
  echo "Running client ${CLIENT}"
  nohup $PYTHON fedshareclient.py "${CLIENT}" &>"logs/${DEST_DIRECTORY}/fedshareclient-${CLIENT}.log" &
done

$PYTHON flask_starter.py
