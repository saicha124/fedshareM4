# FedShare - Federated Learning Framework

## Overview
This is a federated learning research project implementing three algorithms:
- **FedShare**: The main algorithm with privacy-preserving features
- **FedAvg**: Classical federated averaging algorithm  
- **SCOTCH**: Another federated learning approach

The project is built with Python 3.12, TensorFlow 2.20.0, and Flask 3.1.2 for the web interface.

## Usage
The project runs on port 5000 with an enhanced web interface featuring:
- Real-time training progress tracking
- Interactive algorithm execution
- Live log viewing
- Performance metrics visualization

Access the web interface at the main URL. Click buttons to run algorithms directly from the browser.

## Technical Setup
- **Frontend**: Enhanced Flask app (`enhanced_app.py`) on port 5000
- **Backend**: Distributed federated learning clients and servers
- **Configuration**: Optimized for Replit with 3 clients, 2 servers, 2 training rounds
- **Dataset**: MNIST (6,000 samples for faster training)
- **Deployment**: Configured for VM deployment target

## Files Structure
- `enhanced_app.py` - Main web interface with progress tracking
- `config.py` - Configuration for clients, servers, and training parameters
- `start-*.sh` - Shell scripts to launch federated learning algorithms
- `logs/` - Training logs and results storage

## Development Notes  
- **2025-09-30**: Fresh GitHub clone successfully configured for Replit environment
- Python 3.12 environment with all dependencies installed and verified (TensorFlow 2.20.0, Flask 3.1.2, NumPy 2.3.3, Pandas 2.3.3, scikit-learn 1.7.2)
- Enhanced Flask app running successfully on port 5000 with 0.0.0.0 binding
- Workflow configured with proper port waiting (5000) and webview output
- Optimized for fast training iterations in development environment
- Production deployment configured for VM target with correct run command
- .gitignore created to properly exclude Python cache and temporary files

## Technical Fixes Applied
- **Flask Debug Mode**: Fixed debug=False and use_reloader=False in fedavgserver.py and fedavgclient.py to prevent nohup conflicts
- **Memory Optimization**: Added TensorFlow threading limits (OMP_NUM_THREADS=1, TF_NUM_INTRAOP_THREADS=1, TF_NUM_INTEROP_THREADS=1) to prevent OOM issues
- **Configuration Simplified**: Reduced to 1 client, 1 server, 1 training round, 2000 dataset size for stability testing
- **Python Interpreter**: Standardized Python executable usage in shell scripts
- **Process Coordination**: Improved startup delays and error handling in start-fedavg.sh
- **Health Check Endpoints**: Added root "/" endpoints to all federated learning clients (fedshareclient.py, fedavgclient.py, scotchclient.py) to resolve 404 errors during health checks that were causing algorithms to get stuck at 25%

## Current Status (2025-09-30)
- ✅ **Fresh GitHub clone successfully imported and configured for Replit**
- ✅ Web interface fully functional with real-time progress tracking
- ✅ All dependencies installed and verified working (TensorFlow 2.20.0, Flask 3.1.2, NumPy, scikit-learn, etc.)
- ✅ Frontend properly bound to 0.0.0.0:5000 for Replit proxy compatibility  
- ✅ Workflow properly configured: webview output, port 5000 waiting
- ✅ Deployment configured for VM target (stateful app with always-on requirement)
- ✅ All three algorithms (FedShare, FedAvg, SCOTCH) ready to run
- ✅ **Import process completed successfully - ready for use**
