# FedShare - Federated Learning Framework

## Overview
This is a federated learning research project implementing three algorithms:
- **FedShare**: The main algorithm with privacy-preserving features
- **FedAvg**: Classical federated averaging algorithm  
- **SCOTCH**: Another federated learning approach

The project is built with Python 3.11, TensorFlow 2.20.0, and Flask for the web interface.

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
- **2025-09-18**: Successfully imported from GitHub and configured for Replit environment
- **2025-09-18 Import Verification**: All dependencies re-installed and verified working (TensorFlow 2.20.0, Flask 3.1.2, etc.)
- Python 3.11 environment with proper package installation
- Scripts are executable and ready to run
- Enhanced Flask app running successfully on port 5000 with 0.0.0.0 binding
- Optimized for fast training iterations in development environment
- Production deployment configured for VM target

## Technical Fixes Applied
- **Flask Debug Mode**: Fixed debug=False and use_reloader=False in fedavgserver.py and fedavgclient.py to prevent nohup conflicts
- **Memory Optimization**: Added TensorFlow threading limits (OMP_NUM_THREADS=1, TF_NUM_INTRAOP_THREADS=1, TF_NUM_INTEROP_THREADS=1) to prevent OOM issues
- **Configuration Simplified**: Reduced to 1 client, 1 server, 1 training round, 2000 dataset size for stability testing
- **Python Interpreter**: Standardized Python executable usage in shell scripts
- **Process Coordination**: Improved startup delays and error handling in start-fedavg.sh
- **Health Check Endpoints**: Added root "/" endpoints to all federated learning clients (fedshareclient.py, fedavgclient.py, scotchclient.py) to resolve 404 errors during health checks that were causing algorithms to get stuck at 25%

## Current Status  
- ✅ **Project fully imported and configured for Replit environment**
- ✅ Web interface fully functional with real-time progress tracking
- ✅ All dependencies installed and verified working (TensorFlow 2.20.0, Flask 3.1.2, etc.)
- ✅ Frontend properly bound to 0.0.0.0:5000 for Replit proxy compatibility
- ✅ All three algorithms (FedShare, FedAvg, SCOTCH) ready to run and tested working
- ✅ Deployment configured for VM target with proper run command
- ✅ Shell scripts executable and properly configured
- ✅ **Import process completed successfully - ready for production use**
- ✅ **2025-09-29**: Re-verified all setup - Python 3.11, dependencies clean install, algorithms working
