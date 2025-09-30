class Config:
    number_of_clients = 3  # Number of federated learning clients - optimized for Replit
    train_dataset_size = 3600  # Reduced dataset size for faster training in Replit
    clients_dataset_size = [train_dataset_size/number_of_clients] * number_of_clients
    total_dataset_size = sum(clients_dataset_size)
    num_servers = 2  # Number of servers (fog nodes) - optimized for Replit
    training_rounds = 3  # Multiple rounds for proper convergence
    epochs = 1
    batch_size = 16  # Larger batch size for faster training
    verbose = 1
    validation_split = 0.1
    server_base_port = 8500
    master_server_index = 0
    master_server_port = 7501
    client_address = '127.0.0.1'
    server_address = '127.0.0.1'
    master_server_address = '127.0.0.1'
    buffer_size = 4096
    client_base_port = 9500
    fedavg_server_port = 3500
    logger_address = '127.0.0.1'
    logger_port = 8778
    delay = 10


class ClientConfig(Config):
    def __init__(self, client_index):
        self.client_index = client_index


class ServerConfig(Config):
    def __init__(self, server_index):
        self.server_index = server_index


class LeadConfig(Config):
    def __init__(self):
        pass


class FedAvgServerConfig(Config):
    def __init__(self):
        pass


class HierarchicalConfig(Config):
    def __init__(self):
        # Differential Privacy parameters
        self.dp_epsilon = 1.0  # Privacy budget
        self.dp_delta = 1e-5   # Privacy parameter
        self.dp_clip_norm = 1.0  # Gradient clipping norm
        
        # Shamir Secret Sharing parameters
        self.secret_threshold = 2  # Minimum shares needed to reconstruct secret
        self.total_shares = self.num_servers  # Total number of shares
        
        # Three-tier architecture ports
        self.fog_base_port = 4500  # Base port for fog nodes
        self.leader_fog_port = 4000  # Port for leader fog node
        
        # Validator committee settings
        self.validator_committee_size = min(3, self.number_of_clients)  # Rotating validator committee size


class HierarchicalClientConfig(HierarchicalConfig):
    def __init__(self, client_index):
        super().__init__()
        self.client_index = client_index


class FogNodeConfig(HierarchicalConfig):
    def __init__(self, fog_index):
        super().__init__()
        self.fog_index = fog_index


class LeaderFogConfig(HierarchicalConfig):
    def __init__(self):
        super().__init__()
