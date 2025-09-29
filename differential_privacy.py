import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional


class DifferentialPrivacy:
    """
    Differential Privacy implementation for federated learning.
    Provides mechanisms to add calibrated noise to model gradients/weights.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        """
        Initialize Differential Privacy parameters
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter for (ε,δ)-differential privacy
            clip_norm: L2 norm bound for gradient clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Sensitivity calculation - this is the maximum change one individual can make
        self.sensitivity = 2.0 * self.clip_norm
        
    def _compute_noise_scale(self) -> float:
        """
        Compute the scale parameter for Gaussian noise to achieve (ε,δ)-DP
        
        Returns:
            Scale parameter for noise
        """
        # For Gaussian mechanism, σ ≥ Δ * sqrt(2 * ln(1.25/δ)) / ε
        # Where Δ is the sensitivity
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        Clip gradients to have bounded L2 norm
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Tuple of (clipped_gradients, clipping_factor)
        """
        # Compute global L2 norm of all gradients
        global_norm = 0.0
        for grad in gradients:
            global_norm += np.sum(grad ** 2)
        global_norm = np.sqrt(global_norm)
        
        # Compute clipping factor
        clipping_factor = min(1.0, self.clip_norm / (global_norm + 1e-8))
        
        # Apply clipping
        clipped_gradients = []
        for grad in gradients:
            clipped_gradients.append(grad * clipping_factor)
        
        return clipped_gradients, clipping_factor
    
    def add_noise_to_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add calibrated Gaussian noise to model weights for differential privacy
        
        Args:
            weights: List of weight arrays from model
            
        Returns:
            List of noisy weight arrays
        """
        noise_scale = self._compute_noise_scale()
        noisy_weights = []
        
        for weight_array in weights:
            # Generate Gaussian noise with the same shape as weights
            noise = np.random.normal(0, noise_scale, weight_array.shape)
            
            # Add noise to weights
            noisy_weight = weight_array + noise
            noisy_weights.append(noisy_weight.astype(weight_array.dtype))
        
        return noisy_weights
    
    def add_noise_to_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add noise to gradients with clipping for differential privacy
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            List of noisy gradient arrays
        """
        # First clip gradients
        clipped_gradients, clipping_factor = self.clip_gradients(gradients)
        
        # Then add noise
        noise_scale = self._compute_noise_scale()
        noisy_gradients = []
        
        for grad in clipped_gradients:
            # Generate Gaussian noise
            noise = np.random.normal(0, noise_scale, grad.shape)
            
            # Add noise to gradient
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad.astype(grad.dtype))
        
        return noisy_gradients
    
    def compute_privacy_spent(self, num_queries: int) -> Tuple[float, float]:
        """
        Compute total privacy spent after num_queries
        
        Args:
            num_queries: Number of queries/training rounds
            
        Returns:
            Tuple of (total_epsilon, total_delta)
        """
        # For composition of Gaussian mechanisms, we use strong composition
        # This is a simplified calculation - in practice, use more sophisticated accounting
        total_epsilon = self.epsilon * np.sqrt(2 * num_queries * np.log(1 / self.delta))
        total_delta = num_queries * self.delta
        
        return min(total_epsilon, self.epsilon * num_queries), min(total_delta, 1.0)
    
    def get_privacy_parameters(self) -> dict:
        """Get current privacy parameters"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'clip_norm': self.clip_norm,
            'sensitivity': self.sensitivity,
            'noise_scale': self._compute_noise_scale()
        }


def test_differential_privacy():
    """Test function for Differential Privacy implementation"""
    print("Testing Differential Privacy...")
    
    # Initialize DP mechanism
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    
    # Create test weights
    test_weights = [
        np.random.randn(10, 5).astype(np.float32),
        np.random.randn(5, 1).astype(np.float32)
    ]
    
    print(f"Original weights shapes: {[w.shape for w in test_weights]}")
    print(f"Privacy parameters: {dp.get_privacy_parameters()}")
    
    # Add noise to weights
    noisy_weights = dp.add_noise_to_weights(test_weights)
    
    # Calculate noise added
    for i, (orig, noisy) in enumerate(zip(test_weights, noisy_weights)):
        noise_magnitude = np.linalg.norm(orig - noisy)
        weight_magnitude = np.linalg.norm(orig)
        print(f"Weight {i} - Original norm: {weight_magnitude:.4f}, Noise norm: {noise_magnitude:.4f}")
    
    # Test gradient clipping
    large_gradients = [np.random.randn(10, 5) * 10]  # Large gradients
    clipped_grads, clip_factor = dp.clip_gradients(large_gradients)
    
    print(f"Gradient clipping factor: {clip_factor:.4f}")
    
    # Test privacy accounting
    epsilon_spent, delta_spent = dp.compute_privacy_spent(num_queries=10)
    print(f"Privacy spent after 10 queries: ε={epsilon_spent:.4f}, δ={delta_spent:.6f}")
    
    print("Differential Privacy test completed!")


if __name__ == "__main__":
    test_differential_privacy()