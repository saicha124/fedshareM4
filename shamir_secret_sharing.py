import numpy as np
import random
from typing import List, Tuple, Dict, Any


class ShamirSecretSharing:
    """
    Implementation of Shamir's Secret Sharing scheme for federated learning model weights.
    Supports splitting tensors into shares and reconstructing them from threshold shares.
    """
    
    def __init__(self, threshold: int, total_shares: int, prime: int = None):
        """
        Initialize Shamir Secret Sharing
        
        Args:
            threshold: Minimum number of shares needed to reconstruct secret
            total_shares: Total number of shares to generate
            prime: Prime number for finite field operations (auto-generated if None)
        """
        self.threshold = threshold
        self.total_shares = total_shares
        
        # Use a large prime for finite field operations
        # For simplicity, using a fixed large prime that works well with floating point conversion
        self.prime = prime or 2147483647  # 2^31 - 1, a Mersenne prime
        
        if self.threshold > self.total_shares:
            raise ValueError("Threshold cannot be greater than total shares")
    
    def _polynomial_eval(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method"""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result
    
    def _lagrange_interpolation(self, shares: List[Tuple[int, int]], x: int = 0) -> int:
        """
        Perform Lagrange interpolation to reconstruct secret at x=0
        
        Args:
            shares: List of (x, y) coordinate pairs
            x: Point to evaluate at (0 for secret reconstruction)
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        secret = 0
        for i in range(len(shares)):
            xi, yi = shares[i]
            
            # Calculate Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j in range(len(shares)):
                if i != j:
                    xj, _ = shares[j]
                    numerator = (numerator * (x - xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Modular inverse of denominator
            denominator_inv = pow(denominator, self.prime - 2, self.prime)
            
            # Add term to secret
            term = (yi * numerator * denominator_inv) % self.prime
            secret = (secret + term) % self.prime
        
        return secret
    
    def _float_to_int(self, value: float, scale: int = 10000) -> int:
        """Convert float to integer for finite field operations"""
        return int(value * scale) % self.prime
    
    def _int_to_float(self, value: int, scale: int = 10000) -> float:
        """Convert integer back to float"""
        return float(value) / scale
    
    def split_secret(self, secret_array: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Split a numpy array into shares using Shamir's Secret Sharing
        
        Args:
            secret_array: NumPy array to split
            
        Returns:
            Dictionary mapping share_id to share array
        """
        original_shape = secret_array.shape
        flat_secret = secret_array.flatten()
        
        # Convert floats to integers for finite field operations
        int_secret = [self._float_to_int(val) for val in flat_secret]
        
        shares = {}
        for share_id in range(1, self.total_shares + 1):
            shares[share_id] = np.zeros_like(flat_secret, dtype=float)
        
        # Generate shares for each element
        for i, secret_val in enumerate(int_secret):
            # Generate random coefficients for polynomial
            coefficients = [secret_val]  # a_0 = secret
            for _ in range(self.threshold - 1):
                coefficients.append(random.randint(0, self.prime - 1))
            
            # Evaluate polynomial at different points to create shares
            for share_id in range(1, self.total_shares + 1):
                share_val = self._polynomial_eval(coefficients, share_id)
                shares[share_id][i] = self._int_to_float(share_val)
        
        # Reshape shares back to original shape
        for share_id in shares:
            shares[share_id] = shares[share_id].reshape(original_shape)
        
        return shares
    
    def reconstruct_secret(self, shares: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Reconstruct secret from shares
        
        Args:
            shares: Dictionary mapping share_id to share array
            
        Returns:
            Reconstructed secret array
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares for reconstruction")
        
        # Get the first share to determine shape
        first_share_id = next(iter(shares))
        original_shape = shares[first_share_id].shape
        
        # Flatten all shares
        flat_shares = {}
        for share_id, share_array in shares.items():
            flat_shares[share_id] = share_array.flatten()
        
        # Reconstruct each element
        secret_length = len(flat_shares[first_share_id])
        reconstructed = np.zeros(secret_length)
        
        for i in range(secret_length):
            # Collect shares for this element
            element_shares = []
            for share_id, flat_share in flat_shares.items():
                int_val = self._float_to_int(flat_share[i])
                element_shares.append((share_id, int_val))
            
            # Use only threshold number of shares
            element_shares = element_shares[:self.threshold]
            
            # Reconstruct this element
            reconstructed_int = self._lagrange_interpolation(element_shares)
            reconstructed[i] = self._int_to_float(reconstructed_int)
        
        return reconstructed.reshape(original_shape)
    
    def split_model_weights(self, model_weights: List[np.ndarray]) -> Dict[int, List[np.ndarray]]:
        """
        Split entire model weights into shares
        
        Args:
            model_weights: List of weight arrays from model.get_weights()
            
        Returns:
            Dictionary mapping share_id to list of weight shares
        """
        weight_shares = {}
        
        # Initialize share dictionaries
        for share_id in range(1, self.total_shares + 1):
            weight_shares[share_id] = []
        
        # Split each weight array
        for weight_array in model_weights:
            array_shares = self.split_secret(weight_array)
            
            # Append to each participant's share
            for share_id in range(1, self.total_shares + 1):
                weight_shares[share_id].append(array_shares[share_id])
        
        return weight_shares
    
    def reconstruct_model_weights(self, weight_shares: Dict[int, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Reconstruct model weights from shares
        
        Args:
            weight_shares: Dictionary mapping share_id to list of weight shares
            
        Returns:
            Reconstructed list of weight arrays
        """
        if len(weight_shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares for reconstruction")
        
        # Get number of weight arrays
        first_share_id = next(iter(weight_shares))
        num_weights = len(weight_shares[first_share_id])
        
        reconstructed_weights = []
        
        # Reconstruct each weight array
        for weight_idx in range(num_weights):
            # Collect shares for this weight array
            array_shares = {}
            for share_id, shares in weight_shares.items():
                array_shares[share_id] = shares[weight_idx]
            
            # Reconstruct this weight array
            reconstructed_array = self.reconstruct_secret(array_shares)
            reconstructed_weights.append(reconstructed_array)
        
        return reconstructed_weights


def test_shamir_secret_sharing():
    """Test function for Shamir Secret Sharing implementation"""
    print("Testing Shamir Secret Sharing...")
    
    # Test parameters
    threshold = 2
    total_shares = 3
    sss = ShamirSecretSharing(threshold, total_shares)
    
    # Create test model weights
    test_weights = [
        np.random.randn(10, 5).astype(np.float32),
        np.random.randn(5, 1).astype(np.float32)
    ]
    
    print(f"Original weights shapes: {[w.shape for w in test_weights]}")
    
    # Split weights
    weight_shares = sss.split_model_weights(test_weights)
    print(f"Generated {len(weight_shares)} shares")
    
    # Reconstruct from threshold shares
    subset_shares = {k: v for k, v in list(weight_shares.items())[:threshold]}
    reconstructed_weights = sss.reconstruct_model_weights(subset_shares)
    
    print(f"Reconstructed weights shapes: {[w.shape for w in reconstructed_weights]}")
    
    # Check accuracy
    for i, (orig, recon) in enumerate(zip(test_weights, reconstructed_weights)):
        error = np.mean(np.abs(orig - recon))
        print(f"Weight {i} reconstruction error: {error:.6f}")
    
    print("Shamir Secret Sharing test completed!")


if __name__ == "__main__":
    test_shamir_secret_sharing()