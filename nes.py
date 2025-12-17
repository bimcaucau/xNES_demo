"""
Docstring for nes
- Implement xNES algorithm (because using Multivariate-Gaussians and we will use full covariance matrix)
- No adaptation sampling
- No decomposition A into sigma and B for simplification
"""

import numpy as np
from scipy.linalg import expm

class xNES:
    def __init__(self, num_params, pop_size, learning_rate_mu=0.1, learning_rate_sigma=0.05):
        self.d = num_params
        self.pop_size = pop_size # pop_size = lambda - the number of samples we will take

        # Initialize random mean and matrix A
        self.mu = np.random.randn(self.d) * 0.1
        self.A = np.eye(self.d) * 0.5

        self.eta_mu = learning_rate_mu  
        self.eta_sigma = learning_rate_sigma

    def ask(self):
        """
        Docstring for ask
        Sample the population: z = mu + A * s, s ~ N(0, 1)
        """

        self.s = np.random.randn(self.pop_size, self.d)

        self.samples = self.mu + np.dot(self.s, self.A.T)
        
        return self.samples
        
    def tell(self, fitness_scores):
        """
        Docstring for tell
        Update mu and A 
        """
        fitness_scores = np.array(fitness_scores)
        # Fitness shaping with utility function
        ranks = np.argsort(np.argsort(fitness_scores)) # argsort twice to get the ranks, [30,10,20] -> argsort: [1, 2, 0] -> argsort: [2, 0, 1] - the ranks
        utilities = (ranks / (self.pop_size - 1)) - 0.5
        utilities = utilities / self.pop_size 

        # Calculate gradient
        grad_delta = np.dot(utilities, self.s)
        # Calculate grad_M manually instead of np.dot(utilities, s*s - I) because AI believe that would improve performance
        grad_M = np.zeros((self.d, self.d))
        for i in range(self.pop_size):
            s_i = self.s[i].reshape(-1, 1)
            grad_M += utilities[i] * (np.dot(s_i, s_i.T) - np.eye(self.d))

        # Update
        # Updating mu base on old A, then update A to new A later
        self.mu = self.mu + self.eta_mu * np.dot(self.A, grad_delta)

        update_mat = expm((self.eta_sigma / 2) * grad_M)
        self.A = np.dot(self.A, update_mat)
    
    def save(self, filename="nes_checkpoint.npz"):
        """
        Docstring for save
        Save the current distributions 
        """
        np.savez(filename, mu=self.mu, A=self.A)
        print(f"Saved checkpoint to {filename}")
    
    def load(self, filename="nes_checkpoint.npz"):
        """
        Docstring for load
        Load distributions from checkpoint
        """
        try:
            data = np.load(filename)
            self.mu = data['mu']
            self.A = data['A']
            print(f"Successfully loaded checkpoint: {filename}")
            return True
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch.")
            return False
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False