"""
Docstring for model
1. About the file:
- Define the neural network architecture
- Initialize weights and biases base on the sample from the distribution
2. About the contents:
- Mainly use numpy for this project due to the simplicity and doesn't require backpropagation, only forward pass
- torch with the neural network is heavier because we must load 200 different state dicts
- Learning project so I prefer primitive implementations
"""

import numpy as np


# Network: Matrices for forward pass but a flatten vector act as a sample in distribution
class PolicyNetwork:
    def __init__(self, input_size=2, hidden_size=[8, 8], output_size=4):
        self.layer_sizes = [input_size] + hidden_size + [output_size] # [2, 8, 8, 4]
        self.param_shapes = []
        self.total_params = 0

        # Calculate shapes for weights and biases
        for i in range(len(self.layer_sizes)- 1):
            w_shape = (self.layer_sizes[i], self.layer_sizes[i+1])
            b_shape = (self.layer_sizes[i+1], )
            self.param_shapes.append((w_shape, b_shape))
            self.total_params += (w_shape[0] * w_shape[1]) + b_shape[0] 
    
    def get_action(self, flat_params, normalized_pos):
        """
        Docstring for get_action
        Act as a forward pass: Input -> MLP -> Logits -> Action 
        - param flat_params: The genome (sample) from distribution, act as the parameters of the neural network
        - param normalized_pos: Input
        """
        x = normalized_pos.reshape(1, -1) 
        start = 0

        # Unpack flat vector into layers then perform forward pass with activation
        for i, (w_shape, b_shape) in enumerate(self.param_shapes):
            # Extract weights
            w_len = w_shape[0] * w_shape[1]
            w = flat_params[start : start + w_len].reshape(w_shape)
            start += w_len

            # Extract bias
            b_len = b_shape[0]
            b = flat_params[start : start + b_len]
            start += b_len

            # Linear operation
            x = np.dot(x, w) + b

            # Activations for hidden layers 
            if i < len(self.param_shapes) - 1: # Check if not the output layer (final layer)
                x = np.maximum(0, x) # ReLU
        
        logits = x[0] # x.shape = (1,4); for example:  x = [[ 0.1, -0.5, 2.3, 0.9 ]]
        
        return np.argmax(logits)
