# dnn_classifier.py - Implements a customizable Deep Neural Network (DNN)

import random
import csv
from math import exp
from typing import List, Tuple, Dict, Any

# --- 1. Core Activation Functions ---

# Sigmoid activation function (Used for hidden layers and often output layer in binary classification)
def activate(weights: List[float], inputs: List[float]) -> float:
    """Calculate neuron activation based on inputs and weights (including bias)."""
    # Weights list always includes the bias weight as the last element (weights[-1])
    activation = weights[-1] # Bias term
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return 1.0 / (1.0 + exp(-activation))

# Derivative of the activation function (Used in backpropagation)
def transfer_derivative(output: float) -> float:
    """Calculate the derivative of the sigmoid transfer function."""
    return output * (1.0 - output)

# --- 2. Network Initialization (Modified for Deep Architectures) ---

def initialize_network(n_inputs: int, hidden_layers: Tuple[int, ...], n_outputs: int) -> List[List[Dict[str, Any]]]:
    """
    Initializes a Deep Neural Network (DNN) with arbitrary hidden layers.

    Args:
        n_inputs: Number of input features.
        hidden_layers: A tuple/list where each number is the neuron count for that hidden layer.
        n_outputs: Number of output neurons (e.g., number of classes).

    Returns:
        The initialized network structure (list of layers).
    """
    network = list()
    prev_neurons = n_inputs  # Start with the size of the input layer

    # 1. Create all Hidden Layers
    for n_hidden in hidden_layers:
        # Each neuron needs 'prev_neurons' weights + 1 bias weight
        limit = 1.0 / (prev_neurons ** 0.5) # Xavier/Glorot  simple
        layer = [{'weights': [random.uniform(-limit, limit) for _ in range(prev_neurons + 1)]} 
                 for _ in range(n_hidden)]
        network.append(layer)
        # The output of this layer is the input for the next layer
        prev_neurons = n_hidden

    # 2. Create the Output Layer
    # Output layer uses the neuron count from the last hidden layer as input size
    limit = 1.0 / (prev_neurons ** 0.5)
    output_layer = [{'weights': [random.uniform(-limit, limit) for _ in range(prev_neurons + 1)]} 
                    for _ in range(n_outputs)]
    network.append(output_layer)

    print(f"Network initialized with {len(network)} layers (Input:{n_inputs}, Hidden:{hidden_layers}, Output:{n_outputs})")
    return network

# --- 3. Forward Propagation (Modified for Deep Architectures) ---

def forward_propagate(network: List[List[Dict[str, Any]]], row: List[float]) -> List[float]:
    """Pass input through the network and store neuron outputs."""
    inputs = row
    
    # Iterate through ALL layers in the network
    for layer in network:
        new_inputs = [] 
        for neuron in layer:
            # Calculate activation (transfer function)
            neuron['output'] = activate(neuron['weights'], inputs)
            new_inputs.append(neuron['output'])
        # The output of the current layer becomes the input for the next layer
        inputs = new_inputs
    
    # The output of the last layer (inputs) is the network's prediction
    return inputs

# --- 4. Backpropagation (Modified for Deep Architectures) ---

def backward_propagate_error(network: List[List[Dict[str, Any]]], expected: List[float]) -> None:
    """Propagate error backwards and store 'delta' for weight updates."""
    # Iterate backwards through ALL layers (from Output layer to first Hidden layer)
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        if i != len(network) - 1:
            # For Hidden Layers: error depends on the weights of the next layer (i+1)
            for j, neuron in enumerate(layer):
                error = 0.0
                # Sum the weighted errors from the next layer
                for neuron_next in network[i + 1]:
                    # neuron_next['weights'][j] is the weight connecting this neuron (j) 
                    # to the neuron_next in the subsequent layer
                    error += (neuron_next['delta'] * neuron_next['weights'][j])
                errors.append(error)
        else:
            # For the Output Layer: error is (Expected - Actual Output)
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])

        # Calculate Delta (Error * Derivative of Activation)
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# --- 5. Weight Update and Training (Uses delta calculated above) ---

def update_weights(network: List[List[Dict[str, Any]]], row: List[float], l_rate: float) -> None:
    """Update network weights using the stored deltas."""
    inputs = row[:-1]  # Exclude the class label

    # Iterate through ALL layers
    for i, layer in enumerate(network):
        # The inputs to this layer come from the previous layer's outputs, 
        # OR the raw data input if it's the first hidden layer (i=0).
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        
        # Update weights for each neuron in the current layer
        for neuron in layer:
            for j in range(len(inputs)):
                # Update weights for input connections
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # Update bias weight (last element)
            neuron['weights'][-1] += l_rate * neuron['delta']

# Function to load your dataset (Assuming a CSV with features followed by class label)
def load_csv(filename: str) -> List[List[str]]:
    """Load a CSV file and return it as a list of lists."""
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# --- 6. Main Execution ---

if __name__ == '__main__':
    # --- Data Loading and Preprocessing ---
    filename = 'data.csv'
    dataset = load_csv(filename)

    # Convert string columns to floats
    # Assuming the first N-1 columns are features, and the last is the integer class label
    for i in range(len(dataset[0]) - 1):
        for row in dataset:
            row[i] = float(row[i])
            
    # Convert last column (class label) to integer
    for row in dataset:
        row[-1] = int(row[-1])

# ----------------------------------------------------------------------------------
# 🛑 FIX 1: FEATURE NORMALIZATION (Essential to prevent Vanishing Gradients) 🛑
# Must be done after converting features to floats.
# ----------------------------------------------------------------------------------
    

    minmax = list()
    # Iterate over features columns (excluding the last column, which is the label)
    for i in range(len(dataset[0]) - 1): 
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])


    for row in dataset:
        for i in range(len(row) - 1): # Iterate over features
            # Get min/max for the current column
            value_min, value_max = minmax[i] 
            
            # Min-Max Normalization: (x - min) / (max - min)
            if value_max != value_min: 
                row[i] = (row[i] - value_min) / (value_max - value_min)
            else:
                row[i] = 0.0 # If all values are the same, set to 0.0

# ----------------------------------------------------------------------------------
# ✅ FIX 2: LABEL MAPPING (Ensures 0-based indexing for One-Hot Encoding) ✅
# Must be done after all data conversion, and before n_outputs calculation.
# ----------------------------------------------------------------------------------
    
    # 1. Identify all unique labels
    unique_labels = sorted(list(set([row[-1] for row in dataset])))

    # 2. Create the mapping: Original Label -> New 0-based Index
    label_map = {original_label: i for i, original_label in enumerate(unique_labels)}

    print(f"Original Labels: {unique_labels} -> Mapped to Indexes: {list(label_map.values())}")

    # 3. Apply the mapping to the entire dataset
    for row in dataset:
        # Overwrite the original label with the new 0-based index
        row[-1] = label_map[row[-1]]

    # Find the number of unique output classes (for n_outputs)
    n_outputs = len(label_map) # Use the size of the map, which is the correct number of classes
    n_inputs = len(dataset[0]) - 1

    # --- Configuration and Training ---
    
    # ----------------------------------------------------------------------------------
    # !!! USER TASK: CHOOSE THE BEST CONFIGURATION (a, b, or c) FOR SUBMISSION !!!
    # ----------------------------------------------------------------------------------
    # Use the configuration that produced the best result among (5), (50, 25, 10), (50, 25, 10, 5)

    # Currently set to the example config 'b' - CHANGE THIS!
    #HIDDEN_LAYERS_CONFIG = (5,)
    
    # Example:
    # HIDDEN_LAYERS_CONFIG = (5,)          # Configuration a
    # HIDDEN_LAYERS_CONFIG = (50, 25, 10)  # Configuration b
    HIDDEN_LAYERS_CONFIG = (50, 25, 10, 5) # Configuration c
    

  

    # Find the number of unique output classes (for n_outputs)
    n_outputs = len(label_map) # Use the size of the map, which is the correct number of classes
    n_inputs = len(dataset[0]) - 1
    # --- Hyperparameters ---
    l_rate = 0.02
    n_epoch = 400
    
    # Initialize the DNN
    network = initialize_network(n_inputs, HIDDEN_LAYERS_CONFIG, n_outputs)

    print(f"Starting training with rate={l_rate}, epochs={n_epoch}...")
    
    # Training Loop
    for epoch in range(n_epoch):
        sum_error = 0
        for row in dataset:
            outputs = forward_propagate(network, row[:-1]) # Propagate features (excluding label)
            expected = [0.0] * n_outputs
            # One-Hot Encoding: Set the expected output position to 1.0
            expected[row[-1]] = 1.0
            
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            
            backward_propagate_error(network, expected) # Backpropagate error (calculate delta)
            update_weights(network, row, l_rate) # Update weights using delta
        
        if (epoch + 1) % 10 == 0:
            print(f'>Epoch={epoch+1}, lrate={l_rate:.3f}, error={sum_error:.5f}')

    # --- Testing/Prediction (Simple Classification) ---
    print("\n--- Testing Predictions ---")
    correct_count = 0
    
    for row in dataset:
        # Get the network's output
        outputs = forward_propagate(network, row[:-1])
        
        # The predicted class is the index with the highest output value
        prediction = outputs.index(max(outputs))
        
        # The true class is the integer in the last column of the row
        true_label = row[-1]
        
        if prediction == true_label:
            correct_count += 1
        
        # print(f'Expected={true_label}, Got={prediction}')

    accuracy = (correct_count / len(dataset)) * 100
    print(f'\nClassification Complete. Total Accuracy: {accuracy:.2f}%')



    '''
    # DNN 

    1 . update 

       l_rate = 0.05
        n_epoch = 200

    2. not usin random but:
    for n_hidden in hidden_layers:
        # 
        limit = 1.0 / (prev_neurons ** 0.5) # Xavier/Glorot 
        
        layer = [{'weights': [random.uniform(-limit, limit) for _ in range(prev_neurons + 1)]} 
                 for _ in range(n_hidden)]
        network.append(layer)
        prev_neurons = n_hidden

    # 2. Create the Output Layer
    #
    limit = 1.0 / (prev_neurons ** 0.5)
    output_layer = [{'weights': [random.uniform(-limit, limit) for _ in range(prev_neurons + 1)]} 
                    for _ in range(n_outputs)]

    3. with hyperparameter setting, result with different layer 

    (50, 25, 10, 5):Starting training with rate=0.02, epochs=400...
>Epoch=10, lrate=0.020, error=139.58523
>Epoch=20, lrate=0.020, error=139.59937
>Epoch=30, lrate=0.020, error=139.60297
>Epoch=40, lrate=0.020, error=139.60652
>Epoch=50, lrate=0.020, error=139.61007
>Epoch=60, lrate=0.020, error=139.61362
>Epoch=70, lrate=0.020, error=139.61717
>Epoch=80, lrate=0.020, error=139.62072
>Epoch=90, lrate=0.020, error=139.62427
>Epoch=100, lrate=0.020, error=139.62782
>Epoch=110, lrate=0.020, error=139.63136
>Epoch=120, lrate=0.020, error=139.63491
>Epoch=130, lrate=0.020, error=139.63844
>Epoch=140, lrate=0.020, error=139.64198
>Epoch=150, lrate=0.020, error=139.64550
>Epoch=160, lrate=0.020, error=139.64901
>Epoch=170, lrate=0.020, error=139.65250
>Epoch=180, lrate=0.020, error=139.65598
>Epoch=190, lrate=0.020, error=139.65943
>Epoch=200, lrate=0.020, error=139.66287
>Epoch=210, lrate=0.020, error=139.66628
>Epoch=220, lrate=0.020, error=139.66966
>Epoch=230, lrate=0.020, error=139.67301
>Epoch=240, lrate=0.020, error=139.67633
>Epoch=250, lrate=0.020, error=139.67962
>Epoch=260, lrate=0.020, error=139.68287
>Epoch=270, lrate=0.020, error=139.68608
>Epoch=280, lrate=0.020, error=139.68925
>Epoch=290, lrate=0.020, error=139.69238
>Epoch=300, lrate=0.020, error=139.69547
>Epoch=310, lrate=0.020, error=139.69852
>Epoch=320, lrate=0.020, error=139.70152
>Epoch=330, lrate=0.020, error=139.70449
>Epoch=340, lrate=0.020, error=139.70741
>Epoch=350, lrate=0.020, error=139.71029
>Epoch=360, lrate=0.020, error=139.71312
>Epoch=370, lrate=0.020, error=139.71592
>Epoch=380, lrate=0.020, error=139.71868
>Epoch=390, lrate=0.020, error=139.72140
>Epoch=400, lrate=0.020, error=139.72409

--- Testing Predictions ---

Classification Complete. Total Accuracy: 33.33%

    (50, 25, 10)
    Starting training with rate=0.02, epochs=400...
>Epoch=10, lrate=0.020, error=139.06289
>Epoch=20, lrate=0.020, error=139.07589
>Epoch=30, lrate=0.020, error=139.08290
>Epoch=40, lrate=0.020, error=139.08901
>Epoch=50, lrate=0.020, error=139.09430
>Epoch=60, lrate=0.020, error=139.09888
>Epoch=70, lrate=0.020, error=139.10287
>Epoch=80, lrate=0.020, error=139.10644
>Epoch=90, lrate=0.020, error=139.10977
>Epoch=100, lrate=0.020, error=139.11305
>Epoch=110, lrate=0.020, error=139.11651
>Epoch=120, lrate=0.020, error=139.12041
>Epoch=130, lrate=0.020, error=139.12502
>Epoch=140, lrate=0.020, error=139.13063
>Epoch=150, lrate=0.020, error=139.13755
>Epoch=160, lrate=0.020, error=139.14606
>Epoch=170, lrate=0.020, error=139.15644
>Epoch=180, lrate=0.020, error=139.16895
>Epoch=190, lrate=0.020, error=139.18376
>Epoch=200, lrate=0.020, error=139.20100
>Epoch=210, lrate=0.020, error=139.22071
>Epoch=220, lrate=0.020, error=139.24282
>Epoch=230, lrate=0.020, error=139.26718
>Epoch=240, lrate=0.020, error=139.29353
>Epoch=250, lrate=0.020, error=139.32149
>Epoch=260, lrate=0.020, error=139.35065
>Epoch=270, lrate=0.020, error=139.38050
>Epoch=280, lrate=0.020, error=139.41052
>Epoch=290, lrate=0.020, error=139.44018
>Epoch=300, lrate=0.020, error=139.46897
>Epoch=310, lrate=0.020, error=139.49642
>Epoch=320, lrate=0.020, error=139.52214
>Epoch=330, lrate=0.020, error=139.54582
>Epoch=340, lrate=0.020, error=139.56725
>Epoch=350, lrate=0.020, error=139.58631
>Epoch=360, lrate=0.020, error=139.60297
>Epoch=370, lrate=0.020, error=139.61727
>Epoch=380, lrate=0.020, error=139.62936
>Epoch=390, lrate=0.020, error=139.63941
>Epoch=400, lrate=0.020, error=139.64764

--- Testing Predictions ---

Classification Complete. Total Accuracy: 33.33%



    (5,) has the best performance , to improve its performance, 
    

    result came out:
    Network initialized with 2 layers (Input:7, Hidden:(5,), Output:3)
Starting training with rate=0.02, epochs=400...
>Epoch=10, lrate=0.020, error=137.35673
>Epoch=20, lrate=0.020, error=132.21564
>Epoch=30, lrate=0.020, error=122.20982
>Epoch=40, lrate=0.020, error=107.50195
>Epoch=50, lrate=0.020, error=94.45478
>Epoch=60, lrate=0.020, error=85.90013
>Epoch=70, lrate=0.020, error=80.43524
>Epoch=80, lrate=0.020, error=76.61684
>Epoch=90, lrate=0.020, error=73.61779
>Epoch=100, lrate=0.020, error=70.97242
>Epoch=110, lrate=0.020, error=68.39763
>Epoch=120, lrate=0.020, error=65.71720
>Epoch=130, lrate=0.020, error=62.83807
>Epoch=140, lrate=0.020, error=59.74805
>Epoch=150, lrate=0.020, error=56.51361
>Epoch=160, lrate=0.020, error=53.25897
>Epoch=170, lrate=0.020, error=50.12428
>Epoch=180, lrate=0.020, error=47.22231
>Epoch=190, lrate=0.020, error=44.61596
>Epoch=200, lrate=0.020, error=42.32059
>Epoch=210, lrate=0.020, error=40.31944
>Epoch=220, lrate=0.020, error=38.57987
>Epoch=230, lrate=0.020, error=37.06475
>Epoch=240, lrate=0.020, error=35.73863
>Epoch=250, lrate=0.020, error=34.57038
>Epoch=260, lrate=0.020, error=33.53376
>Epoch=270, lrate=0.020, error=32.60718
>Epoch=280, lrate=0.020, error=31.77301
>Epoch=290, lrate=0.020, error=31.01693
>Epoch=300, lrate=0.020, error=30.32729
>Epoch=310, lrate=0.020, error=29.69458
>Epoch=320, lrate=0.020, error=29.11101
>Epoch=330, lrate=0.020, error=28.57016
>Epoch=340, lrate=0.020, error=28.06669
>Epoch=350, lrate=0.020, error=27.59617
>Epoch=360, lrate=0.020, error=27.15487
>Epoch=370, lrate=0.020, error=26.73963
>Epoch=380, lrate=0.020, error=26.34777
>Epoch=390, lrate=0.020, error=25.97699
>Epoch=400, lrate=0.020, error=25.62534

--- Testing Predictions ---

Classification Complete. Total Accuracy: 93.33%
    '''