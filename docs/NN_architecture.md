### Neural Network Architecture

The neural network model is designed using TensorFlow's Keras API and consists of the following layers:

1. **Input Layer**:
   - **Shape**: `(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * MEMORY_SIZE)`
   - **Name**: `"input_state"`

2. **Convolutional Layers**:
   - **Conv2D Layer 1**:
     - Filters: 32
     - Kernel Size: 5
     - Activation: ReLU
     - Kernel Initializer: HeNormal
   - **MaxPooling2D Layer 1**:
     - Pool Size: 3
   - **Conv2D Layer 2**:
     - Filters: 64
     - Kernel Size: 5
     - Activation: ReLU
     - Kernel Initializer: HeNormal
   - **MaxPooling2D Layer 2**:
     - Pool Size: 3
   - **Conv2D Layer 3**:
     - Filters: 128
     - Kernel Size: 5
     - Activation: ReLU
     - Kernel Initializer: HeNormal
   - **MaxPooling2D Layer 3**:
     - Pool Size: 5

3. **Flatten Layer**:
   - Flattens the input.

4. **Dense Layers**:
   - **Dense Layer 1**:
     - Units: 64
     - Activation: ReLU
     - Kernel Initializer: HeNormal
   - **Dense Layer 2**:
     - Units: 64
     - Activation: ReLU
     - Kernel Initializer: HeNormal
   - **Dense Layer 3**:
     - Units: 128
     - Activation: ReLU
     - Kernel Initializer: HeNormal

5. **Output Layer**:
   - **Dense Layer**:
     - Units: Number of action permutations
     - Activation: Linear
     - Kernel Initializer: HeNormal

### Model Compilation

- **Optimizer**: Adam
- **Learning Rate**: Defined by `LEARNING_RATE`
- **Loss Function**: Mean Squared Error (MSE)

### Additional Details

- Two models are defined: `running_model` and `target_model`.
- The `target_model` is initialized with the weights of the `running_model`.
- Debugging information is printed based on the `debug_level`.
