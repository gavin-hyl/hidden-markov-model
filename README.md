# Hidden Markov Models in Python

A comprehensive implementation of Hidden Markov Models (HMMs) and their associated algorithms for sequence modeling, pattern recognition, and probabilistic inference.

## Overview

This project provides a complete, from-scratch implementation of Hidden Markov Models in Python, featuring all major algorithms for training, inference, and sequence generation. The implementation supports both supervised and unsupervised learning paradigms and includes utilities for data preprocessing and model evaluation.

Hidden Markov Models are powerful statistical models for sequential data where the underlying system is assumed to be a Markov process with unobserved (hidden) states. This implementation is designed for educational purposes and practical applications in bioinformatics, natural language processing, speech recognition, and time series analysis.

## Architecture

### Model Specifications
- **States**: Discrete hidden states (N configurable)
- **Observations**: Discrete observation symbols (V configurable)  
- **Transitions**: Full N×N transition matrix with start/end state support
- **Emissions**: N×V observation probability matrix
- **Flexibility**: Supports variable-length sequences and custom alphabets

### Core Components
- **Transition Matrix (A)**: P(state_t+1 | state_t) - governs state transitions
- **Observation Matrix (O)**: P(observation_t | state_t) - governs emissions
- **Initial Distribution (π)**: P(state_1) - starting state probabilities
- **End Probabilities**: P(end | state_t) - termination probabilities

## Key Algorithms

### 1. Viterbi Algorithm
**Purpose**: Find the most likely sequence of hidden states

```python
# Find optimal state sequence for given observations
optimal_states = hmm.viterbi(observations)
```

- **Time Complexity**: O(N²T) where T is sequence length
- **Space Complexity**: O(NT)
- **Applications**: Speech recognition, gene finding, part-of-speech tagging

### 2. Forward Algorithm
**Purpose**: Compute forward probabilities for sequence likelihood

```python
# Calculate forward probabilities
alphas = hmm.forward(observations, normalize=True)
```

- **Computes**: P(observations_1:t, state_t = j) for all t, j
- **Numerical Stability**: Optional normalization to prevent underflow
- **Applications**: Likelihood computation, filtering

### 3. Backward Algorithm  
**Purpose**: Compute backward probabilities for sequence analysis

```python
# Calculate backward probabilities
betas = hmm.backward(observations, normalize=True)
```

- **Computes**: P(observations_t+1:T | state_t = j) for all t, j
- **Complements**: Forward algorithm for complete probability calculations
- **Applications**: Smoothing, parameter estimation

### 4. Forward-Backward Algorithm
**Purpose**: Combine forward and backward for optimal inference

- **Posterior Probabilities**: P(state_t = j | observations_1:T)
- **Smoothing**: Optimal state estimation using all available data
- **Foundation**: Basis for Baum-Welch parameter estimation

### 5. Supervised Learning
**Purpose**: Train model parameters from labeled data

```python
# Train with known state sequences
hmm.supervised_learning(observation_sequences, state_sequences)
```

- **Maximum Likelihood**: Closed-form parameter estimation
- **Requirements**: Paired observation and state sequences
- **Speed**: Direct computation without iterative optimization

### 6. Unsupervised Learning (Baum-Welch)
**Purpose**: Train model parameters from unlabeled data

```python
# Train with only observations using EM algorithm
hmm.unsupervised_learning(observation_sequences, n_iterations)
```

- **EM Algorithm**: Expectation-Maximization for parameter estimation
- **E-Step**: Compute expected sufficient statistics using Forward-Backward
- **M-Step**: Update parameters to maximize expected log-likelihood
- **Convergence**: Iterative improvement until local optimum

## Core Features

### Data Flexibility
- **Variable-Length Sequences**: Handles sequences of different lengths
- **Custom Alphabets**: Maps arbitrary objects to internal indices
- **Batch Processing**: Efficient training on multiple sequences
- **Missing Data**: Robust handling of incomplete observations

### Numerical Stability
- **Log-Space Computation**: Prevents numerical underflow
- **Normalization Options**: Configurable probability normalization
- **Robust Implementation**: Handles edge cases and zero probabilities

### Model Generation
- **Sequence Sampling**: Generate synthetic sequences from trained models
- **Controllable Generation**: Optional end-state termination
- **Statistical Validation**: Verify model quality through generation

## Implementation Details

### Class Structure
```python
class HiddenMarkovModel:
    def __init__(self, A=None, O=None, A_start=None, A_end=None, 
                 X_obj_to_idx=None, Y_obj_to_idx=None, N=1, V=1)
```

### Key Methods
- `viterbi(x)`: Most likely state sequence
- `forward(x, normalize=False)`: Forward probabilities
- `backward(x, normalize=False)`: Backward probabilities  
- `supervised_learning(X, Y)`: Train from labeled data
- `unsupervised_learning(X, N_iters)`: Train from unlabeled data
- `generate_emission(M, use_end=False)`: Generate synthetic sequences
- `probability_alphas(x)`: Sequence likelihood via forward algorithm
- `probability_betas(x)`: Sequence likelihood via backward algorithm

## Utility Functions

### Data Preprocessing
```python
# Convert object sequences to indexed sequences
indexed_seqs, obj_to_idx = index_sequence(sequences)

# Invert mapping dictionaries
idx_to_obj = invert_map(obj_to_idx)

# Train supervised HMM with automatic setup
hmm = supervised_HMM(observations, states)

# Train unsupervised HMM with specified parameters
hmm = unsupervised_HMM(observations, n_states, n_iterations)
```

### Features
- **Automatic Indexing**: Converts arbitrary objects to numerical indices
- **Bidirectional Mapping**: Maintains both forward and reverse mappings
- **Model Factories**: Simplified model creation and training
- **Type Flexibility**: Works with any hashable objects as symbols

## File Structure

```
hidden-markov-model/
├── README.md           # This comprehensive documentation
├── hmm.py             # Core HMM implementation (411 lines)
├── hmm_utils.py       # Utility functions and helpers (96 lines)
├── examples.ipynb     # Practical usage examples and demonstrations
├── .gitignore         # Git ignore patterns
└── .git/              # Git repository metadata
```

## Usage Examples

### Basic Supervised Learning
```python
import hmm_utils

# Prepare training data
observations = [['a', 'b', 'c'], ['a', 'c', 'b']]
states = [[0, 1, 2], [0, 2, 1]]

# Train model
hmm = hmm_utils.supervised_HMM(observations, states)

# Make predictions
predicted_states = hmm.viterbi(['a', 'b', 'c'])
```

### Unsupervised Learning
```python
# Train on unlabeled sequences
observations = [['sunny', 'walk'], ['rainy', 'shop'], ['sunny', 'walk']]
hmm = hmm_utils.unsupervised_HMM(observations, n_states=2, N_iters=100)

# Generate new sequences
new_sequence = hmm.generate_emission(10, use_end=True)
```

### Advanced Analysis
```python
# Compute sequence probabilities
likelihood = hmm.probability_alphas(observations)
posterior_probs = hmm.forward(observations) * hmm.backward(observations)

# State sequence analysis
most_likely_states = hmm.viterbi(observations)
```

## Applications

### Bioinformatics
- **Gene Finding**: Identify coding regions in DNA sequences
- **Protein Structure**: Predict secondary structure from amino acid sequences
- **Sequence Alignment**: Profile HMMs for multiple sequence alignment

### Natural Language Processing
- **Part-of-Speech Tagging**: Assign grammatical categories to words
- **Named Entity Recognition**: Identify and classify entities in text
- **Language Modeling**: Capture sequential dependencies in text

### Speech Recognition
- **Phoneme Recognition**: Model acoustic features to phoneme mappings
- **Word Segmentation**: Identify word boundaries in continuous speech
- **Speaker Modeling**: Capture speaker-specific characteristics

### Time Series Analysis
- **Regime Detection**: Identify changing market conditions
- **Anomaly Detection**: Spot unusual patterns in sequential data
- **Forecasting**: Predict future values based on hidden states

## Technical Requirements

### Dependencies
- **NumPy**: Efficient numerical computations and matrix operations
- **Python 3.7+**: Modern Python features and type hints support

### Performance Characteristics
- **Training Time**: O(N²T) per iteration for unsupervised learning
- **Memory Usage**: O(NT) for sequence processing
- **Scalability**: Efficient for moderate-sized problems (N < 100, T < 1000)

## Educational Value

This implementation serves as an excellent learning resource for:
- **Probabilistic Modeling**: Understanding sequential data modeling
- **Algorithm Implementation**: Translating mathematical concepts to code
- **Numerical Methods**: Handling floating-point precision and stability
- **Machine Learning**: Supervised vs. unsupervised learning paradigms
- **Dynamic Programming**: Efficient algorithms for sequence problems

## Advanced Features

### Numerical Stability
- **Log-Space Computation**: Prevents underflow in long sequences
- **Normalization**: Optional probability normalization for stability
- **Zero Handling**: Robust treatment of impossible transitions/emissions

### Extensibility
- **Modular Design**: Easy to extend with new algorithms
- **Clean Interfaces**: Well-defined API for integration
- **Documentation**: Comprehensive docstrings and examples

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install numpy
   ```

2. **Basic Usage**:
   ```python
   from hmm import HiddenMarkovModel
   import hmm_utils
   
   # Create and train a model
   hmm = hmm_utils.supervised_HMM(observations, states)
   
   # Make predictions
   predictions = hmm.viterbi(new_observations)
   ```

3. **Explore Examples**:
   - Open `examples.ipynb` in Jupyter
   - Run provided examples and experiments
   - Modify parameters to understand behavior

---

*This implementation provides a solid foundation for understanding and applying Hidden Markov Models across diverse domains, from computational biology to natural language processing.*
