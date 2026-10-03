import torch
import numpy as np
from src.model import FraudMLP, state_to_numpy, numpy_to_state, clone_state

def test_fraud_mlp_forward():
    model = FraudMLP(n_features=10, hidden_dim=16)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5,) # Squeezed output

def test_state_to_numpy_and_numpy_to_state():
    model1 = FraudMLP(n_features=5, hidden_dim=8)
    model2 = FraudMLP(n_features=5, hidden_dim=8)
    
    state1 = state_to_numpy(model1)
    assert isinstance(state1, list)
    assert isinstance(state1[0], np.ndarray)
    
    # Load into model2
    numpy_to_state(model2, state1)
    
    state2 = state_to_numpy(model2)
    
    # Verify arrays are identical
    for a1, a2 in zip(state1, state2):
        np.testing.assert_array_equal(a1, a2)

def test_clone_state():
    model = FraudMLP(n_features=5, hidden_dim=8)
    state = state_to_numpy(model)
    cloned = clone_state(state)
    
    for s, c in zip(state, cloned):
        np.testing.assert_array_equal(s, c)
        assert s is not c # Must be different objects in memory
