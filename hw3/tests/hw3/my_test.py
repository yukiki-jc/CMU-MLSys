import sys

sys.path.append("./python")
from needle import backend_ndarray as nd
import numpy as np
matmul_dims = [
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]

def test_matmul(m, n, p, device):
    _A = np.random.randint(0, 5, (m, n))
    _B = np.random.randint(0, 5, (n, p))
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    print(f"A: {A}")
    print(f"B: {B}") 
    
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)

test_matmul(4, 4, 4, nd.cpu())