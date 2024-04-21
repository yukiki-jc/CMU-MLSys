import sys

sys.path.append("./python")
from needle import backend_ndarray as nd
import numpy as np

class _ShapeAndSlices(nd.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        return self.shape, idxs


ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))
def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides
params = {
            "lhs": ShapeAndSlices(4, 5, 6)[1:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:2, 0, 0],
        }
def test_setitem_ewise(params, device):
    lhs_shape, lhs_slices = params["lhs"]
    rhs_shape, rhs_slices = params["rhs"]
    _A = np.random.randn(*lhs_shape)
    _B = np.random.randn(*rhs_shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    start_ptr = A._handle.ptr()
    print(f"lhs {A[lhs_slices]} rhs {B[rhs_slices]}")
    A[lhs_slices] = 5
    _A[lhs_slices] = 5
    # print(f"after set item lhs {A[lhs_slices]} rhs {B[rhs_slices]}")
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    compare_strides(_A, A)
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)

test_setitem_ewise(params, nd.cuda())