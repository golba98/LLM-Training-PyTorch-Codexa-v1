"""Bitwise comparison helper for small independent CPU recovery fixtures."""

import numpy as np
import torch


def assert_equal(left,right):
    if isinstance(left,torch.Tensor):
        assert left.dtype==right.dtype and left.shape==right.shape
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left,np.ndarray):
        assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            assert_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert type(left)==type(right) and len(left)==len(right)
        for a,b in zip(left,right):
            assert_equal(a,b)
    else:
        assert left==right
