import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from claraenc.IndexCoder import DiffArray

def test_real_data():
    test = np.array([10,20,30,50,70,100])
    d = DiffArray(test)
    restored = d.restore()
    assert np.array_equal(d.diffs, [10,20,30])
    assert np.array_equal(d.index, [0,0,0,1,1,2])
    assert np.array_equal(restored, test)


def test_hermes_weights(hermes_weights):
    d = DiffArray(hermes_weights)
    restored = d.restore()
    assert np.array_equal(restored, hermes_weights)
