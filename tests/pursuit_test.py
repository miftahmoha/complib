import numpy as np
from pursuit import MP, OMP, StOMP, CoSaMP, BP

# Parameters
D_EVAL = np.array(
    [
        [np.sqrt(2) / 2, np.sqrt(3) / 3, np.sqrt(6) / 3, 2 / 3, -1 / 3],
        [-np.sqrt(2) / 2, -np.sqrt(3) / 3, -np.sqrt(6) / 6, 2 / 3, -2 / 3],
        [0, -np.sqrt(3) / 3, np.sqrt(6) / 6, 1 / 3, 2 / 3],
    ],
    float,
)
X_EVAL = np.array([4 / 3 - np.sqrt(2) / 2, 4 / 3 + np.sqrt(2) / 2, 2 / 3], float)
MAX_ITER = 5000
EPS = 1e-2


def test_mp():
    mp = MP()
    alpha = mp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS)
    assert np.allclose(alpha, [-1.0, 0.0, 0.0, 2.0, 0.0])


def test_omp():
    omp = OMP()
    alpha = omp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS)
    assert np.allclose(alpha, [-1.0, 0.0, 0.0, 2.0, 0.0])


def test_bp():
    bp = BP()
    alpha = bp.fit(X_EVAL, D_EVAL)
    assert np.allclose(alpha, [-1.0, 0.0, 0.0, 2.0, 0.0])


# CoSaMP & StOMP

omp = OMP()
true_alpha = omp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS)


def test_cosamp():
    cosamp = CoSaMP()
    alpha = cosamp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS, 3)
    assert np.linalg.norm(true_alpha - alpha) / np.linalg.norm(true_alpha) < 0.4


def test_stomp():
    stomp = StOMP()
    alpha = stomp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS, 1.5)
    assert np.linalg.norm(true_alpha - alpha) / np.linalg.norm(true_alpha) < 0.2
