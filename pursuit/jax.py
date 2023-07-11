import jax.numpy as jnp
import jax
import numpy as np
import copy
from functools import partial


class ErrorHashableJAX:
    def __init__(self, input_array):
        self._array = np.array(input_array, float)

    def __getattr__(self, attr):
        return getattr(self._array, attr)

    def __getitem__(self, key):
        return self._array.__getitem__(key)

    def __setitem__(self, key, value):
        return self._array.__setitem__(key, value)

    def __hash__(self):
        return id(self)


# MP Class


@jax.jit
def norm_dot(x, y):
    return abs(jnp.dot(x, y)) / jnp.linalg.norm(x)


class MP_JAX:
    @partial(jax.jit, static_argnums=(0, 3))
    def fit(self, X, D, MAX_ITER, EPS):
        r = copy.deepcopy(X)
        k = 0
        previous = jnp.zeros_like(X)
        row, col = D.shape
        alpha = jnp.zeros(col)
        while k < MAX_ITER:
            previous = r
            # compute scalar products between reduce signal and each normalized atom.
            # using list comprehension
            # M = [abs(jnp.matmul(jnp.transpose(D[:, j]), r)) / jnp.linalg.norm(D[:, j]) for j in range(0, col)]
            # using vmap
            vmapped_vdot = jax.pmap(norm_dot, (1, None), devices=1)
            M = vmapped_vdot(D, r)
            # M = jnp.array(M)
            l = jnp.argmax(M)
            # z = jnp.matmul(jnp.transpose(D[:, l]), r) / (jnp.linalg.norm(D[:, l])) ** 2
            z = jnp.dot(D[:, l], r) / (jnp.linalg.norm(D[:, l])) ** 2
            # alpha[l] = alpha[l] + z
            alpha = alpha.at[l].set(alpha[l] + z)
            r = r - z * D[:, l]
            k = k + 1
        return alpha


class MP_recursive_JAX:
    @partial(jax.jit, static_argnums=(0, 3))
    def fit(self, X, D, MAX_ITER, EPS):
        r = copy.deepcopy(X)
        previous = jnp.zeros_like(X)
        row, col = D.shape
        alpha = jnp.zeros(col)
        k = 0

        def recurse(r, D, alpha, k):
            previous = r
            vmapped_vdot = jax.pmap(norm_dot, (1, None), devices=1)
            M = vmapped_vdot(D, r)
            l = jnp.argmax(M)
            z = jnp.dot(D[:, l], r) / (jnp.linalg.norm(D[:, l])) ** 2
            alpha = alpha.at[l].set(alpha[l] + z)
            r = r - z * D[:, l]
            k = k + 1
            if k < MAX_ITER:
                return recurse(r, D, alpha, k)
            else:
                return alpha

        alpha = recurse(r, D, jnp.zeros(col), k)
        return alpha


# OMP Class


class OMP_JAX:
    @partial(jax.jit, static_argnums=(0, 3))
    def fit(self, X, D, MAX_ITER, EPS):
        row, col = D.shape
        R = X
        # R = copy.deepcopy(X)
        # R = jax.tree_map(jnp.array, X)
        previous = 0
        k = 0
        id_alpha = jnp.zeros(col)

        Gamma = jnp.empty((row, 0))
        alpha_dict = {}
        while k < MAX_ITER:
            previous = R

            # choosing the vector with the highest projection
            M = jax.vmap(jnp.dot, (1, None))(D, R)
            # capturing the index highest atome
            l = jnp.argmax(abs(M))
            # adding the atome to the Gamma matrix
            # Gamma = Gamma + [D[:, l]]
            Gamma = jnp.hstack([Gamma, jnp.reshape(D[:, l], (row, 1))])
            # saving the index highest atome to "order" the alpha vector
            alpha_dict[l] = True
            # pre-processing Gamma for next computation
            # Gamma_ = np.transpose(np.array(Gamma))
            # solving mean-squares minimization for non-indexed alpha (nid_alpha)
            nid_alpha = (
                jnp.linalg.pinv(jnp.transpose(Gamma) @ Gamma) @ jnp.transpose(Gamma) @ X
            )
            #  indexing alpha (id_alpha)
            for i, j in zip(alpha_dict.keys(), range(0, len(nid_alpha))):
                # id_alpha[i] = nid_alpha[j]
                id_alpha = id_alpha.at[i].set(nid_alpha[j])
            # updating R
            R = X - Gamma @ nid_alpha
            k = k + 1
        return id_alpha


class MP_recursive_JAX_jit:
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def fit(self, X, D, MAX_ITER, EPS):
        row, col = D.shape
        R = copy.deepcopy(X)
        # previous = jnp.zeros_like(X)
        k = 0
        id_alpha = jnp.zeros(col)

        Gamma = np.empty((row, 0))
        alpha_dict = {}

        def recurse(R, D, Gamma, alpha_dict, id_alpha, k):
            previous = copy.deepcopy(R)
            # choosing the vector with the highest projection
            # M = jax.vmap(jnp.dot, (1, None))(D._array, R._array)
            M = [abs(np.transpose(D[:, j]) @ R) for j in range(col)]
            # capturing the index highest atome
            l = np.argmax(M)
            # adding the atome to the Gamma matrix
            # Gamma = Gamma + [D[:, l]]
            Gamma = np.hstack([Gamma, np.reshape(D._array[:, l], (row, 1))])
            # saving the index highest atome to "order" the alpha vector
            alpha_dict[l] = True
            # pre-processing Gamma for next computation
            # Gamma_ = np.transpose(np.array(Gamma))
            # solving mean-squares minimization for non-indexed alpha (nid_alpha)
            nid_alpha = (
                np.linalg.pinv(np.transpose(Gamma) @ Gamma)
                @ np.transpose(Gamma)
                @ X._array
            )
            #  indexing alpha (id_alpha)
            for i, j in zip(alpha_dict.keys(), range(0, len(nid_alpha))):
                # id_alpha[i] = nid_alpha[j]
                id_alpha = id_alpha.at[i].set(nid_alpha[j])
            # updating R
            R = X._array - Gamma @ nid_alpha
            k = k + 1
            if (
                k < MAX_ITER
                and np.linalg.norm(R - previous) > EPS
                and np.linalg.norm(R) > EPS
            ):
                return recurse(R, D, Gamma, alpha_dict, id_alpha, k)
            else:
                print(k)
                return id_alpha

        return recurse(R, D, Gamma, alpha_dict, id_alpha, k)


class OMP_recursive_JAX:
    # @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def fit(self, X, D, MAX_ITER, EPS):
        row, col = D.shape
        R = X
        # previous = jnp.zeros_like(X)
        k = 0
        id_alpha = jnp.zeros(col)

        Gamma = jnp.empty((row, 0))
        alpha_dict = {}

        def recurse(R, D, Gamma, alpha_dict, id_alpha, k):
            previous = R
            # choosing the vector with the highest projection
            M = jax.vmap(jnp.dot, (1, None))(D, R)
            # capturing the index highest atome
            l = int(jnp.argmax(abs(M)))
            # adding the atome to the Gamma matrix
            # Gamma = Gamma + [D[:, l]]
            Gamma = jnp.hstack([Gamma, jnp.reshape(D[:, l], (row, 1))])
            # saving the index highest atome to "order" the alpha vector
            alpha_dict[l] = True
            # pre-processing Gamma for next computation
            # Gamma_ = np.transpose(np.array(Gamma))
            # solving mean-squares minimization for non-indexed alpha (nid_alpha)
            nid_alpha = (
                jnp.linalg.pinv(jnp.transpose(Gamma) @ Gamma) @ jnp.transpose(Gamma) @ X
            )
            #  indexing alpha (id_alpha)
            for i, j in zip(alpha_dict.keys(), range(0, len(nid_alpha))):
                # id_alpha[i] = nid_alpha[j]
                id_alpha = id_alpha.at[i].set(nid_alpha[j])
            # updating R
            R = X - Gamma @ nid_alpha
            k = k + 1
            if (
                k < MAX_ITER
                and jnp.linalg.norm(R - previous) > EPS
                and jnp.linalg.norm(R) > EPS
            ):
                return recurse(R, D, Gamma, alpha_dict, id_alpha, k)
            else:
                return id_alpha

        return recurse(R, D, Gamma, alpha_dict, id_alpha, k)
