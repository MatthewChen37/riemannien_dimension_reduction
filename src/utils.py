import numpy as np


def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


def multisym(A):
    """Vectorized matrix symmetrization.

    Given an array ``A`` of matrices (represented as an array of shape ``(k, n,
    n)``), returns a version of ``A`` with each matrix symmetrized, i.e.,
    every matrix ``A[i]`` satisfies ``A[i] == A[i].T``.
    """
    return 0.5 * (A + multitransp(A))


def retraction(point, tangent_vector):
    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return multisym(point + tangent_vector + tangent_vector @ p_inv_tv / 2)


def norm_SPD(point, tangent_vector):
    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return np.sqrt(
        np.tensordot(p_inv_tv, multitransp(p_inv_tv), axes=tangent_vector.ndim)
    )
