import numpy as np
import plotly.graph_objects as go


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


def plot_results_R(res, labels, title, legends):
    fig = go.Figure()
    N = res.shape[0] // 3
    unique_labels = np.unique(labels)
    all_col = ["blue", "red", "green"]
    for i in range(3):
        idx = np.where(labels == unique_labels[i])[0]
        fig = fig.add_trace(
            go.Scatter3d(
                x=res[idx, 0, 0],
                y=res[idx, 0, 1],
                z=res[idx, 1, 1],
                mode="markers",
                name=legends[i],
                marker=dict(size=8, color=all_col[i], opacity=0.9),
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="c"),
        width=900,
        height=700,
        autosize=False,
        margin=dict(t=30, b=0, l=0, r=0),
        template="plotly_white",
    )
    fig.write_html(f"{title}.html")
    fig.show()


def plot_results_E(res, labels, title, legends):
    fig = go.Figure()
    N = res.shape[0] // 3
    unique_labels = np.unique(labels)
    all_col = ["blue", "red", "green"]
    for i in range(3):
        idx = np.where(labels == unique_labels[i])[0]
        fig = fig.add_trace(
            go.Scatter3d(
                x=res[idx, 0],
                y=res[idx, 1],
                z=res[idx, 2],
                mode="markers",
                name=legends[i],
                marker=dict(size=8, color=all_col[i], opacity=0.9),
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        width=900,
        height=700,
        autosize=False,
        margin=dict(t=40, b=0, l=0, r=0),
        template="plotly_white",
    )
    fig.write_html(f"{title}.html")
    fig.show()
