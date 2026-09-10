import numpy as np
import pytest
from anndata import read_h5ad

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("umap")
import matplotlib.pyplot as plt

from scene import joint_umap, plot_joint_embedding
from test_analysis import _adata


def test_joint_umap_and_plot_round_trip(tmp_path):
    adata = _adata()
    adata.obs["type"] = ["A", "B"]
    original = adata.obsm["joint"].copy()
    xy = joint_umap(adata, key="joint", n_neighbors=3)
    assert xy.shape == (6, 2)
    np.testing.assert_array_equal(xy[:2], adata.obsm["joint_umap"])
    np.testing.assert_array_equal(xy[2:], adata.varm["joint_umap"])
    np.testing.assert_array_equal(original, adata.obsm["joint"])
    assert adata.uns["joint"]["projection"]["metric"] == "euclidean"
    adata.write_h5ad(tmp_path / "projected.h5ad")
    saved = read_h5ad(tmp_path / "projected.h5ad")
    figure, ax = plt.subplots()
    assert plot_joint_embedding(saved, key="joint", color="type", genes=["a", "c"], ax=ax) is ax
    assert [text.get_text() for text in ax.texts] == ["a", "c"]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), xy[:1])
    figure.savefig(tmp_path / "joint.png")
    plt.close(figure)


def test_plot_validation_and_numeric_color():
    adata = _adata()
    with pytest.raises(ValueError, match="joint_umap"):
        plot_joint_embedding(adata, key="joint")
    # Plotting consumes stored coordinates and never recomputes the projection.
    adata.obsm["joint_umap"] = adata.obsm["joint"][:, :2]
    adata.varm["joint_umap"] = adata.varm["joint"][:, :2]
    adata.uns["joint"] = {"projection": {"key": "joint_umap"}}
    adata.obs["number"] = [1., 2.]
    ax = plot_joint_embedding(adata, key="joint", color="number", genes="a")
    assert len(ax.figure.axes) == 2
    plt.close(ax.figure)
    with pytest.raises(KeyError):
        plot_joint_embedding(adata, key="joint", color="unknown")
    with pytest.raises(KeyError):
        plot_joint_embedding(adata, key="joint", genes="unknown")


@pytest.mark.parametrize("options", [{"n_neighbors": 1}, {"min_dist": -1}, {"min_dist": 2}])
def test_projection_validation(options):
    with pytest.raises(ValueError):
        joint_umap(_adata(), key="joint", **options)


def test_shared_axes_can_be_rendered():
    adata = _adata()
    adata.obsm["joint_umap"] = adata.obsm["joint"][:, :2]
    adata.varm["joint_umap"] = adata.varm["joint"][:, :2]
    adata.uns["joint"] = {"projection": {"key": "joint_umap"}}
    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax in axes:
        plot_joint_embedding(adata, key="joint", genes=["a", "b"], ax=ax)
    figure.canvas.draw()
    plt.close(figure)


def test_nearby_gene_labels_do_not_overlap():
    adata = _adata()
    adata.obsm["joint_umap"] = np.array([[0., 0.], [4., 2.]])
    adata.varm["joint_umap"] = np.array([[1., 1.], [1.01, 1.01], [1.02, 1.02], [1.03, 1.03]])
    adata.uns["joint"] = {"projection": {"key": "joint_umap"}}
    ax = plot_joint_embedding(adata, key="joint", genes=adata.var_names)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in ax.texts]
    for i, box in enumerate(boxes):
        assert all(not box.overlaps(other) for other in boxes[:i])
    plt.close(ax.figure)
