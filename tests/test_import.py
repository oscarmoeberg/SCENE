import os
from pathlib import Path
import subprocess
import sys


def test_public_api_exports():
    import scene
    assert set(scene.__all__) == {
        "SCENE", "nearest_genes", "nearest_cells", "cell_gene_distances",
        "laplacian_init", "joint_umap", "plot_joint_embedding",
    }
    for name in ("fit_scene", "train_scene", "rank_genes"):
        assert not hasattr(scene, name)
    from scene import train
    assert not hasattr(train, "fit_scene") and not hasattr(train, "train_scene")


def test_core_works_without_plotting_dependencies():
    code = '''
import sys
from importlib.abc import MetaPathFinder
class BlockPlotting(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"matplotlib", "umap", "scanpy"}:
            raise ImportError("optional dependency deliberately unavailable")
sys.meta_path.insert(0, BlockPlotting())
from scene import SCENE, joint_umap
from anndata import AnnData
import numpy as np
adata = AnnData(np.ones((4, 3)))
model = SCENE(latent_dim=2).fit(adata, layer=None, epochs=1, print_every=None)
assert len(model.expected_counts(cells="0", genes="1")) == 1
try:
    joint_umap(adata)
except ImportError as error:
    assert "plotting" in str(error)
else:
    raise AssertionError("Expected optional dependency error")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, text=True)
