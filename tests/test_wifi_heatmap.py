import numpy as np
import torch

from sura.models.wifi_heatmap import Grid, WifiHeatmapNet, encode_wifi, heatmap_covariance, soft_argmax


def test_wifi_encoding_and_heatmap_shapes():
    raw = np.array([[-100.0, -90.0, -60.0, -30.0]])
    encoded = encode_wifi(raw)
    np.testing.assert_allclose(encoded, [[0.0, 0.0, 0.5, 1.0]])

    grid = Grid(np.array([0.0, 1.0]), np.array([0.0, 1.0]), cell=1.0)
    model = WifiHeatmapNet(4, grid.n_cells)
    logits = model(torch.tensor(encoded))
    coordinates = torch.tensor(grid.coords, dtype=torch.float32)
    position = soft_argmax(logits, coordinates)
    covariance = heatmap_covariance(logits, coordinates)

    assert logits.shape == (1, grid.n_cells)
    assert position.shape == (1, 2)
    assert covariance.shape == (1, 2, 2)
    assert torch.isfinite(position).all()
