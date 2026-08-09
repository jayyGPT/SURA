import torch

from sura.fusion import AnomalyDualKalmanNet, MagneticAnomalyMap, WiFiOnlyKalmanNet


def test_wifi_only_fusion_shape():
    model = WiFiOnlyKalmanNet(hidden_size=8)
    motion = torch.zeros(2, 5, 2)
    wifi = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, 1)
    output = model(motion, wifi, mask)
    assert output.shape == (2, 5, 2)
    assert torch.isfinite(output).all()


def test_legacy_anomaly_fusion_shape():
    values = torch.zeros(4, 4)
    magnetic_map = MagneticAnomalyMap(
        values=values,
        gradient_x=torch.ones_like(values),
        gradient_y=torch.ones_like(values),
        x0=0.0,
        y0=0.0,
        cell=1.0,
    )
    model = AnomalyDualKalmanNet(magnetic_map, hidden_size=8)
    motion = torch.zeros(2, 5, 2)
    wifi = torch.zeros(2, 5, 2)
    wifi_mask = torch.ones(2, 5, 1)
    magnetic = torch.zeros(2, 5, 1)
    start = torch.zeros(2, 2)
    output = model(motion, wifi, wifi_mask, magnetic, start)
    assert output.shape == (2, 5, 2)
    assert torch.isfinite(output).all()
