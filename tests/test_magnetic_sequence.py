import torch

from sura.models.magnetic_sequence_cnn import MagSequenceMatcher, heteroscedastic_nll


def test_magnetic_model_output_and_loss_are_finite():
    model = MagSequenceMatcher()
    sequence = torch.randn(3, 84, 4)
    truth = torch.randn(3, 2)
    position, log_variance = model(sequence)
    loss = heteroscedastic_nll(position, log_variance, truth)

    assert position.shape == (3, 2)
    assert log_variance.shape == (3, 1)
    assert torch.isfinite(loss)
