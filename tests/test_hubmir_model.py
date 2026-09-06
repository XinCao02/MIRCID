import torch

from mircid.hubmir.model import HubmiRNet


def test_hubmir_shape() -> None:
    model = HubmiRNet(input_size=7, hidden_size=16, output_size=5, num_blocks=2, dropout=0.1)
    model.eval()
    assert model(torch.zeros(3, 7)).shape == (3, 5)

