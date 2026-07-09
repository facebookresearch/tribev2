import torch
from torch import nn

from tribev2.attribution import (
    integrated_gradients_attribution,
    occlusion_attribution,
    select_output,
)


class Batch:
    def __init__(self, data):
        self.data = data


class ToyModel(nn.Module):
    def forward(self, batch, pool_outputs=True):
        text = batch.data["text"].sum(dim=1, keepdim=True)
        audio = 2.0 * batch.data["audio"].sum(dim=1, keepdim=True)
        return text + audio


def test_select_output_reduces_to_batch_scores():
    output = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    scores = select_output(output, target_vertices=[1, 2], target_timesteps=[0, 3])

    expected = output[:, [1, 2]][:, :, [0, 3]].mean(dim=(1, 2))
    assert torch.equal(scores, expected)


def test_integrated_gradients_returns_modality_time_scores():
    text = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    audio = torch.tensor([[[1.0, 0.0, 2.0]]])
    batch = Batch({"text": text, "audio": audio})

    scores = integrated_gradients_attribution(
        ToyModel(),
        batch,
        modalities=["text", "audio"],
        n_steps=8,
    )

    assert torch.allclose(scores["text"], torch.tensor([[5.0 / 3, 7.0 / 3, 3.0]]))
    assert torch.allclose(scores["audio"], torch.tensor([[2.0 / 3, 0.0, 4.0 / 3]]))
    assert batch.data["text"] is text
    assert batch.data["audio"] is audio


def test_integrated_gradients_can_target_output_timestep():
    text = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    audio = torch.zeros(1, 1, 3)
    batch = Batch({"text": text, "audio": audio})

    scores = integrated_gradients_attribution(
        ToyModel(),
        batch,
        modalities=["text"],
        target_timesteps=[2],
        n_steps=4,
    )

    assert torch.allclose(scores["text"], torch.tensor([[0.0, 0.0, 9.0]]))


def test_occlusion_returns_positive_drop_for_supporting_features():
    text = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    audio = torch.tensor([[[1.0, 0.0, 2.0]]])
    batch = Batch({"text": text, "audio": audio})

    scores = occlusion_attribution(
        ToyModel(),
        batch,
        modalities=["text", "audio"],
        window=1,
        stride=1,
    )

    assert torch.allclose(scores["text"], torch.tensor([[5.0 / 3, 7.0 / 3, 3.0]]))
    assert torch.allclose(scores["audio"], torch.tensor([[2.0 / 3, 0.0, 4.0 / 3]]))
    assert batch.data["text"] is text
    assert batch.data["audio"] is audio
