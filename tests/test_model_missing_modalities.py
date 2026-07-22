import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _install_model_import_stubs() -> None:
    einops = types.ModuleType("einops")

    def rearrange(*args, **kwargs):
        raise AssertionError("rearrange should not be used in these tests")

    einops.rearrange = rearrange

    neuralset = types.ModuleType("neuralset")
    neuralset.dataloader = types.ModuleType("neuralset.dataloader")
    neuralset.dataloader.SegmentData = object

    neuraltrain = types.ModuleType("neuraltrain")
    neuraltrain.models = types.ModuleType("neuraltrain.models")
    neuraltrain.models.base = types.ModuleType("neuraltrain.models.base")
    neuraltrain.models.common = types.ModuleType("neuraltrain.models.common")
    neuraltrain.models.transformer = types.ModuleType("neuraltrain.models.transformer")

    class BaseModelConfig:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_post_init(self, __context):
            return None

    class Mlp(BaseModelConfig):
        def build(self, input_dim: int, output_dim: int) -> nn.Module:
            return nn.Linear(input_dim, output_dim, bias=False)

    class SubjectLayers(BaseModelConfig):
        def build(self, in_channels: int, out_channels: int) -> nn.Module:
            del out_channels
            return nn.Identity()

    class SubjectLayersModel(nn.Module):
        pass

    class TransformerEncoder(BaseModelConfig):
        def build(self, dim: int) -> nn.Module:
            del dim
            return nn.Identity()

    neuraltrain.models.base.BaseModelConfig = BaseModelConfig
    neuraltrain.models.common.Mlp = Mlp
    neuraltrain.models.common.SubjectLayers = SubjectLayers
    neuraltrain.models.common.SubjectLayersModel = SubjectLayersModel
    neuraltrain.models.transformer.TransformerEncoder = TransformerEncoder

    sys.modules["einops"] = einops
    sys.modules["neuralset"] = neuralset
    sys.modules["neuralset.dataloader"] = neuralset.dataloader
    sys.modules["neuraltrain"] = neuraltrain
    sys.modules["neuraltrain.models"] = neuraltrain.models
    sys.modules["neuraltrain.models.base"] = neuraltrain.models.base
    sys.modules["neuraltrain.models.common"] = neuraltrain.models.common
    sys.modules["neuraltrain.models.transformer"] = neuraltrain.models.transformer


def _load_model_module():
    _install_model_import_stubs()
    module_name = "tribev2_model_under_test"
    sys.modules.pop(module_name, None)
    module_path = Path(__file__).resolve().parents[1] / "tribev2" / "model.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyProjector:
    def build(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Linear(input_dim, output_dim, bias=False)


class DummyCombiner:
    def build(self, input_dim: int, output_dim: int) -> nn.Module:
        assert input_dim == output_dim
        return nn.Identity()


class DummySubjectLayers:
    def build(self, in_channels: int, out_channels: int) -> nn.Module:
        del in_channels, out_channels
        return nn.Identity()


def _make_config(extractor_aggregation: str):
    return SimpleNamespace(
        projector=DummyProjector(),
        combiner=DummyCombiner(),
        encoder=None,
        time_pos_embedding=False,
        subject_embedding=False,
        subject_layers=DummySubjectLayers(),
        hidden=12,
        max_seq_len=32,
        dropout=0.0,
        extractor_aggregation=extractor_aggregation,
        layer_aggregation="mean",
        linear_baseline=True,
        modality_dropout=0.0,
        temporal_dropout=0.0,
        low_rank_head=None,
        temporal_smoothing=None,
    )


def _make_batch(**modalities):
    return SimpleNamespace(data=modalities)


def _project_modality(model, batch, modality: str) -> torch.Tensor:
    data = batch.data[modality].to(torch.float32)
    if data.ndim == 3:
        data = data.unsqueeze(1)
    data = data.mean(dim=1).transpose(1, 2)
    return model.projectors[modality](data)


def test_cat_aggregation_uses_zero_fill_with_missing_modalities():
    module = _load_model_module()
    model = module.FmriEncoderModel(
        feature_dims={"text": (2, 4), "audio": (2, 4), "video": (2, 4)},
        n_outputs=5,
        n_output_timesteps=6,
        config=_make_config("cat"),
    )
    batch = _make_batch(text=torch.randn(2, 2, 4, 5))

    aggregated = model.aggregate_features(batch)

    assert aggregated.shape == (2, 5, 12)
    text_dim = model.projector_output_dims["text"]
    audio_dim = model.projector_output_dims["audio"]
    assert torch.count_nonzero(aggregated[:, :, :text_dim]) > 0
    assert torch.equal(
        aggregated[:, :, text_dim : text_dim + audio_dim],
        torch.zeros_like(aggregated[:, :, text_dim : text_dim + audio_dim]),
    )
    assert torch.equal(
        aggregated[:, :, text_dim + audio_dim :],
        torch.zeros_like(aggregated[:, :, text_dim + audio_dim :]),
    )


def test_sum_aggregation_matches_present_modality_projection_when_others_missing():
    module = _load_model_module()
    model = module.FmriEncoderModel(
        feature_dims={"text": (2, 4), "audio": (2, 4), "video": (2, 4)},
        n_outputs=5,
        n_output_timesteps=6,
        config=_make_config("sum"),
    )
    batch = _make_batch(text=torch.randn(2, 2, 4, 5))

    aggregated = model.aggregate_features(batch)
    expected = _project_modality(model, batch, "text")

    assert aggregated.shape == (2, 5, 12)
    assert torch.allclose(aggregated, expected)


def test_stack_aggregation_keeps_missing_modalities_as_zero_time_blocks():
    module = _load_model_module()
    model = module.FmriEncoderModel(
        feature_dims={"text": (2, 4), "audio": (2, 4), "video": (2, 4)},
        n_outputs=5,
        n_output_timesteps=6,
        config=_make_config("stack"),
    )
    batch = _make_batch(audio=torch.randn(2, 2, 4, 5))

    aggregated = model.aggregate_features(batch)
    expected = _project_modality(model, batch, "audio")
    chunk = expected.shape[1]

    assert aggregated.shape == (2, chunk * 3, 12)
    assert torch.equal(aggregated[:, :chunk], torch.zeros_like(aggregated[:, :chunk]))
    assert torch.allclose(aggregated[:, chunk : 2 * chunk], expected)
    assert torch.equal(
        aggregated[:, 2 * chunk :], torch.zeros_like(aggregated[:, 2 * chunk :])
    )


def test_missing_projector_path_uses_same_fallback_width_as_present_modalities():
    module = _load_model_module()
    model = module.FmriEncoderModel(
        feature_dims={"text": (2, 4), "audio": None},
        n_outputs=5,
        n_output_timesteps=6,
        config=_make_config("sum"),
    )
    batch = _make_batch(text=torch.randn(2, 2, 4, 5))

    aggregated = model.aggregate_features(batch)
    expected = _project_modality(model, batch, "text")

    assert aggregated.shape == expected.shape == (2, 5, 12)
    assert torch.allclose(aggregated, expected)


@pytest.mark.parametrize("extractor_aggregation", ["cat", "sum", "stack"])
def test_missing_modality_paths_never_change_projected_width(extractor_aggregation: str):
    module = _load_model_module()
    model = module.FmriEncoderModel(
        feature_dims={"text": (2, 4), "audio": (2, 4), "video": None},
        n_outputs=5,
        n_output_timesteps=6,
        config=_make_config(extractor_aggregation),
    )

    if extractor_aggregation == "cat":
        expected_dim = 4
    else:
        expected_dim = 12

    assert model.projector_output_dims == {
        "text": expected_dim,
        "audio": expected_dim,
        "video": expected_dim,
    }
