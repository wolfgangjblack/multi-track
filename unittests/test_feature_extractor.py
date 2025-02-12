import numpy
import pytest
import torch

from project.pipeline_utils import FeatureExtractor


@pytest.fixture
def feature_extractor():
    return FeatureExtractor(embedding_dim=128)


def test_feature_extractor_init(feature_extractor):
    assert isinstance(feature_extractor, torch.nn.Module)
    assert feature_extractor.embedding.out_features == 128


def test_feature_extractor_batch():
    extractor = FeatureExtractor(embedding_dim=32)
    batch_size = 4
    dummy_batch = torch.randn(batch_size, 3, 224, 224)
    if torch.backends.mps.is_available():
        dummy_batch = dummy_batch.to("mps")

    with torch.no_grad():
        features = extractor(dummy_batch)

    assert features.shape == (batch_size, 32)

    norms = torch.norm(features, dim=1)
    assert torch.allclose(
        norms, torch.ones_like(torch.tensor([1.0] * batch_size)).to("mps")
    )
