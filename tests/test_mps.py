import torch
from services.inference.mps import SpatialMaxPool3d


def test_spatial_pool_matches_reference_3d_pool_exactly():
    torch.manual_seed(7)
    for shape in [(1, 4, 11, 44, 44), (2, 3, 5, 17, 19)]:
        x = torch.randn(shape)
        expected = torch.nn.functional.max_pool3d(x, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        actual = SpatialMaxPool3d()(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
