"""Factory validation plus a GPU smoke test for the actual sparse backend."""
import pytest

from lab1.segmentation import build_model


def test_rejects_unimplemented_architecture():
    with pytest.raises(ValueError, match="PTv3"):
        build_model("ptv3")


def test_rejects_unknown_model_option():
    with pytest.raises(TypeError, match="Unexpected model options"):
        build_model(typo_channels=16)


def test_sparse_unet_backward_and_voxel_order():
    torch = pytest.importorskip("torch")
    pytest.importorskip("spconv.pytorch")
    if not torch.cuda.is_available():
        pytest.skip("sparse convolution smoke test requires CUDA")

    torch.manual_seed(7)
    model = build_model(num_classes=4, channels=(8, 16, 32)).cuda()
    # Two scenes with deliberately unsorted, unique voxel coordinates.
    axis = torch.arange(4, device="cuda", dtype=torch.int32)
    xyz = torch.cartesian_prod(axis, axis, axis)
    coords = torch.cat([
        torch.cat([torch.full((len(xyz), 1), b, device="cuda",
                              dtype=torch.int32), xyz], dim=1)
        for b in range(2)
    ])
    perm = torch.randperm(len(coords), device="cuda")
    coords = coords[perm]
    features = torch.randn(len(coords), 3, device="cuda")
    labels = torch.arange(len(coords), device="cuda") % 4

    logits = model(features, coords)
    assert logits.shape == (len(coords), 4)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, labels).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    # A permutation of inputs must cause the same permutation of outputs.
    model.eval()
    reorder = torch.randperm(len(coords), device="cuda")
    with torch.no_grad():
        reference = model(features, coords)
        reordered = model(features[reorder], coords[reorder])
    torch.testing.assert_close(reordered, reference[reorder], atol=2e-4, rtol=2e-3)
