"""
Unit tests for video compression model components.
Run with: python -m pytest tests/test_models.py -v
"""

import sys
import os

# Skip if torch is not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, skipping tests")

if TORCH_AVAILABLE:
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.mapping_net import (
        GatedFusionModule,
        ConditionFusionNet,
        NullTextNet,
        MappingNet
    )


def test_gated_fusion_module():
    """Test GatedFusionModule forward pass."""
    if not TORCH_AVAILABLE:
        return
    
    batch_size = 4
    feature_dim = 1024
    hidden_dim = 1024
    
    module = GatedFusionModule(feature_dim, hidden_dim)
    
    # Test with 2D input (pooled features)
    f_a = torch.randn(batch_size, feature_dim)
    f_b = torch.randn(batch_size, feature_dim)
    
    output = module(f_a, f_b)
    
    assert output.shape == (batch_size, hidden_dim), \
        f"Expected shape ({batch_size}, {hidden_dim}), got {output.shape}"
    
    # Test with 3D input (token features)
    num_tokens = 257
    f_a_3d = torch.randn(batch_size, num_tokens, feature_dim)
    f_b_3d = torch.randn(batch_size, num_tokens, feature_dim)
    
    output_3d = module(f_a_3d, f_b_3d)
    
    assert output_3d.shape == (batch_size, hidden_dim), \
        f"Expected shape ({batch_size}, {hidden_dim}), got {output_3d.shape}"
    
    print("✓ GatedFusionModule tests passed")


def test_condition_fusion_net():
    """Test ConditionFusionNet forward pass."""
    if not TORCH_AVAILABLE:
        return
    
    batch_size = 4
    feature_dim = 1024
    seq_len = 77
    embed_dim = 1024
    rank = 8
    
    net = ConditionFusionNet(
        feature_dim=feature_dim,
        hidden_dim=1024,
        seq_len=seq_len,
        embed_dim=embed_dim,
        rank=rank
    )
    
    f_a = torch.randn(batch_size, feature_dim)
    f_b = torch.randn(batch_size, feature_dim)
    
    U, V, condition = net(f_a, f_b)
    
    assert U.shape == (batch_size, seq_len, rank), \
        f"U: Expected shape ({batch_size}, {seq_len}, {rank}), got {U.shape}"
    
    assert V.shape == (batch_size, rank, embed_dim), \
        f"V: Expected shape ({batch_size}, {rank}, {embed_dim}), got {V.shape}"
    
    assert condition.shape == (batch_size, seq_len, embed_dim), \
        f"condition: Expected shape ({batch_size}, {seq_len}, {embed_dim}), got {condition.shape}"
    
    # Verify condition = U @ V / sqrt(rank)
    expected_condition = torch.bmm(U, V) / (rank ** 0.5)
    assert torch.allclose(condition, expected_condition, atol=1e-5), \
        "Condition matrix should equal U @ V / sqrt(rank)"
    
    print("✓ ConditionFusionNet tests passed")


def test_null_text_net():
    """Test NullTextNet forward pass."""
    if not TORCH_AVAILABLE:
        return
    
    batch_size = 4
    feature_dim = 1024
    seq_len = 77
    embed_dim = 1024
    
    net = NullTextNet(
        feature_dim=feature_dim,
        hidden_dim=1024,
        seq_len=seq_len,
        embed_dim=embed_dim
    )
    
    # Test with 2D input
    f_a = torch.randn(batch_size, feature_dim)
    null_text = net(f_a)
    
    assert null_text.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape ({batch_size}, {seq_len}, {embed_dim}), got {null_text.shape}"
    
    # Test with 3D input (should use CLS token)
    num_tokens = 257
    f_a_3d = torch.randn(batch_size, num_tokens, feature_dim)
    null_text_3d = net(f_a_3d)
    
    assert null_text_3d.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape ({batch_size}, {seq_len}, {embed_dim}), got {null_text_3d.shape}"
    
    print("✓ NullTextNet tests passed")


def test_mapping_net():
    """Test complete MappingNet forward pass."""
    if not TORCH_AVAILABLE:
        return
    
    batch_size = 4
    feature_dim = 1024
    seq_len = 77
    embed_dim = 1024
    rank = 8
    
    net = MappingNet(
        feature_dim=feature_dim,
        hidden_dim=1024,
        seq_len=seq_len,
        embed_dim=embed_dim,
        rank=rank
    )
    
    f_a = torch.randn(batch_size, feature_dim)
    f_b = torch.randn(batch_size, feature_dim)
    
    U, V, condition, null_text = net(f_a, f_b)
    
    assert U.shape == (batch_size, seq_len, rank)
    assert V.shape == (batch_size, rank, embed_dim)
    assert condition.shape == (batch_size, seq_len, embed_dim)
    assert null_text.shape == (batch_size, seq_len, embed_dim)
    
    assert net.get_rank() == rank
    
    print("✓ MappingNet tests passed")


def test_gradient_flow():
    """Test that gradients flow correctly through MappingNet."""
    if not TORCH_AVAILABLE:
        return
    
    batch_size = 2
    feature_dim = 1024
    rank = 8
    
    net = MappingNet(
        feature_dim=feature_dim,
        hidden_dim=512,  # Smaller for faster test
        seq_len=77,
        embed_dim=1024,
        rank=rank
    )
    
    f_a = torch.randn(batch_size, feature_dim, requires_grad=True)
    f_b = torch.randn(batch_size, feature_dim, requires_grad=True)
    
    U, V, condition, null_text = net(f_a, f_b)
    
    # Compute a dummy loss
    loss = condition.mean() + null_text.mean()
    loss.backward()
    
    # Check gradients exist
    for name, param in net.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ Gradient flow tests passed")


def test_low_rank_compression():
    """Test that low-rank decomposition reduces parameters."""
    if not TORCH_AVAILABLE:
        return
    
    seq_len = 77
    embed_dim = 1024
    rank = 8
    
    # Full matrix parameters
    full_params = seq_len * embed_dim  # 77 * 1024 = 78,848
    
    # Low-rank parameters
    low_rank_params = seq_len * rank + rank * embed_dim  # 77*8 + 8*1024 = 616 + 8192 = 8,808
    
    compression_ratio = full_params / low_rank_params
    
    print(f"Full matrix params: {full_params:,}")
    print(f"Low-rank params (r={rank}): {low_rank_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    assert compression_ratio > 8, "Low-rank should provide significant compression"
    
    print("✓ Low-rank compression tests passed")


def run_all_tests():
    """Run all tests."""
    print("\nRunning Video Compression Model Tests\n" + "=" * 50)
    
    if not TORCH_AVAILABLE:
        print("Skipping tests - PyTorch not available")
        return
    
    test_gated_fusion_module()
    test_condition_fusion_net()
    test_null_text_net()
    test_mapping_net()
    test_gradient_flow()
    test_low_rank_compression()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
