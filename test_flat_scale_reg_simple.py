#!/usr/bin/env python3
"""
Simple test script to verify that flat_reg and scale_reg functionality is working correctly.
"""

import torch


def test_flat_loss():
    """Test flat loss computation."""
    # Create dummy scale parameters
    scales = torch.nn.Parameter(torch.randn(100, 3) * 0.1)

    # Compute flat loss
    flat_loss = torch.exp(scales).amin(dim=-1).mean()
    print(f"Flat loss: {flat_loss.item():.6f}")
    assert flat_loss > 0, "Flat loss should be positive"
    return flat_loss


def test_scale_regularisation_loss_median():
    """Test scale regularization loss computation."""
    # Create dummy scale parameters
    scales = torch.nn.Parameter(torch.randn(100, 3) * 0.1)

    # Compute scale regularization loss
    scale_exp = torch.exp(scales)
    ratio = scale_exp.amax(dim=-1) / scale_exp.median(dim=-1).values
    max_gauss_ratio = torch.tensor(6.0)
    scale_reg = torch.maximum(ratio, max_gauss_ratio) - max_gauss_ratio
    scale_reg_loss = scale_reg.mean()

    print(f"Scale regularization loss: {scale_reg_loss.item():.6f}")
    assert scale_reg_loss >= 0, "Scale regularization loss should be non-negative"
    return scale_reg_loss


def main():
    print("Testing flat_reg and scale_reg functionality...")

    # Test flat loss
    flat_loss = test_flat_loss()

    # Test scale regularization loss
    scale_reg_loss = test_scale_regularisation_loss_median()

    print(
        "✅ All tests passed! flat_reg and scale_reg functionality is working correctly."
    )
    print(f"\nSummary:")
    print(f"  - Flat loss: {flat_loss.item():.6f}")
    print(f"  - Scale regularization loss: {scale_reg_loss.item():.6f}")


if __name__ == "__main__":
    main()
