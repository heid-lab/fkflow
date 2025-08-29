#!/usr/bin/env python3
"""
Training script for FK Steering Hypercube Benchmark

This script trains Schrödinger Bridge flow and score models for hypercube steering tasks.
"""

import argparse
from core import train_sb_models, get_device, set_random_seed


def main():
    parser = argparse.ArgumentParser(
        description="Train FK steering models for hypercube benchmark"
    )

    parser.add_argument(
        "--dimensions",
        type=str,
        default="1-10",
        help="Dimension range to train (e.g., '1-10' or '5,7,9')",
    )
    parser.add_argument(
        "--base_epochs",
        type=int,
        default=4000,
        help="Base number of epochs (will be multiplied by dimension)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="Noise level for Schrödinger Bridge"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cuda', 'mps', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--no_save", action="store_true", help="Don't save trained models"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible training"
    )

    args = parser.parse_args()

    # Set random seed for reproducible training
    if args.seed is not None:
        set_random_seed(args.seed)

    # Parse dimensions
    if "-" in args.dimensions:
        start, end = map(int, args.dimensions.split("-"))
        dimensions = list(range(start, end + 1))
    else:
        dimensions = [int(d.strip()) for d in args.dimensions.split(",")]

    # Setup device
    device = get_device(args.device)

    print(f"Using device: {device}")
    print(f"Training dimensions: {dimensions}")
    print(f"Base epochs: {args.base_epochs} (scaled by dimension)")
    print(f"Sigma: {args.sigma}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()

    # Train models
    for dim in dimensions:
        epochs = args.base_epochs * dim

        try:
            train_sb_models(
                dim=dim,
                num_epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                sigma=args.sigma,
                save_models=not args.no_save,
                device=device,
            )
        except Exception as e:
            print(f"Error training dimension {dim}: {e}")
            continue

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
