#!/usr/bin/env python3
"""
Inference script for constant dimension and different particle counts
"""

import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from core import (
    load_sb_models,
    FKSteeringSDE,
    indicator_potential,
    distance_potential,
    sample_uniform_hypercube,
    compute_success_rate,
    wasserstein_2_to_target,
    get_device,
    set_random_seed,
)


def benchmark_particles_for_dimension(dim, particle_counts, args, device):
    """
    Benchmark different particle counts for FK steering on a single dimension.
    
    Tests how FK steering performance varies with the number of particles
    in the particle filter, keeping all other parameters fixed.
    
    Args:
        dim: Problem dimension to benchmark
        particle_counts: List of particle counts to test
        args: Parsed arguments containing experiment parameters
        device: Device to run benchmarks on
        
    Returns:
        list: List of result dictionaries, one per particle count tested
    """
    try:
        flow_model, score_model = load_sb_models(dim, device=device)
    except FileNotFoundError:
        return None

    # Create potential function based on type
    if args.potential_type == "indicator":
        potential_fn = lambda x: indicator_potential(
            x, dim, threshold=args.threshold, log_weight=args.log_weight
        )
    elif args.potential_type == "distance":
        potential_fn = lambda x: distance_potential(
            x,
            dim,
            threshold=args.threshold,
            weight_scale=args.weight_scale,
            exponent=args.exponent,
        )
    else:
        raise ValueError(f"Unknown potential type: {args.potential_type}")

    # Create generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    results = []
    print(f"Benchmarking dimension {dim}:")

    for num_particles in particle_counts:
        print(f"  {num_particles:4d} particles:", end=" ", flush=True)

        fk_steerer = FKSteeringSDE(
            flow_model=flow_model,
            score_model=score_model,
            potential_fn=potential_fn,
            sigma=args.sigma,
            num_samples=num_particles,
            fk_steering_temperature=args.temperature,
            resample_freq=args.resample_freq,
            fk_potential_scheduler=args.scheduler,
            generator=generator,
        )

        success_rates = []
        w2_dists = []

        # Calculate total steps for this particle count
        total_steps = args.num_runs * (args.num_steps - 1)

        def progress_callback():
            pbar.update(1)

        with tqdm(
            total=total_steps,
            desc=f"{num_particles:4d}",
            leave=False,
            ncols=70,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            unit="steps",
        ) as pbar:
            for run in range(args.num_runs):
                with torch.no_grad():
                    fk_samples = fk_steerer.generate_samples(
                        x0_sampler=lambda n, device: sample_uniform_hypercube(
                            n, dim, device=device, generator=generator
                        ),
                        num_steps=args.num_steps,
                        batch_size=args.samples_per_run,
                        device=device,
                        progress_callback=progress_callback,
                    )

                success_rate = compute_success_rate(
                    fk_samples, threshold=args.threshold
                )
                w2_dist = wasserstein_2_to_target(
                    fk_samples,
                    threshold=args.threshold,
                    squared=True,
                    n_projections=args.w2_projections,
                )

                success_rates.append(success_rate)
                w2_dists.append(w2_dist)

        success_rates = np.array(success_rates)
        w2_dists = np.array(w2_dists)

        result = {
            "dimension": dim,
            "num_particles": num_particles,
            "mean_success_rate": success_rates.mean(),
            "std_success_rate": success_rates.std(),
            "mean_w2_dist": w2_dists.mean(),
            "std_w2_dist": w2_dists.std(),
            "success_rates": success_rates.tolist(),
            "w2_dists": w2_dists.tolist(),
        }

        results.append(result)
        print(f"{success_rates.mean():.1f}% ± {success_rates.std():.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FK steering performance vs particle count"
    )

    # Dimension settings
    parser.add_argument(
        "--dimension", type=int, required=True, help="Dimension to benchmark"
    )
    parser.add_argument(
        "--particles",
        type=str,
        default="2,4,8,16,32,64,128,256,512,1024",
        help="Particle counts to test (comma-separated)",
    )

    # Inference parameters
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of inference runs per particle count",
    )
    parser.add_argument(
        "--samples_per_run", type=int, default=1024, help="Number of samples per run"
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of SDE integration steps"
    )

    # FK steering parameters
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="FK steering temperature"
    )
    parser.add_argument("--sigma", type=float, default=2.0, help="SDE noise level")
    parser.add_argument(
        "--resample_freq", type=int, default=10, help="Resampling frequency"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="harmonic_sum",
        choices=["sum", "harmonic_sum", "difference"],
        help="FK potential scheduler",
    )

    # Task parameters
    parser.add_argument(
        "--threshold", type=float, default=0.0, help="Target region threshold"
    )

    # Potential function parameters
    parser.add_argument(
        "--potential_type",
        type=str,
        default="indicator",
        choices=["indicator", "distance"],
        help="Type of potential function",
    )
    parser.add_argument(
        "--log_weight",
        type=float,
        default=None,
        help="Log weight for indicator potential (default: log(2^dim-1))",
    )
    parser.add_argument(
        "--weight_scale",
        type=float,
        default=1.0,
        help="Weight scale for distance potential",
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=1.0,
        help="Exponent for distance potential (1.0=linear, 2.0=quadratic)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--w2_projections",
        type=int,
        default=4096,
        help="Number of projections for Wasserstein distance",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: auto-generated)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # System settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cuda', 'mps', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)

    # Parse particle counts
    particle_counts = [int(p.strip()) for p in args.particles.split(",")]

    # Setup device
    device = get_device(args.device)

    print("FK Steering Particle Benchmark")
    print(f"Using device: {device}")
    print(f"Dimension: {args.dimension}")
    print(f"Particle counts: {particle_counts}")
    print(f"Parameters: temp={args.temperature}, sigma={args.sigma}")
    print(
        f"Steps: {args.num_steps}, resample_freq={args.resample_freq}, scheduler={args.scheduler}"
    )
    print(f"Runs: {args.num_runs} x {args.samples_per_run} samples")
    print()

    # Run benchmark
    results = benchmark_particles_for_dimension(
        args.dimension, particle_counts, args, device
    )

    if not results:
        print("No results obtained. Check that trained models exist.")
        return

    # Create DataFrames
    summary_data = []
    all_runs_data = []

    for r in results:
        summary_data.append(
            {
                "dimension": r["dimension"],
                "num_particles": r["num_particles"],
                "mean_success_rate": r["mean_success_rate"],
                "std_success_rate": r["std_success_rate"],
                "mean_w2_dist": r["mean_w2_dist"],
                "std_w2_dist": r["std_w2_dist"],
                "num_runs": args.num_runs,
                "samples_per_run": args.samples_per_run,
                "temperature": args.temperature,
                "sigma": args.sigma,
                "num_steps": args.num_steps,
                "resample_freq": args.resample_freq,
                "scheduler": args.scheduler,
                "threshold": args.threshold,
                "potential_type": args.potential_type,
                "log_weight": args.log_weight,
                "weight_scale": args.weight_scale,
                "exponent": args.exponent,
                "seed": args.seed,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Flatten for individual run records
        for i, (sr, w2) in enumerate(zip(r["success_rates"], r["w2_dists"])):
            all_runs_data.append(
                {
                    "dimension": r["dimension"],
                    "num_particles": r["num_particles"],
                    "run": i + 1,
                    "success_rate": sr,
                    "w2_dist": w2,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    summary_df = pd.DataFrame(summary_data)
    runs_df = pd.DataFrame(all_runs_data)

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"fk_particle_benchmark_dim{args.dimension}_{timestamp}.csv"

    # Save results
    summary_df.to_csv(args.output, index=False)

    # Also save individual runs
    runs_output = args.output.replace(".csv", "_runs.csv")
    runs_df.to_csv(runs_output, index=False)

    print("Results saved to:")
    print(f"  Summary: {args.output}")
    print(f"  Individual runs: {runs_output}")

    # Print summary
    print(f"\nParticle Benchmark Summary (Dimension {args.dimension}):")
    for _, row in summary_df.iterrows():
        particles = int(row["num_particles"])
        success = row["mean_success_rate"]
        success_std = row["std_success_rate"]
        w2 = row["mean_w2_dist"]
        w2_std = row["std_w2_dist"]
        print(
            f"  {particles:4d} particles: {success:5.1f}% ± {success_std:4.1f}% | W2: {w2:.4f} ± {w2_std:.4f}"
        )


if __name__ == "__main__":
    main()
