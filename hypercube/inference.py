#!/usr/bin/env python3
"""
Inference script for FK Steering Hypercube Benchmark

This script runs inference with trained FK steering models and outputs CSV results.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
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


def run_single_inference(
    fk_steerer, dim, args, device, run_id, progress_bar=None, generator=None
):
    """
    Run a single inference run with FK steering.
    
    Args:
        fk_steerer: FKSteeringSDE instance
        dim: Problem dimension
        args: Parsed arguments containing run parameters
        device: Device to run on
        run_id: Run identifier (for logging)
        progress_bar: Optional progress bar to update
        generator: Random number generator for reproducibility
        
    Returns:
        tuple: (success_rate, w2_distance) performance metrics
    """

    def progress_callback():
        if progress_bar is not None:
            progress_bar.update(1)

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

    success_rate = compute_success_rate(fk_samples, threshold=args.threshold)
    w2_dist = wasserstein_2_to_target(
        fk_samples,
        threshold=args.threshold,
        squared=True,
        n_projections=args.w2_projections,
    )

    return success_rate, w2_dist


def run_inference_for_dimension(dim, args, device):
    """
    Run inference for a single dimension with parallel runs.
    
    Loads pre-trained models, sets up FK steering, and runs multiple
    independent inference runs in parallel for statistical analysis.
    
    Args:
        dim: Problem dimension
        args: Parsed arguments containing experiment parameters
        device: Device to run inference on
        
    Returns:
        tuple: (result_dict, error_message) where result_dict contains
               mean/std statistics and individual run results, or None if error
    """
    try:
        flow_model, score_model = load_sb_models(dim, device=device)
    except FileNotFoundError as e:
        return None, f"No trained model for dimension {dim}"

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

    fk_steerer = FKSteeringSDE(
        flow_model=flow_model,
        score_model=score_model,
        potential_fn=potential_fn,
        sigma=args.sigma,
        num_samples=args.particles,
        fk_steering_temperature=args.temperature,
        resample_freq=args.resample_freq,
        fk_potential_scheduler=args.scheduler,
        generator=generator,
    )

    # Run inference with progress bar tracking total steps
    success_rates = []
    w2_dists = []

    # Calculate total steps: num_runs * (num_steps - 1) per dimension
    total_steps = args.num_runs * (args.num_steps - 1)

    with ThreadPoolExecutor(max_workers=min(args.num_runs, 4)) as executor:
        # Create shared progress bar for all runs in this dimension
        with tqdm(
            total=total_steps,
            desc=f"Dim {dim:2d}",
            leave=True,
            ncols=80,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}",
            unit="steps",
        ) as pbar:
            futures = [
                executor.submit(
                    run_single_inference,
                    fk_steerer,
                    dim,
                    args,
                    device,
                    run,
                    pbar,
                    generator,
                )
                for run in range(args.num_runs)
            ]

            for future in futures:
                success_rate, w2_dist = future.result()
                success_rates.append(success_rate)
                w2_dists.append(w2_dist)

    success_rates = np.array(success_rates)
    w2_dists = np.array(w2_dists)

    # Print result on the same line as the completed progress bar
    tqdm.write(
        f"Dim {dim:2d}: {success_rates.mean():5.1f}% ± {success_rates.std():4.1f}% | W2: {w2_dists.mean():.4f} ± {w2_dists.std():.4f}"
    )

    result = {
        "dimension": dim,
        "mean_success_rate": success_rates.mean(),
        "std_success_rate": success_rates.std(),
        "mean_w2_dist": w2_dists.mean(),
        "std_w2_dist": w2_dists.std(),
        "success_rates": success_rates.tolist(),
        "w2_dists": w2_dists.tolist(),
    }

    return result, None


def main():
    parser = argparse.ArgumentParser(
        description="Run FK steering inference for hypercube benchmark"
    )

    # Dimension settings
    parser.add_argument(
        "--dimensions",
        type=str,
        default="1-10",
        help="Dimension range to test (e.g., '1-10' or '5,7,9')",
    )

    # Inference parameters
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of inference runs per dimension",
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
        "--particles",
        type=int,
        default=1024,
        help="Number of particles for FK steering",
    )
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

    # Parse dimensions
    if "-" in args.dimensions:
        start, end = map(int, args.dimensions.split("-"))
        dimensions = list(range(start, end + 1))
    else:
        dimensions = [int(d.strip()) for d in args.dimensions.split(",")]

    # Setup device
    device = get_device(args.device)

    print(f"FK Steering Hypercube Benchmark - Inference")
    print(f"Using device: {device}")
    print(f"Dimensions: {dimensions}")
    print(
        f"Parameters: temp={args.temperature}, sigma={args.sigma}, particles={args.particles}"
    )
    print(
        f"Steps: {args.num_steps}, resample_freq={args.resample_freq}, scheduler={args.scheduler}"
    )
    print(f"Runs: {args.num_runs} x {args.samples_per_run} samples")
    print()

    # Run inference
    results = []
    all_runs_data = []

    for dim in dimensions:
        result, error = run_inference_for_dimension(dim, args, device)
        if result is not None:
            results.append(result)

            # Flatten for individual run records
            for i, (sr, w2) in enumerate(
                zip(result["success_rates"], result["w2_dists"])
            ):
                all_runs_data.append(
                    {
                        "dimension": dim,
                        "run": i + 1,
                        "success_rate": sr,
                        "w2_dist": w2,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        elif error:
            print(f"✗ {error}")
            continue

    if not results:
        print("No results obtained. Check that trained models exist.")
        return

    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append(
            {
                "dimension": r["dimension"],
                "mean_success_rate": r["mean_success_rate"],
                "std_success_rate": r["std_success_rate"],
                "mean_w2_dist": r["mean_w2_dist"],
                "std_w2_dist": r["std_w2_dist"],
                "num_runs": args.num_runs,
                "samples_per_run": args.samples_per_run,
                "temperature": args.temperature,
                "sigma": args.sigma,
                "particles": args.particles,
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

    summary_df = pd.DataFrame(summary_data)
    runs_df = pd.DataFrame(all_runs_data)

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"fk_hypercube_results_{timestamp}.csv"

    # Save results
    summary_df.to_csv(args.output, index=False)

    # Also save individual runs
    runs_output = args.output.replace(".csv", "_runs.csv")
    runs_df.to_csv(runs_output, index=False)

    print(f"Results saved to:")
    print(f"  Summary: {args.output}")
    print(f"  Individual runs: {runs_output}")

    # Print summary
    print("\nSummary:")
    for _, row in summary_df.iterrows():
        dim = int(row["dimension"])
        success = row["mean_success_rate"]
        success_std = row["std_success_rate"]
        w2 = row["mean_w2_dist"]
        w2_std = row["std_w2_dist"]
        print(
            f"  Dim {dim:2d}: {success:5.1f}% ± {success_std:4.1f}% | W2: {w2:.4f} ± {w2_std:.4f}"
        )


if __name__ == "__main__":
    main()
