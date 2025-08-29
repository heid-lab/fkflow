#!/usr/bin/env python3
"""
Utility functions for FK Steering Hypercube Benchmark
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_summarize_results(csv_path):
    """Load CSV results and print summary statistics"""
    df = pd.read_csv(csv_path)

    print(f"Results from: {csv_path}")
    print(
        f"Timestamp: {df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'Unknown'}"
    )
    print()

    if "dimension" in df.columns:
        print("Results by dimension:")
        for _, row in df.iterrows():
            dim = int(row["dimension"])
            if "mean_success_rate" in row:
                success = row["mean_success_rate"]
                success_std = row["std_success_rate"]
                print(f"  Dim {dim:2d}: {success:5.1f}% ± {success_std:4.1f}%")
                if "mean_w2_dist" in row:
                    w2 = row["mean_w2_dist"]
                    w2_std = row["std_w2_dist"]
                    print(f"          W2: {w2:.4f} ± {w2_std:.4f}")

    if "num_particles" in df.columns:
        print("Results by particle count:")
        for _, row in df.iterrows():
            particles = int(row["num_particles"])
            success = row["mean_success_rate"]
            success_std = row["std_success_rate"]
            print(f"  {particles:4d} particles: {success:5.1f}% ± {success_std:4.1f}%")


def compare_results(csv_paths):
    """Compare results from multiple CSV files"""
    results = {}

    for path in csv_paths:
        path = Path(path)
        df = pd.read_csv(path)
        results[path.stem] = df

    print("Comparison across experiments:")
    print()

    # If all have dimensions, compare by dimension
    if all("dimension" in df.columns for df in results.values()):
        dimensions = set()
        for df in results.values():
            dimensions.update(df["dimension"].values)

        for dim in sorted(dimensions):
            print(f"Dimension {dim}:")
            for name, df in results.items():
                dim_data = df[df["dimension"] == dim]
                if len(dim_data) > 0:
                    row = dim_data.iloc[0]
                    success = row["mean_success_rate"]
                    success_std = row["std_success_rate"]
                    print(f"  {name:20s}: {success:5.1f}% ± {success_std:4.1f}%")
            print()


def generate_summary_report(csv_path, output_path=None):
    """Generate a text summary report from CSV results"""
    df = pd.read_csv(csv_path)

    if output_path is None:
        output_path = csv_path.replace(".csv", "_report.txt")

    with open(output_path, "w") as f:
        f.write("FK Steering Hypercube Benchmark Results\n")
        f.write("=" * 40 + "\n\n")

        if "timestamp" in df.columns:
            f.write(f"Generated: {df['timestamp'].iloc[0]}\n")
        f.write(f"Source: {csv_path}\n\n")

        # Experiment parameters
        if "temperature" in df.columns:
            params = df.iloc[0]
            f.write("Parameters:\n")
            f.write(f"  Temperature: {params['temperature']}\n")
            f.write(f"  Sigma: {params['sigma']}\n")
            f.write(
                f"  Particles: {params.get('particles', params.get('num_particles', 'N/A'))}\n"
            )
            f.write(f"  Steps: {params['num_steps']}\n")
            f.write(f"  Resample freq: {params['resample_freq']}\n")
            f.write(f"  Scheduler: {params['scheduler']}\n")
            f.write(f"  Runs: {params['num_runs']}\n")
            f.write(f"  Samples per run: {params['samples_per_run']}\n")
            f.write("\n")

        # Results
        f.write("Results:\n")
        if "dimension" in df.columns:
            for _, row in df.iterrows():
                dim = int(row["dimension"])
                success = row["mean_success_rate"]
                success_std = row["std_success_rate"]
                f.write(f"  Dimension {dim:2d}: {success:5.1f}% ± {success_std:4.1f}%")
                if "mean_w2_dist" in row:
                    w2 = row["mean_w2_dist"]
                    w2_std = row["std_w2_dist"]
                    f.write(f" | W2: {w2:.4f} ± {w2_std:.4f}")
                f.write("\n")

        if "num_particles" in df.columns:
            for _, row in df.iterrows():
                particles = int(row["num_particles"])
                success = row["mean_success_rate"]
                success_std = row["std_success_rate"]
                f.write(
                    f"  {particles:4d} particles: {success:5.1f}% ± {success_std:4.1f}%"
                )
                if "mean_w2_dist" in row:
                    w2 = row["mean_w2_dist"]
                    w2_std = row["std_w2_dist"]
                    f.write(f" | W2: {w2:.4f} ± {w2_std:.4f}")
                f.write("\n")

    print(f"Summary report saved to: {output_path}")


def check_trained_models():
    """Check which model dimensions are available"""
    model_dir = Path("trained_models")

    if not model_dir.exists():
        print("No trained_models directory found")
        return

    model_files = list(model_dir.glob("sb_models_dim*.pt"))

    if not model_files:
        print("No trained models found in trained_models/")
        return

    dimensions = []
    for model_file in model_files:
        # Extract dimension from filename like sb_models_dim5.pt
        try:
            dim_str = model_file.stem.split("dim")[1]
            dim = int(dim_str)
            dimensions.append(dim)
        except (IndexError, ValueError):
            continue

    dimensions.sort()
    print(f"Available trained models for dimensions: {dimensions}")

    return dimensions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utility functions for FK benchmark")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Summarize results from CSV")
    summary_parser.add_argument("csv_file", help="CSV file to summarize")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple CSV files")
    compare_parser.add_argument("csv_files", nargs="+", help="CSV files to compare")

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate text report from CSV"
    )
    report_parser.add_argument("csv_file", help="CSV file to process")
    report_parser.add_argument("--output", help="Output text file")

    # Check models command
    check_parser = subparsers.add_parser(
        "check-models", help="Check available trained models"
    )

    args = parser.parse_args()

    if args.command == "summary":
        load_and_summarize_results(args.csv_file)
    elif args.command == "compare":
        compare_results(args.csv_files)
    elif args.command == "report":
        generate_summary_report(args.csv_file, args.output)
    elif args.command == "check-models":
        check_trained_models()
    else:
        parser.print_help()
