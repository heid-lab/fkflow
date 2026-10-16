"""Collect the paper results: Sella + IRC summaries, per-run metrics, tables over seeds.

    PYTHONPATH=. .venv/bin/python scripts/paper/aggregate.py summarize   # every check; CPU job
    PYTHONPATH=. .venv/bin/python scripts/paper/aggregate.py collect     # CSVs and tables

summarize runs sella_check.py summarize and irc_check.py summarize on every
check of the campaign (references first), which writes sella_results.csv and
irc_results.csv next to the chunks and the printed summaries to summary.txt.
collect reads those and the test_results.csv of every run and writes to
$PAPER_OUT/results/:

    runs.csv         one row per sampling run: geometry and chirality metrics
    checks.csv       one row per (surface, run): Sella and IRC rates, % of all test reactions
    by_stereo.csv    correct saddle and IRC success by number of stereocentres
    forces.csv       one row per run: forces of the generated TSs on g-xTB and AIMNet2-rxn
    runs_mean.csv    runs.csv and forces.csv, mean and std over seeds per configuration
    checks_mean.csv  checks.csv, mean and std over seeds per (surface, configuration)
    tables.md        one table per block of make_manifest.py, mean ± std
"""

import contextlib
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import irc_check
import sella_check

HERE = Path(__file__).parent
STEREO_CSV = HERE / "stereo_counts.csv"
OUT = Path(os.environ.get("PAPER_OUT", "paper"))
RESULTS = OUT / "results"

RUN_METRICS = {
    "rmse": "RMSE (Å)",
    "rmse_with_flip": "RMSE up to mirror (Å)",
    "rmse_correct_chir": "RMSE, right chirality (Å)",
    "tce": "TCE (%)",
    "tce_chiral": "TCE, chiral (%)",
    "steric_clash": "clash",
    "ess_mean": "ESS",
}
CHECK_METRICS = {
    "first_order": "1st-order (%)",
    "correct_saddle": "correct saddle (%)",
    "irc_connects": "IRC connects (%)",
    "connects_any_stereo": "IRC connects, any stereo (%)",
}
FORCE_METRICS = {
    "gxtb_rms_median": "g-xTB RMS force (eV/Å)",
    "gxtb_below_ref": "g-xTB, below reference TS (%)",
    "aimnet_rms_median": "AIMNet RMS force (eV/Å)",
    "aimnet_below_ref": "AIMNet, below reference TS (%)",
    "aimnet_spread_median": "AIMNet spread (eV/Å)",
}
MAIN_ARM = {"A": "chir_fk_N30", "B": "gx_fk_N30", "C": "aim_fk_N30"}
CATEGORIES = [
    "correct_saddle",
    "connects",
    "connects_wrong_stereo",
    "same_side",
    "one_side",
    "unrelated",
    "irc_failed",
    "no_first_order_saddle",
]


def manifest():
    runs = pd.read_csv(HERE / "runs.tsv", sep="\t")
    runs["config"] = runs.run.str.replace(r"_s\d+$", "", regex=True)
    return runs


def summarize_check(check_dir):
    """Sella then IRC summary of one check; output to summary.txt."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            sella_check.summarize(check_dir / "sella")
            print()
            irc_check.summarize(check_dir / "irc", STEREO_CSV)
        error = ""
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        buf.write(f"\n!!! {error}\n")
    (check_dir / "summary.txt").write_text(buf.getvalue())
    return check_dir, error


def summarize(processes=32):
    import warnings

    warnings.filterwarnings("ignore")
    checks = sorted(p.parent for p in OUT.glob("saddle/*/*/irc"))
    references = [c for c in checks if c.name == "reference"]
    others = [c for c in checks if c.name != "reference"]
    # the run checks read the references' sella_results.csv; summarize those first
    with Pool(processes) as pool:
        results = pool.map(summarize_check, references)
        results += pool.map(summarize_check, others, chunksize=1)
    failed = [(c, e) for c, e in results if e]
    for c, e in failed:
        print(f"!!! {c}: {e}")
    print(f"{len(results)} checks summarized, {len(failed)} failed")
    return not failed


def run_row(run_dir):
    t = pd.read_csv(run_dir / "test_results.csv")
    chiral = t.has_chiral == 1
    right = chiral & (t.chirality == 0)
    return dict(
        n_reactions=len(t),
        rmse=t.rmse.mean(),
        rmse_median=t.rmse.median(),
        rmse_with_flip=t.rmse_with_flip.mean(),
        mae=t.mae.mean(),
        rmse_correct_chir=t.rmse[right].mean(),
        tce=100 * t.chirality.mean(),
        tce_chiral=100 * t.chirality[chiral].mean(),
        steric_clash=t.steric_clash.mean(),
        ess_mean=t.ess_mean.mean() if "ess_mean" in t else float("nan"),  # FK runs only
        time=pd.read_csv(run_dir / "stats.csv").time.iloc[0],
    )


def force_row(run):
    """Forces of a run's generated TSs (force_worker.sh), per surface, against the reference TSs."""
    row = {}
    for surface in ("gxtb", "aimnet"):
        f = OUT / "force" / surface / f"{run}.tsv"
        if not f.exists():
            continue
        ref = pd.read_csv(OUT / "force" / surface / "reference.tsv", sep="\t")
        df = pd.read_csv(f, sep="\t").merge(ref, on="rxn_index", suffixes=("", "_ref"))
        row |= {
            f"{surface}_rms_median": df.rms_force.median(),
            f"{surface}_rms_mean": df.rms_force.mean(),
            f"{surface}_max_median": df.max_force.median(),
            f"{surface}_below_ref": 100 * (df.rms_force < df.rms_force_ref).mean(),
            f"{surface}_ratio_to_ref_median": (df.rms_force / df.rms_force_ref).median(),
            f"{surface}_failed": int(df.rms_force.isna().sum()),
        }
        if surface == "aimnet":
            row["aimnet_spread_median"] = df.force_spread.median()
    return row


def check_rows(surface_dir):
    """checks.csv and by_stereo.csv rows of one check (its label only)."""
    df = pd.read_csv(surface_dir / "irc" / "irc_results.csv", low_memory=False)
    label = surface_dir.name
    df = df[df.label == label]
    first_order = df.converged & (df.n_imag == 1)
    cats = df.category.value_counts(normalize=True).reindex(CATEGORIES, fill_value=0) * 100
    row = dict(
        surface=surface_dir.parent.name,
        run=label,
        n_reactions=len(df),
        first_order=100 * first_order.mean(),
        correct_saddle=100 * df.correct.mean(),
        irc_connects=100 * df.irc_success.mean(),
        connects_any_stereo=100 * df.outcome.isin(["connects", "connects_wrong_stereo"]).mean(),
        sella_steps_median=df.loc[df.converged, "steps"].median(),
        start_1_imag=100 * (df.n_imag_start == 1).mean(),
        sella_errors=int((df.error.fillna("") != "").sum()),
        **{f"cat_{c}": v for c, v in cats.items()},
    )
    bins = df.n_stereo.clip(upper=3)
    stereo = (
        df.assign(n_stereo_bin=bins)
        .groupby("n_stereo_bin")
        .agg(
            n=("rxn_index", "size"),
            correct_saddle=("correct", "mean"),
            irc_connects=("irc_success", "mean"),
        )
        .reset_index()
    )
    stereo[["correct_saddle", "irc_connects"]] *= 100
    stereo.insert(0, "run", label)
    stereo.insert(0, "surface", row["surface"])
    return row, stereo


def mean_std(df, keys, metrics):
    g = df.groupby(keys, sort=False)[list(metrics)]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out["n_seeds"] = df.groupby(keys, sort=False).size()
    return out.reset_index()


def fmt(mean, std, digits):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown(df, keys, metrics):
    cols = {k: k for k in keys} | metrics
    lines = ["| " + " | ".join(cols.values()) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        cells = [str(r[k]) for k in keys]
        for m in metrics:
            digits = 3 if m.startswith("rmse") or m.endswith(("_median", "spread")) else 1
            cells.append(fmt(r[f"{m}_mean"], r[f"{m}_std"], digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect():
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = manifest()
    rows = [run_row(OUT / "runs" / r) for r in runs.run]
    runs = runs.drop(columns=["idx", "overrides", "surfaces"]).join(pd.DataFrame(rows))
    runs.to_csv(RESULTS / "runs.csv", index=False)
    forces = pd.DataFrame([force_row(r) for r in runs.run], index=runs.index)
    has_forces = not forces.empty and set(FORCE_METRICS) <= set(forces.columns)
    if has_forces:
        runs[["run"]].join(forces).to_csv(RESULTS / "forces.csv", index=False)
        runs = runs.join(forces[list(FORCE_METRICS)])

    rows, stereo = [], []
    for d in sorted(OUT.glob("saddle/*/*/irc/irc_results.csv")):
        row, st = check_rows(d.parent.parent)
        rows.append(row)
        stereo.append(st)
    meta = runs[["run", "study", "block", "config", "seed", "n"]]
    checks = pd.DataFrame(rows).merge(meta, on="run", how="left")
    checks.loc[checks.run == "reference", ["config", "block"]] = ["reference", "main"]
    checks.to_csv(RESULTS / "checks.csv", index=False)
    pd.concat(stereo).merge(meta, on="run", how="left").to_csv(
        RESULTS / "by_stereo.csv", index=False
    )

    run_metrics = RUN_METRICS | (FORCE_METRICS if has_forces else {})
    run_mean = mean_std(runs, ["study", "block", "config", "n"], run_metrics)
    run_mean.to_csv(RESULTS / "runs_mean.csv", index=False)
    check_metrics = CHECK_METRICS | {f"cat_{c}": c for c in CATEGORIES}
    check_mean = mean_std(checks, ["surface", "block", "config"], check_metrics)
    check_mean.to_csv(RESULTS / "checks_mean.csv", index=False)

    md = ["# Paper results\n", "Mean ± std over seeds; check rates are % of all test reactions.\n"]
    for block in sorted(set(run_mean.block) - {"main"}):
        # every block next to GoFlow and the main FK arm of its study
        context = ["reference", "goflow_N30", MAIN_ARM[block[0]]]
        sel = run_mean[(run_mean.block == block) | run_mean.config.isin(context)]
        md += [f"## {block}: sampling\n", markdown(sel, ["config", "n"], RUN_METRICS), ""]
        if has_forces:
            md += [f"## {block}: forces of the generated TSs\n"]
            md += [markdown(sel, ["config", "n"], FORCE_METRICS), ""]
        for surface in ("gxtb", "aimnet"):
            on = check_mean[check_mean.surface == surface]
            sel = on[(on.block == block) | on.config.isin(context)]
            if not sel.block.eq(block).any():
                continue
            md += [f"## {block}: Sella + IRC on {surface}\n"]
            md += [markdown(sel, ["config"], CHECK_METRICS), ""]
    (RESULTS / "tables.md").write_text("\n".join(md))
    print(f"wrote {RESULTS}: {len(runs)} runs, {len(checks)} checks")


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        sys.exit(0 if summarize() else 1)
    collect()
