"""Write the run manifests of the paper: every sampling run and every Sella + IRC check.

    python scripts/paper/make_manifest.py    # writes runs.tsv and checks.tsv next to this file

runs.tsv has one row per sampling run (gpu_worker.sh): a name, a seed and the
Hydra overrides. checks.tsv has one row per Sella + IRC chunk (saddle_worker.sh):
the run it starts from, the potential surface, and the chunk of the test reactions.

All runs share one recipe and differ only in what the tables vary: the time grid
has 51 points, 30 steps on [0, 0.8] and 20 in the steering window [0.8, 1]; noise
sigma = 0.3 with the sqrt_window schedule, switched off at t = 0.95; FK resamples
after every step in the window up to t = 0.95 and after the last step, with the
difference schedule. The studies:

    A  chirality potential alone (main table, method x particles, and the FK settings)
    B  g-xTB force + chirality + E/Z (the DFT arm), g-xTB Sella + IRC
    C  AIMNet2-rxn force (ensemble mean + spread) + chirality + E/Z, AIMNet Sella + IRC

GoFlow without steering does not depend on the potential, so its runs are shared.
The main arms of B and C are also checked on the other surface (cross-evaluation).
Rows are ordered main tables first, then by seed, so whatever has finished is
whole seeds of the most important tables.
"""

import csv
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = (1, 2, 3, 4, 5)
N_CHUNKS = 16  # chunks per Sella + IRC check of the 1194 test reactions

# GPU memory sets the batch; the batch also fixes which prior a molecule gets,
# so it depends only on N and is the same for every method at that N
BATCH = {1: 1200, 2: 1200, 4: 1200, 8: 1200, 16: 1100, 30: 600, 64: 300, 128: 150}

GRID = {
    "model.num_steps": 51,
    "+model.window_num_steps": 20,
    "+model.steering_t_start": 0.8,
    "+model.noise_t_start": 0.8,
    "+model.noise_schedule": "sqrt_window",
    "+model.steering_base_variance": 0.3,
    "+model.noise_t_end": 0.95,
}
GOFLOW = {"+model.inference_sampling_method": "median"}
FK = {
    "+model.inference_sampling_method": "fk_steering",
    "+model.fk_potential_scheduler": "difference",
    "+model.resample_freq": 1,
    "+model.steering_t_end": 0.95,
    "+model.fk_steering_temperature": 100,
}
# importance sampling: SDE particles, one resampling event after the last step
IS = {
    "+model.inference_sampling_method": "fk_steering",
    "+model.fk_potential_scheduler": "sum",
    "+model.resample_freq": 1000,
    "+model.fk_steering_temperature": 100,
}
# best of N: independent ODE samples (no noise), selected by the potential at the end
BEST = {**IS, "+model.steering_base_variance": 0.0}

CHIR = {"+fk_potentials": "chirality"}
GX = {"+fk_potentials": "gxtb_force_chirality_ez"}
AIM = {
    "+fk_potentials": "aimnet_force_chirality_ez",
    "model.fk_potential_fns.1.uncertainty_weight": 1.0,
}

# the configuration of the original manuscript (uniform 31-point grid, steering from t = 0)
PUBLISHED = {
    "model.num_steps": 31,
    "+model.steering_t_start": 0.0,
    "+model.noise_schedule": "linear",
    "+model.steering_base_variance": 0.3,
}


def bare(key):
    return key.lstrip("+")


def merge(*parts):
    """Later parts win; a key given with and without '+' counts as one key."""
    out = {}
    for part in parts:
        for k, v in part.items():
            out = {kk: vv for kk, vv in out.items() if bare(kk) != bare(k)}
            out[k] = v
    return out


def drop(overrides, *keys):
    return {k: v for k, v in overrides.items() if bare(k) not in keys}


RUNS = []  # (tier, study, block, name, n, overrides, surfaces)


def run(tier, study, block, name, n, overrides, surfaces=()):
    RUNS.append((tier, study, block, name, n, overrides, tuple(surfaces)))


# --- main tables (tier 0): N = 30 ---------------------------------------------
BOTH = ("gxtb", "aimnet")
run(0, "shared", "main", "goflow_N30", 30, merge(GRID, GOFLOW), BOTH)
run(0, "A", "A1", "chir_fk_N30", 30, merge(GRID, CHIR, FK), BOTH)
run(0, "A", "A1", "chir_is_N30", 30, merge(GRID, CHIR, IS))
run(0, "A", "A1", "chir_best_N30", 30, merge(GRID, CHIR, BEST))
for tag, pot in (("gx", GX), ("aim", AIM)):
    study = "B" if tag == "gx" else "C"
    block = f"{study}1"
    run(0, study, block, f"{tag}_fk_N30", 30, merge(GRID, pot, FK), BOTH)
    run(0, study, block, f"{tag}_is_N30", 30, merge(GRID, pot, IS), BOTH)
    run(0, study, block, f"{tag}_best_N30", 30, merge(GRID, pot, BEST), BOTH)

# --- particle number (tier 1) -------------------------------------------------
run(1, "shared", "A1", "goflow_N1", 1, merge(GRID, GOFLOW))
for n in (2, 4, 8, 16, 64):
    run(1, "shared", "A1", f"goflow_N{n}", n, merge(GRID, GOFLOW))
    run(1, "A", "A1", f"chir_fk_N{n}", n, merge(GRID, CHIR, FK))
    run(1, "A", "A1", f"chir_is_N{n}", n, merge(GRID, CHIR, IS))
    run(1, "A", "A1", f"chir_best_N{n}", n, merge(GRID, CHIR, BEST))
for tag, pot, home in (("gx", GX, "gxtb"), ("aim", AIM, "aimnet")):
    study = "B" if tag == "gx" else "C"
    for n in (8, 16, 64, 128):
        run(1, study, f"{study}5", f"{tag}_fk_N{n}", n, merge(GRID, pot, FK), (home,))
        run(1, study, f"{study}5", f"{tag}_is_N{n}", n, merge(GRID, pot, IS), (home,))


# --- FK settings and potential composition (tier 2), N = 30 --------------------
def settings_a(block, name, overrides):
    run(2, "A", block, name, 30, overrides)


chir_fk = merge(GRID, CHIR, FK)
for sched in ("difference", "sum", "harmonic_sum", "max"):
    for lam in (1, 10, 100, 1000):
        if (sched, lam) != ("difference", 100):  # that one is chir_fk_N30
            schedule = {
                "+model.fk_potential_scheduler": sched,
                "+model.fk_steering_temperature": lam,
            }
            settings_a("A2", f"chir_fk_{sched}_lam{lam}", merge(chir_fk, schedule))
for freq in (2, 5):
    settings_a("A3", f"chir_fk_freq{freq}", merge(chir_fk, {"+model.resample_freq": freq}))
for sigma in (0.0, 0.1, 1.0):
    settings_a(
        "A3", f"chir_fk_sigma{sigma}", merge(chir_fk, {"+model.steering_base_variance": sigma})
    )
window_05 = {"+model.steering_t_start": 0.5, "+model.noise_t_start": 0.5}
settings_a("A3", "chir_fk_t05", merge(chir_fk, window_05))
# steering from t = 0: no window, so the grid is uniform over the same 50 steps
window_00 = {"+model.steering_t_start": 0.0, "+model.noise_t_start": 0.0}
settings_a("A3", "chir_fk_t00", merge(drop(chir_fk, "model.window_num_steps"), window_00))
published_fk = {
    "+model.inference_sampling_method": "fk_steering",
    "+model.fk_potential_scheduler": "sum",
    "+model.resample_freq": 6,
    "+model.fk_steering_temperature": 0.4,
}
run(2, "shared", "A4", "goflow_published", 30, merge(PUBLISHED, GOFLOW))
settings_a("A4", "chir_fk_published", merge(PUBLISHED, CHIR, published_fk))

# composition: force alone and force + chirality. The force term's index in
# fk_potential_fns differs between the configs, hence the explicit overrides.
COMPOSITION = {
    "gx": ({"+fk_potentials": "gxtb_force"}, {"+fk_potentials": "gxtb_force_chirality"}),
    "aim": (
        {"+fk_potentials": "aimnet_force", "model.fk_potential_fns.0.uncertainty_weight": 1.0},
        {
            "+fk_potentials": "aimnet_force_chirality",
            "model.fk_potential_fns.1.uncertainty_weight": 1.0,
        },
    ),
}
for tag, pot, home in (("gx", GX, "gxtb"), ("aim", AIM, "aimnet")):
    study = "B" if tag == "gx" else "C"
    force_only, force_chir = COMPOSITION[tag]
    fk = merge(GRID, pot, FK)

    def settings_bc(block, name, overrides, study=study, home=home):
        run(2, study, f"{study}{block}", name, 30, overrides, (home,))

    settings_bc(2, f"{tag}_fk_force", merge(GRID, FK, force_only))
    settings_bc(2, f"{tag}_fk_force_chir", merge(GRID, FK, force_chir))
    for lam in (10, 30, 50):
        settings_bc(3, f"{tag}_fk_lam{lam}", merge(fk, {"+model.fk_steering_temperature": lam}))
    settings_bc(4, f"{tag}_fk_freq2", merge(fk, {"+model.resample_freq": 2}))
    window_07 = {"+model.steering_t_start": 0.7, "+model.noise_t_start": 0.7}
    settings_bc(4, f"{tag}_fk_t07", merge(fk, window_07))
no_spread = {"model.fk_potential_fns.1.uncertainty_weight": 0.0}
run(2, "C", "C6", "aim_fk_u0", 30, merge(GRID, AIM, FK, no_spread), ("aimnet",))


def main():
    names = [r[3] for r in RUNS]
    assert len(names) == len(set(names)), "duplicate run names"
    rows = []
    for tier in sorted({r[0] for r in RUNS}):
        for seed in SEEDS:
            for t, study, block, name, n, over, surfaces in RUNS:
                if t != tier:
                    continue
                over = merge(
                    over, {"seed": seed, "model.num_samples": n, "data.batch_size": BATCH[n]}
                )
                rows.append(
                    dict(
                        study=study,
                        block=block,
                        run=f"{name}_s{seed}",
                        seed=seed,
                        n=n,
                        # "-", not empty: bash read with IFS=tab merges consecutive tabs
                        surfaces=",".join(surfaces) or "-",
                        overrides=" ".join(f"{k}={v}" for k, v in over.items()),
                    )
                )
    with open(HERE / "runs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["idx", *rows[0]], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows({"idx": i, **r} for i, r in enumerate(rows))

    # references first (they gate every check on their surface), then in run order
    reference_from = "goflow_N30_s1"
    checks = [
        dict(surface=s, run="reference", data_run=reference_from, chunk=c)
        for s in ("gxtb", "aimnet")
        for c in range(N_CHUNKS)
    ]
    for r in rows:
        for s in filter(lambda s: s != "-", r["surfaces"].split(",")):
            checks += [
                dict(surface=s, run=r["run"], data_run=r["run"], chunk=c) for c in range(N_CHUNKS)
            ]
    with open(HERE / "checks.tsv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            ["idx", "surface", "run", "data_run", "chunk", "n_chunks"],
            delimiter="\t",
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows({"idx": i, **c, "n_chunks": N_CHUNKS} for i, c in enumerate(checks))

    n_checks = len(checks) // N_CHUNKS
    print(f"{len(rows)} runs, {n_checks} Sella + IRC checks ({len(checks)} chunks)")
    for study in ("shared", "A", "B", "C"):
        print(f"  {study:6s} {sum(r['study'] == study for r in rows)} runs")


if __name__ == "__main__":
    main()
