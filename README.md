# Feynman-Kac Flow Matching

Research code for **Feynman-Kac-Flow: Inference Steering of Conditional Flow
Matching to an Energy-Tilted Posterior**. FK steering samples from a trained
conditional flow-matching model tilted by a potential, p₁(x) exp(−λ U(x)), at
inference time and without retraining: a set of particles is integrated as an
SDE with the same marginals as the flow and resampled by intermediate potentials
evaluated at the one-step estimate of the final sample.

The repository has one directory per part of the paper, each with its own
environment and README:

| Directory | Part of the paper | Environment |
| --- | --- | --- |
| [`hypercube/`](hypercube/README.md) | high-dimensional mode isolation, FK vs importance sampling | uv |
| [`synthetic/`](synthetic/README.md) | 2D distributions, FK vs guided flow matching | conda |
| [`chiflow/`](chiflow/README.md) | transition-state generation: chirality, E/Z and force (g-xTB, AIMNet2-rxn) steering of GoFlow, Sella and IRC checks | uv |

## Hypercube benchmark

A Schrödinger-bridge model maps a uniform distribution on [−1, 1]^d to Gaussians
at the corners of [−2, 2]^d; FK and importance sampling isolate one corner with
an indicator or a distance potential, for d = 3–15.

```bash
cd hypercube/
uv sync
./train_models.sh             # flow and score models for d = 3-15
./hypercube_exp_run_all.sh    # dimension and particle scaling; CSVs in results/
```

The distance to the target is the sliced Wasserstein-2 distance (SW₂, 4096
random projections); the result CSVs store its square.

## Synthetic 2D distributions

FK steering and guided flow matching
([Feng et al.](https://arxiv.org/abs/2502.02150)) on circle → S-curve,
uniform → 8 Gaussians and 8 Gaussians → moons, adapted from that work's code.

```bash
cd synthetic/
conda env create -f environment.yml && conda activate guided_flow
pip install -e .
bash script/train_cfm.sh      # base models; the notebooks run the comparison
```

## Transition states (ChiFlow)

GoFlow, trained on RDB7, generates a TS from the reactant and product graphs;
FK steering adds potentials for the R/S and E/Z configuration and for small
forces on the g-xTB or AIMNet2-rxn surface. Generated TSs are checked by a Sella
TS optimisation and an IRC on the same surfaces.

```bash
cd chiflow/
uv sync --group aimnet
bash preprocessing.sh
# trained checkpoint (Zenodo, below) -> artifacts/paper/flow3/checkpoints/flow3-paper.ckpt
```

Installation, data, the trained model and single runs are described in
[`chiflow/README.md`](chiflow/README.md); every run of the paper is defined in
[`chiflow/scripts/paper/`](chiflow/scripts/paper/README.md).

All run settings of the paper (scheduler, seeds, particle numbers, potentials,
steering window, noise) are defined in one place,
[`chiflow/scripts/paper/make_manifest.py`](chiflow/scripts/paper/make_manifest.py);
it writes the run lists `runs.tsv` and `checks.tsv` that the SLURM scripts read.

## Data and trained models

Zenodo: **DOI to be added** — the trained GoFlow checkpoint, the processed RDB7
data and split, the generated samples and Sella/IRC results of the paper, and
the inputs and outputs of the DFT validation calculations.

## Citation

```bibtex
@article{mark2025feynman,
  title={Feynman-Kac-Flow: Inference Steering of Conditional Flow Matching to an
 Energy-Tilted Posterior},
  author={Mark, Konstantin and Galustian, Leonard and Kovar, Maximilian P.-P. and Heid, Esther},
  journal={arXiv preprint arXiv:2509.01543},
  year={2025}
}
```

## License

MIT.

## Acknowledgments

- **GoFlow**: [Galustian et al.](https://chemrxiv.org/engage/chemrxiv/article-details/6850098f3ba0887c33dbd713)
- **RDB7**: [Spiekermann et al.](https://www.nature.com/articles/s41597-022-01529-6)
- **Guided flow matching**: ["On the Guidance of Flow Matching"](https://arxiv.org/abs/2502.02150); `synthetic/` is adapted from [AI4Science-WestlakeU/flow_guidance](https://github.com/AI4Science-WestlakeU/flow_guidance) (MIT, see `synthetic/LICENSE`)
- **GotenNet**: [Aykent and Xia](https://github.com/sarpaykent/GotenNet); `chiflow/gotennet/` is adapted from it (MIT, see `chiflow/gotennet/LICENSE`)
- **Flow matching**: [torchcfm](https://github.com/atong01/conditional-flow-matching)
- **AIMNet2**: [isayevlab/aimnetcentral](https://github.com/isayevlab/aimnetcentral); **g-xTB**: [grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb); **Sella**: [zadorlab/sella](https://github.com/zadorlab/sella)
