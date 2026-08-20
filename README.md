# paralog-pairing-scripts

Julia scripts for protein-family coevolution and paralog-pairing methods
(DCA/PMI/mutual-information-based, and RBM-based via the
`RestrictedBoltzmannMachines`/`LatentAlignedRBMs` stack), developed against
the HK-RR and Pfam PF00072/PF00512/PF01339 paired-family datasets.

## Environment

These scripts depend on packages (`FASTX`, `BioSequences`, `Hungarian.jl`,
`RestrictedBoltzmannMachines`, etc.) declared in `env/Project.toml`, not the
default global Julia environment. Run scripts with:

```
julia --project=env <script>.jl
```

## Notes

- `MI_IPA.jl` — Julia port of the MI-IPA iterative paralog-pairing algorithm
  (Bitbol, PLoS Comput Biol 2018, https://github.com/anneflo/MI_IPA).
  Entry point: `run_mi_ipa(fasta_path; LengthA, Nincrement, kwargs...)`.
- Datasets, trained models, and result outputs are not included here — this
  repo holds source code only.
