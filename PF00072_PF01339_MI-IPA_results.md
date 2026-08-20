# MI-IPA on PF00072-PF01339 (Julia port) — first-iterations check

Run date: 2026-07-17
Script: `MI_IPA.jl` (`run_mi_ipa_with_training`), same Julia port validated against
Bitbol 2018's Figure 1A on the HK-RR dataset.

## Dataset

- File: `pfam/PF00072_PF01339_paired.fasta` (not `misc/PF00072_PF01339_paired.faa` —
  same underlying data, but this file's header format matches the `species_code`
  convention already used in `paralog_pairing_pfam.jl`).
- Domain split: PF00072 (HisKA) = 111 columns, PF01339 = 174 columns.
  `LengthA=111`, matching the `L_A=111` constant in `paralog_pairing_pfam.jl`.
- Species parsed with a custom `species_of_header` matching `species_code`/
  `is_placeholder_code` from `paralog_pairing_pfam.jl`: species = token after the
  last `_` in the identifier before `/`; placeholder codes (`9XXX`, `UNC*`, `UNK*`
  — UniProt genus-unknown mnemonics that lump many species together) are dropped.
- 6833 records in the file, 1 rejected (non-standard character), 6832 read.
- After dropping placeholder species and species with <2 paralogs:
  **616 pairs across 225 species** (mean species size 2.7, max 8).
  For comparison, the HK-RR standard dataset used for Fig. 1A had 5064 pairs /
  459 species (mean 11).
- Analytic chance level (`chance_tp_fraction`): **36.53%** (vs. HK-RR's 9.06%) —
  much higher, since small species make random guessing far more likely to hit
  (a size-2 species is already 50% by chance).
- Ground truth convention is the same as HK-RR: each fasta record's two domains
  come from the same protein/accession, so row `i`'s A-part and B-part are the
  true pair; TP = fraction of predicted (hk_row, rr_row) pairs with hk_row==rr_row.

## Method

Same protocol as Fig. 1A (Bitbol 2018 Methods): start from `N_start` known-correct
pairs (fixed in the training set throughout), grow the training set by
`N_increment` highest-confidence predicted pairs per iteration, re-predict over
the full dataset each iteration. Here: **5 iterations, 1 replicate**,
`N_increment=6`, same `N_start` values used for the HK-RR reproduction, to sanity
-check that the Julia port behaves sensibly on a different, unrelated dataset —
not a statistically powered run (no replicate averaging, few iterations).

## Results

TP fraction (fraction of all 616 pairs predicted correctly), by iteration:

| N_start | used | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 |
|--------:|-----:|-------:|-------:|-------:|-------:|-------:|
| 1       | 1    | 0.602  | 0.644  | 0.774  | 0.748  | 0.787  |
| 2       | 2    | 0.498  | 0.594  | 0.584  | 0.731  | 0.765  |
| 5       | 5    | 0.719  | 0.734  | 0.758  | 0.782  | 0.758  |
| 10      | 10   | 0.760  | 0.776  | 0.787  | 0.781  | 0.821  |
| 20      | 20   | 0.849  | 0.844  | 0.857  | 0.857  | 0.869  |
| 50      | 50   | 0.870  | 0.894  | 0.894  | 0.893  | 0.898  |
| 100     | 100  | 0.904  | 0.911  | 0.919  | 0.925  | 0.935  |
| 200     | 200  | 0.956  | 0.956  | 0.953  | 0.959  | 0.959  |
| 500     | 500  | 1.000  | 1.000  | 1.000  | 1.000  | 1.000  |
| 1000    | 616 (capped) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2000    | 616 (capped) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Chance level (same units, for reference): 0.365

## Caveats

- **N_start=500/1000/2000 are not meaningful here.** The dataset only has 616
  pairs total. N_start=1000 and 2000 both get capped to 616 (`min(N_start, Ntot)`
  in `run_mi_ipa_with_training`) — the entire dataset becomes the training set,
  leaving nothing to predict, so TP=1.0 is trivial/degenerate rather than signal.
  N_start=500 leaves only 116 test pairs, which the model gets perfectly. The
  informative part of the curve is N_start=1 through 200.
- Single replicate, 5 iterations only — this was an exploratory check that the
  algorithm generalizes correctly to a different dataset (different domain
  lengths, different header/species convention, much smaller and differently
  shaped species distribution), not a converged/statistically powered result.
  Re-run with more replicates and iterations (see `mi_ipa_fig1a_replicate.jl`
  for the pattern used on HK-RR) if a real curve is needed.
- The task is intrinsically easier here than HK-RR: TP is well above chance even
  at N_start=1, and rises to ~96% by N_start=200 in just a few iterations.
  Plausible reason: PF00072 and PF01339 are two domains fused in the *same*
  protein here (an intramolecular relationship), not two separate interacting
  proteins as in HK-RR, so there's less pairing ambiguity to resolve.
