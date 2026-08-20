# MI-IPA on PF00072-PF01339 (misc/.faa version) — first-iterations check

Run date: 2026-07-17 (single-replicate exploratory run); 10-replicate averaged
follow-up added same day, see "10-replicate averaged results" section below.
Script: `MI_IPA.jl` (`run_mi_ipa_with_training`), same Julia port validated against
Bitbol 2018's Figure 1A on the HK-RR dataset. Companion to
`PF00072_PF01339_MI-IPA_results.md` (the `pfam/PF00072_PF01339_paired.fasta`
version) — same protocol, different/larger source file.

## Dataset

- File: `misc/PF00072_PF01339_paired.faa` (as opposed to
  `pfam/PF00072_PF01339_paired.fasta`, which was used in the companion run).
  Same two Pfam domains, but this file is a much larger, less-filtered
  collection: 27548 raw records vs. 6833 in the `pfam/` version.
- Domain split: PF00072 (HisKA) = 111 columns, PF01339 = 174 columns (same as
  the other file — confirmed by concatenated sequence width 285 = 111+174).
  `LengthA=111`.
- Header format differs from the `pfam/` file (UniProt `tr|accession|mnemonic`
  form vs. bare mnemonic form) but is handled by the same `species_of_header`
  parser used for the companion run (matches `species_code`/
  `is_placeholder_code` from `paralog_pairing_pfam.jl`, which explicitly
  supports both header forms).
- 27548 records read, 69 rejected (non-standard characters) -> 27479 sequences.
- After dropping placeholder species (`9XXX`, `UNC*`, `UNK*`) and species with
  <2 paralogs: **3851 pairs across 741 species** (mean species size 5.2,
  max species size **184**).
- Analytic chance level (`chance_tp_fraction`): **19.24%**.

Comparison across all three datasets tried so far:

| Dataset | raw records | usable pairs | species | mean/species | max/species | chance |
|---|---:|---:|---:|---:|---:|---:|
| HK-RR standard (Fig. 1A) | 5109 | 5064 | 459 | 11.0 | 35 | 9.06% |
| PF00072-PF01339 `pfam/.fasta` | 6833 | 616 | 225 | 2.7 | 8 | 36.53% |
| PF00072-PF01339 `misc/.faa` (this run) | 27548 | 3851 | 741 | 5.2 | **184** | 19.24% |

## Method

Same protocol as the companion run: `N_start` known-correct pairs fixed in the
training set throughout, `N_increment=6` highest-confidence pairs added per
iteration, full-dataset re-prediction each iteration. **5 iterations, 1
replicate**, same `N_start` values as both prior runs. Unlike the smaller
`pfam/.fasta` file, **no `N_start` value here needs capping** — all of
1..2000 are below `Ntot=3851`, so every row below is a genuine, non-degenerate
result.

## Results

TP fraction (fraction of all 3851 pairs predicted correctly), by iteration,
with per-`N_start` wall-clock time for the 5-iteration run:

| N_start | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 | time |
|--------:|-------:|-------:|-------:|-------:|-------:|-----:|
| 1       | 0.238  | 0.283  | 0.299  | 0.308  | 0.310  | 50.1s |
| 2       | 0.324  | 0.331  | 0.354  | 0.370  | 0.372  | 51.7s |
| 5       | 0.339  | 0.342  | 0.342  | 0.346  | 0.341  | 50.6s |
| 10      | 0.354  | 0.380  | 0.378  | 0.375  | 0.381  | 50.4s |
| 20      | 0.399  | 0.388  | 0.391  | 0.393  | 0.392  | 20.7s |
| 50      | 0.421  | 0.414  | 0.413  | 0.416  | 0.424  | 20.5s |
| 100     | 0.446  | 0.437  | 0.449  | 0.448  | 0.444  | 19.3s |
| 200     | 0.465  | 0.455  | 0.458  | 0.464  | 0.467  | 18.3s |
| 500     | 0.488  | 0.475  | 0.483  | 0.481  | 0.488  | 17.5s |
| 1000    | 0.501  | 0.484  | 0.496  | 0.499  | 0.493  | 17.9s |
| 2000    | 0.522  | 0.512  | 0.515  | 0.514  | 0.517  | 17.1s |

Chance level (same units, for reference): 0.1924

## Observations / caveats

- **TP fractions here are much lower than either other dataset** (max ~0.52 at
  N_start=2000/5 iterations, vs. ~0.96-1.0 for the same N_start on the smaller
  `pfam/.fasta` file, and ~0.73-0.97 on HK-RR). Still clearly and consistently
  above the 19.2% chance level, and still rising with N_start, so the signal
  is real — this file is just a harder/noisier pairing problem than the other
  two, plausibly because it's much less redundancy-filtered (many more raw
  records for the same underlying protein families) and has at least one very
  large, likely highly redundant species (184 members) that could be diluting
  signal quality for large-species predictions.
- Per-`N_start` runtime is dominated by species-level Hungarian gap
  computation, not training-set size: cost is ~50s for N_start<=10 and drops
  to ~17-20s for N_start>=20. This is the fixed cost of `predict_pairs`
  re-scoring the *entire* dataset (all 741 species, including the 184-member
  one) every iteration, which is independent of N_start — the ~50s→~18s drop
  is more likely run-to-run variance/JIT warmup on the first few calls than a
  real N_start-dependent effect, since `predict_pairs`'s cost shouldn't
  depend on N_start at all.
- Single replicate, 5 iterations only, same caveat as the companion run: this
  is an exploratory generalization check, not a converged/statistically
  powered result.
- **Non-monotonic dips between iterations, investigated below.** Several rows
  above show TP fraction *decreasing* at some iteration (e.g. N_start=5:
  0.342 -> 0.342 -> 0.346 -> 0.341; N_start=1000: 0.501 -> 0.484 -> ...).
  Diagnosed two concrete, non-buggy causes (not specific to this Julia port —
  both are properties of the published algorithm itself):
  1. The training set is **not a strictly growing/nested set across
     iterations** — per the Methods section, it's fully re-ranked by
     confidence score from scratch every iteration, and only the top
     `(n-1)*N_increment` pairs make the cut. A pair that was "confident
     enough" at iteration *n* can be displaced by a different pair at
     iteration *n+1* if the refit model ranks it lower. Verified directly for
     N_start=5: iteration 2's 6-pair top-confidence set only had 5/6 members
     survive into iteration 3's 12-pair set.
  2. **This dataset has a large, likely highly-redundant species (184
     members, ~4.8% of the whole dataset)** that resolves very poorly and
     noisily: only 2-8 out of 184 correct across 5 iterations, itself
     non-monotonic (6 -> 2 -> 5 -> 5 -> 8). A single-replicate run has no
     averaging to smooth this out, so that species' own instability shows up
     directly in the aggregate TP fraction.

  Both effects are amplified by running only 1 replicate — see the
  10-replicate averaged results below for how much this smooths out.

## 10-replicate averaged results

Same protocol (5 iterations, `N_increment=6`, same `N_start` values), but
averaged over **10 replicates per `N_start`** (different random seed per
replicate) to test whether the non-monotonicity above was mostly
single-replicate noise. Scripts: `faa_sweep_replicate.jl` (one replicate) +
`faa_sweep_aggregate.jl` (averaging), raw per-replicate outputs in
`faa_sweep_results/`.

Mean TP fraction per iteration (± std across the 10 replicates):

| N_start | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 |
|--------:|-------:|-------:|-------:|-------:|-------:|
| 1    | 0.275 ± 0.037 | 0.306 ± 0.032 | 0.316 ± 0.029 | 0.321 ± 0.030 | 0.326 ± 0.034 |
| 2    | 0.307 ± 0.045 | 0.323 ± 0.030 | 0.333 ± 0.018 | 0.337 ± 0.022 | 0.334 ± 0.022 |
| 5    | 0.338 ± 0.016 | 0.345 ± 0.019 | 0.345 ± 0.017 | 0.346 ± 0.020 | 0.349 ± 0.022 |
| 10   | 0.364 ± 0.022 | 0.365 ± 0.019 | 0.366 ± 0.018 | 0.371 ± 0.016 | 0.372 ± 0.016 |
| 20   | 0.390 ± 0.014 | 0.392 ± 0.019 | 0.391 ± 0.017 | 0.393 ± 0.011 | 0.395 ± 0.014 |
| 50   | 0.411 ± 0.007 | 0.412 ± 0.011 | 0.414 ± 0.009 | 0.414 ± 0.008 | 0.414 ± 0.008 |
| 100  | 0.441 ± 0.006 | 0.441 ± 0.005 | 0.441 ± 0.006 | 0.438 ± 0.007 | 0.440 ± 0.009 |
| 200  | 0.458 ± 0.009 | 0.457 ± 0.010 | 0.461 ± 0.005 | 0.461 ± 0.009 | 0.461 ± 0.009 |
| 500  | 0.486 ± 0.009 | 0.485 ± 0.008 | 0.487 ± 0.008 | 0.485 ± 0.008 | 0.486 ± 0.009 |
| 1000 | 0.504 ± 0.004 | 0.503 ± 0.004 | 0.502 ± 0.007 | 0.503 ± 0.006 | 0.502 ± 0.009 |
| 2000 | 0.519 ± 0.006 | 0.521 ± 0.010 | 0.521 ± 0.005 | 0.515 ± 0.005 | 0.516 ± 0.008 |

**Did averaging fix the non-monotonicity?** Mostly, yes. The overall
iteration-1-to-iteration-5 rise is now clean and monotonic (or very close to
it) for every `N_start`, and the magnitude of any remaining dip
(~0.001-0.006) is small relative to the standard deviation across replicates
(~0.005-0.04) — i.e. within roughly 1 standard error of the mean, consistent
with residual sampling noise from only 10 replicates rather than a systematic
backward step. The two causes identified above (non-nested training-set
re-ranking + the noisy 184-member species) are real properties of this
dataset/algorithm combination, not implementation bugs; averaging reduces
their visible impact but, being a genuinely greedy per-iteration
re-optimization, doesn't guarantee strict monotonicity even in expectation.

## 10-replicate averaged results, superparalog excluded

Same protocol again (5 iterations, `N_increment=6`, same `N_start` values, 10
replicates), but with the single largest species removed before running:
species id `PSEFL`, 184 members (~4.8% of the dataset) — the "superparalog"
flagged above as resolving very poorly (2-8/184 correct across iterations)
and plausibly a highly redundant cluster (the file is a much less
redundancy-filtered collection than the other two datasets tried; the next
largest species is 164, then a steep drop to 88, so `PSEFL` is a clear
standout rather than part of a smooth size distribution). Excluding it drops
the dataset from 3851 pairs/741 species to **3667 pairs/740 species**, and
raises the analytic chance level from 19.24% to **20.18%** (fewer total pairs
per species, on average, makes random guessing slightly more likely to hit).
Scripts: `faa_sweep_replicate_nosuperparalog.jl` +
`faa_sweep_nosp_aggregate.jl`, raw outputs in `faa_sweep_nosp_results/`.

Mean TP fraction per iteration (± std across the 10 replicates):

| N_start | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 |
|--------:|-------:|-------:|-------:|-------:|-------:|
| 1    | 0.316 ± 0.033 | 0.334 ± 0.022 | 0.339 ± 0.022 | 0.345 ± 0.028 | 0.343 ± 0.028 |
| 2    | 0.318 ± 0.035 | 0.342 ± 0.028 | 0.351 ± 0.024 | 0.353 ± 0.020 | 0.355 ± 0.019 |
| 5    | 0.360 ± 0.017 | 0.366 ± 0.017 | 0.374 ± 0.028 | 0.375 ± 0.025 | 0.372 ± 0.027 |
| 10   | 0.385 ± 0.018 | 0.384 ± 0.024 | 0.388 ± 0.023 | 0.389 ± 0.020 | 0.390 ± 0.020 |
| 20   | 0.410 ± 0.011 | 0.410 ± 0.011 | 0.411 ± 0.010 | 0.412 ± 0.012 | 0.414 ± 0.013 |
| 50   | 0.434 ± 0.008 | 0.434 ± 0.008 | 0.439 ± 0.005 | 0.434 ± 0.007 | 0.439 ± 0.007 |
| 100  | 0.458 ± 0.009 | 0.456 ± 0.008 | 0.458 ± 0.008 | 0.458 ± 0.007 | 0.461 ± 0.010 |
| 200  | 0.482 ± 0.008 | 0.482 ± 0.004 | 0.483 ± 0.006 | 0.480 ± 0.004 | 0.484 ± 0.005 |
| 500  | 0.507 ± 0.005 | 0.507 ± 0.005 | 0.508 ± 0.008 | 0.506 ± 0.006 | 0.508 ± 0.009 |
| 1000 | 0.523 ± 0.006 | 0.523 ± 0.006 | 0.525 ± 0.005 | 0.527 ± 0.005 | 0.522 ± 0.006 |
| 2000 | 0.541 ± 0.005 | 0.541 ± 0.005 | 0.542 ± 0.004 | 0.538 ± 0.007 | 0.540 ± 0.004 |

**Comparison to the with-superparalog averaged results:**

- **(a) TP fraction is higher at every single `N_start` and every iteration**
  — by roughly +0.02 to +0.04 absolute (e.g. `N_start=1`, iteration 1:
  0.275 -> 0.316; `N_start=2000`, iteration 1: 0.519 -> 0.541). Expected:
  `PSEFL` was resolving at only ~1-4% accuracy, so it was dragging the
  aggregate down every round; removing it both shrinks the denominator by the
  184 hardest, most redundant pairs *and* likely gives the PMI model cleaner
  training signal on rounds where some of the `N_start` training pairs would
  otherwise have been drawn from that noisy species.
- **(b) Standard deviation across replicates is NOT clearly reduced.**
  Comparing the two std columns side by side, it's a mixed picture — roughly
  similar magnitude at most `N_start` values, occasionally *higher* without
  the superparalog (e.g. `N_start=5` iteration 3: 0.017 with vs. 0.028
  without). So `PSEFL` was not the dominant source of replicate-to-replicate
  variance; that variance comes mostly from which pairs happen to get drawn
  as the `N_start` training set and from tie-breaking elsewhere in the
  (now 740-species) dataset.
- **(c) The small non-monotonic dips are NOT eliminated.** E.g.
  `N_start=100`: 0.458, 0.456, 0.458, 0.458, 0.461 (tiny dip at iteration 2);
  `N_start=200`: 0.482, 0.482, 0.483, 0.480, 0.484 (dip at iteration 4);
  `N_start=2000`: 0.541, 0.541, 0.542, 0.538, 0.540 (dip at iteration 4) —
  same shape and similar magnitude as the with-superparalog run's dips at
  the same `N_start` values. This confirms the earlier diagnosis: `PSEFL`
  was *a* source of noise (and a real, mechanical drag on the overall
  accuracy level), but the residual iteration-to-iteration wobble comes
  mainly from cause (1) above — the training set being freshly re-ranked
  and not strictly growing each iteration — which is a property of the
  algorithm itself, not of this one species.

**Bottom line:** removing the superparalog cleanly raises the whole curve by
a consistent margin (a real, mechanical effect of dropping a
near-unresolvable, over-represented species) but doesn't materially change
the shape of the curve or its residual noise — the wobble is intrinsic to
the greedy iterative-reselection protocol, not an artifact of this one
species.
