using CUDA
using HDF5
using RestrictedBoltzmannMachines: Potts, PottsGumbel, nsReLU, RBM, initialize!, standardize, unstandardize
using RestrictedBoltzmannMachines: log_pseudolikelihood, pcd!, save_rbm, load_rbm
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_v, mean_h_from_v, gpu, cpu
using RestrictedBoltzmannMachines: ∂free_energy
using Optimisers: Adam, ClipNorm, OptimiserChain
using Statistics: mean
using Random
using LatentAlignedRBMs

# CPU fallback: when the shared GPU is monopolized by another user's job for
# a long time (a recurring problem on this machine -- see session notes),
# PFAM_USE_CPU=1 runs the whole pipeline on CPU instead of blocking/waiting
# for GPU headroom. Deliberately an environment variable, not an ARGS
# position or part of the filename tag: device choice is an infrastructure
# decision, not a modeling one -- a CPU-trained and GPU-trained model from
# the same recipe are the same experiment and should be cache-interchangeable,
# not tracked as separate cached artifacts. `dev(x)` replaces direct `gpu(x)`
# calls throughout; CD_STEPS/steps=... etc. are untouched since the
# vector/broadcast code in RestrictedBoltzmannMachines.jl works transparently
# on plain Arrays, just without GPU acceleration.
const USE_GPU = lowercase(get(ENV, "PFAM_USE_CPU", "0")) ∉ ("1", "true", "yes")
dev(x) = USE_GPU ? gpu(x) : cpu(x)
println(USE_GPU ? "Device: GPU" : "Device: CPU (PFAM_USE_CPU=1)")
using FASTX
using BioSequences

# Overridable so multiple paired-model training runs can reuse the exact
# same RBM A/B checkpoints (whose own filenames never depend on SEED -- see
# single_suffix_A/B() below) while still exploring training-stochasticity
# variance in the paired model itself. Parsed here, ahead of the rest of the
# CONFIG block below, because it must be set before this Random.seed! call.
const SEED = length(ARGS) >= 14 ? parse(Int, ARGS[14]) : 42
Random.seed!(SEED)

# =============================================================================
# CONFIG
# =============================================================================
# Same paired-alignment layout as train_small_pfam.jl, but instead of keeping
# only one family's columns we use both: family A = sites 1:SPLIT_SITE, family
# B = sites SPLIT_SITE+1:end of the same paired sequences. Nothing downstream
# (N_VIS_A/N_VIS_B, q, everything derived from them) assumes a particular pair
# of Pfam families or domain lengths — both are read off the actual data — so
# any similarly-formatted "two domains concatenated end to end" FASTA works by
# just pointing FASTA_PATH/SPLIT_SITE at it (e.g. the PF00072-PF01339 pairing,
# same split site 111 but a 174-site second domain instead of PF00512's 66).
#
# Same ARGS convention as analyze_results_pfam.jl / pair_results_pfam.jl:
# N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN_A=30] [N_HIDDEN_B=30] [FASTA_PATH=./PF00072_PF00512_paired.fasta] [SPLIT_SITE=111] [N_ITERS_B=N_ITERS] [PAIRED_TRAIN_FRAC=1.0] [USE_REWEIGHTING=true] [REG_B=0] [REG_PAIRED=0] [REG_A=0] [SEED=42] [CD_STEPS=100] [BATCH_SIZE=256] [LR=1e-4] [CLIP_NORM=1.0] [PAIRED_TRAIN_N=0]
const FASTA_PATH  = length(ARGS) >= 6 ? ARGS[6] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE  = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 111   # family A ends here; family B is SPLIT_SITE+1:end
# Derived from the actual FASTA file (not hand-specified) so a run against a
# different dataset can never silently collide with another dataset's files
# under the same tag — see the DATASET_TAG mismatch this replaces, which used
# to hardcode "PF00072_PF00512" regardless of which FASTA_PATH was loaded.
const DATASET_TAG = "$(replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => ""))_l$(SPLIT_SITE)"
const OUTPUT_DIR  = "./results_pfam"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

## Previously hardcoded to 150/100 regardless of dataset -- fine for
# PF00072_PF00512 (domain A=111 sites > domain B=66 sites, so more hidden
# units for the larger domain matched), but backwards for PF00072_PF01339
# (domain B=174 sites > domain A=111 sites): the larger, more complex domain
# ended up with FEWER hidden units. Now ARGS-configurable (matching
# analyze_results_pfam.jl / pair_results_pfam.jl's existing convention)
# instead of hardcoded, so capacity can be allocated proportionally to each
# dataset's actual visible-site counts.
const N_HIDDEN_A  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const N_HIDDEN_B  = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 30
const H_ADD       = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
# CD_STEPS/BATCH_SIZE/LR/CLIP_NORM: now ARGS-configurable (appended after
# SEED so every existing call site's positions 1-14 are untouched) instead of
# hardcoded — these already flow unconditionally into single_suffix_A/B() and
# paired_suffix() below (see the comment there), so any previously-run
# combination of them still resolves to its own distinct, correctly-tagged
# filename; only the *source* of the value changed, not the tagging scheme.
const CD_STEPS    = length(ARGS) >= 15 ? parse(Int, ARGS[15]) : 100
const BATCH_SIZE  = length(ARGS) >= 16 ? parse(Int, ARGS[16]) : 256
const LR          = length(ARGS) >= 17 ? parse(Float32, ARGS[17]) : 1f-4
# Gradient-norm clipping (applied before Adam sees the gradient): with β1=0
# (no momentum smoothing on the raw gradient) and a slow-moving β2=0.999
# second-moment estimate, a single large gradient spike late in a long run
# can produce an enormous, poorly-normalized Adam step before the
# second-moment average catches up. Ported from TwoFamilyIsing/train_potts.jl,
# where this was found to cause a runaway weight-norm blowup in the
# added-unit block during long (10,000-step) paired training at lr=1e-4 —
# the same (β1=0, β2=0.999, no clipping) Adam configuration this script used
# before this fix, in a near-identical PCD/pcd! training setup. Clipping the
# raw gradient's global norm to CLIP_NORM before the Adam update removes the
# spike that triggers it.
const CLIP_NORM   = length(ARGS) >= 18 ? parse(Float64, ARGS[18]) : 1.0
const N_ITERS     = parse(Int, ARGS[1])   # RBM A's training length
# RBM B's training length. Defaults to N_ITERS (the common, backward-compatible
# case) but can be overridden independently — e.g. to give a domain that trains
# trickier than its partner more (or fewer) iterations without having to
# retrain the other, already-good RBM at a new length too.
const N_ITERS_B   = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : N_ITERS
# Fraction of the (species-grouped) training split that the PAIRED model
# actually gets to see, distinct from what RBM A/B train on. RBM A/B always
# train on the *entire* training split (real biology gives you plenty of
# single-family sequences); PAIRED_TRAIN_FRAC<1 restricts only the paired
# model to a smaller subset of it, simulating the realistic scenario where
# genuinely paired (both-domains-confirmed) sequences are much scarcer than
# single-family data. 1.0 (default) uses the whole training split, matching
# behavior before this feature existed.
const PAIRED_TRAIN_FRAC = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0
# Absolute-count alternative to PAIRED_TRAIN_FRAC, for sweeping "how many
# paired sequences did the model see" directly (256/512/1024/2048/...)
# instead of via an awkward fraction of a train_idx size that itself isn't a
# round number (e.g. PAIRED_TRAIN_FRAC=0.0132759425 to land on some N was
# already how earlier sweeps did this by hand -- this makes that the primary,
# exact, non-floating-point-fragile interface instead). 0 (default) means
# "not set, fall back to PAIRED_TRAIN_FRAC" -- fully backward compatible.
# Appended after CLIP_NORM (position 19) so every existing call site's
# positions 1-18 are untouched, same convention as every other parameter
# added after the fact in this file.
const PAIRED_TRAIN_N = length(ARGS) >= 19 ? parse(Int, ARGS[19]) : 0
# Toggle for sequence reweighting (see sequence_weights/weighted_resample
# below) — on by default (matches the pipeline's current behavior), but can
# be turned off to run/compare the pipeline without it. Does NOT affect the
# split's near-duplicate leakage merge (DEDUP_IDENTITY_THRESHOLD), which
# stays on regardless: that's a data-integrity fix, not an experimental
# modeling choice like reweighting is.
const USE_REWEIGHTING = length(ARGS) >= 10 ? parse(Bool, ARGS[10]) : true
const PAIRED_ITERS = parse(Int, ARGS[2])
const REG         = 0
# RBM B-specific override for l2l1_weights (Tubiana & Monasson group-sparsity
# regularization, eLife 2019 Eq. 8) -- defaults to REG (0) so existing runs
# are unaffected. Added to test whether penalizing weight magnitude smooths
# out the spurious-low-energy-region pathology observed in RBM B's Gibbs
# sampling free-energy trace (keeps decreasing indefinitely past its
# data-correlation peak, unlike RBM A's, which plateaus -- see session notes).
const REG_B       = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : REG
# Paired-model-specific override for l2l1_weights, independent of REG_B:
# REG_B only regularizes RBM B's own private training: the paired model's
# joint training (its free H_ADD block and the couplings tying it to both
# visible families) still used the always-0 global REG regardless of REG_B.
# This lets that be tested separately -- defaults to REG (0) so existing runs
# are unaffected.
const REG_PAIRED  = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : REG
# RBM A-specific override for l2l1_weights, symmetric to REG_B -- defaults to
# REG (0) so existing runs are unaffected.
const REG_A       = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : REG
const LOG_EVERY   = 100

# Periodic full-model snapshots, saved *in addition to* the final model, so a
# post-hoc pass (e.g. pair_results_pfam.jl run once per checkpoint) can pick
# whichever iteration actually has the best equilibrium behaviour instead of
# committing to a single guessed N_ITERS/PAIRED_ITERS up front. Motivated by
# the ab_check trajectories observed in earlier runs: the cross-family signal
# rose to a peak partway through paired training and then declined, so the
# final checkpoint is not necessarily the best one.
const CKPT_EVERY_SINGLE = 1000
const CKPT_EVERY_PAIRED = 500
const CKPT_DIR = joinpath(OUTPUT_DIR, "checkpoints")
isdir(CKPT_DIR) || mkpath(CKPT_DIR)

const TRAIN_FRAC  = 0.7
const VAL_FRAC    = 0.15

# Two DIFFERENT identity thresholds for two DIFFERENT purposes — do not
# conflate them:
#   DEDUP_IDENTITY_THRESHOLD: for the train/val split's leakage-prevention
#     merge (see grouped_train_val_split). Must be conservative/high: natural
#     protein family members are routinely >80% identical to each other, so a
#     low threshold here would wrongly merge huge swaths of the family into
#     one group — the same failure mode fixed for ambiguous species codes,
#     in a different guise.
#   REWEIGHT_IDENTITY_THRESHOLD: the standard DCA/Potts-model "Meff" sequence
#     reweighting threshold (Weigt et al. 2009, Morcos et al. 2011 and
#     essentially every DCA implementation since) — deliberately much lower,
#     since its purpose is to down-weight generically over-represented
#     lineages during training, not to detect near-identical duplicates.
const DEDUP_IDENTITY_THRESHOLD    = 0.97
const REWEIGHT_IDENTITY_THRESHOLD = 0.8

# N_HIDDEN_A/N_HIDDEN_B can now differ (e.g. asymmetric capacity experiments),
# and BATCH_SIZE/LR are no longer fixed across every run, so the suffix must
# carry all of them instead of assuming shared/fixed values.
#
# H_ADD deliberately does NOT appear in single_suffix(): RBM A/B training
# never touches H_ADD (it only sizes the paired model's free hidden block),
# so two runs that differ only in H_ADD should resolve to the exact same
# rbm_A/rbm_B files. This is what lets an H_ADD sweep train A/B once and
# reuse them for every paired-training run instead of redundantly retraining
# the (expensive) individual RBMs at every sweep point — see the
# isfile(path_rbm_A)/isfile(path_rbm_B) skip-training checks below.
# single_suffix_A/B are separate (not just one shared single_suffix()) so
# N_ITERS_B can differ from N_ITERS (RBM A's length) without touching RBM A's
# filename — e.g. retraining B alone for longer reuses the exact same rbm_A_*
# file untouched. When N_ITERS_B == N_ITERS (the common case) single_suffix_B()
# is byte-identical to single_suffix_A(), so nothing here changes existing
# filenames for runs that don't use this feature.
## CLIP=$(CLIP_NORM) is included unconditionally (not just when non-default)
# so that models trained before gradient clipping was introduced can never be
# silently reused as if they were trained under the new (clipped) optimizer —
# they simply resolve to a different filename now, so a run against this
# version of the script always retrains fresh rather than picking up a stale
# pre-clipping cache.
## SPLIT=bal is included unconditionally (same reasoning as CLIP=$(CLIP_NORM)
# above): the train/val split now groups sequences by species so paralogs
# never straddle the split (see grouped_train_val_split below), which draws a
# genuinely different random sequence than the old per-sequence split even
# with the same seed. A model cached under the old split's train_idx/val_idx
# would have its "held-out" guarantee silently broken if reused here, so this
# tag forces every run under this version of the script to retrain fresh
# rather than picking up a stale pre-grouping cache.
# RW=<0|1> reflects USE_REWEIGHTING and is always present in the filename
# (not just when non-default) for the same reason as CLIP/SPLIT above:
# training data is now a weighted bootstrap resample when reweighting is on
# (see weighted_resample), not the raw split, and the split itself now also
# merges near-duplicates (DEDUP_IDENTITY_THRESHOLD) on top of species+
# exact-duplicate grouping regardless of USE_REWEIGHTING — a model cached
# under the pre-reweighting/pre-near-dup code would silently look identical
# by name but was fit to different data, so this forces a fresh retrain
# under this version of the script rather than reusing it. The RW=0/RW=1
# distinction on top of that lets both modes coexist without colliding with
# each other, for direct with/without-reweighting comparisons.
# Canonicalizes an integer-valued Float64 (e.g. from `parse(Float64, ARGS[n])`)
# to the same string an untyped Int 0 literal would produce ("0", not "0.0"),
# so a REG_A/REG_B/REG_PAIRED explicitly passed as "0" on the command line
# resolves to the exact same filename as the default (untyped Int) fallback --
# every existing pre-regularization model file was saved with the bare "REG=0"
# spelling, so any mismatch here means silently failing to find it.
regstr(r) = isinteger(r) ? string(Int(r)) : string(r)
single_suffix_A() = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG_A))_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)"
single_suffix_B() = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS_B)_REG=$(regstr(REG_B))_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)"
single_suffix()   = single_suffix_A()

# =============================================================================
# CHECKPOINT-RESUME: find the best available warm-start point for a model
# =============================================================================
# Serves two related scenarios without special-casing: (1) resuming a run
# that was interrupted mid-way (same target N_ITERS as before), and (2)
# extending an already-completed shorter run to a longer target (different
# N_ITERS, everything else in the "recipe" unchanged) -- both just mean
# "start from the highest-iteration compatible checkpoint available, then
# train the remaining iterations". Caveat: only the model's PARAMETERS are
# restored -- pcd!'s persistent fantasy chains and the Adam optimiser's
# momentum/variance state are not saved/restored, so this is a warm start,
# not a bit-identical continuation of an uninterrupted run.
escape_regex(s::AbstractString) = replace(s, r"([.^$|()\[\]{}*+?\\])" => s"\\\1")

function split_on_niters(suffix::AbstractString, niters::Int; marker_name::AbstractString="N_ITERS")
    marker = "$(marker_name)=$(niters)"
    # findfirst("N_ITERS=...") on a suffix that also contains "PAIRED_ITERS=..."
    # would otherwise be ambiguous, since "PAIRED_ITERS=40" literally contains
    # "ITERS=40" but NOT "N_ITERS=40" as a substring (P-A-I-R-E-D-_-I-T-E-R-S
    # has no "N_" immediately before "ITERS") -- so a plain marker_name="N_ITERS"
    # search correctly targets only RBM A/B's own N_ITERS token. The paired
    # model instead passes marker_name="PAIRED_ITERS" explicitly to target its
    # own distinct token instead.
    idx = findfirst(marker, suffix)
    isnothing(idx) && error("$marker_name marker '$marker' not found in suffix '$suffix' -- suffix format changed?")
    prefix = suffix[1:first(idx)-1] * "$(marker_name)="
    after  = suffix[last(idx)+1:end]
    return prefix, after
end

# Returns (path, reached_iters) for the best (highest reached_iters ≤
# target_iters) compatible warm-start point across both completed final
# models (reached = their own embedded iteration marker) and mid-training
# checkpoints (reached = the trailing _iter=K; the checkpoint's own embedded
# marker value is that past run's TARGET, not the reached point, so it's
# wildcarded away rather than constrained to match target_iters), or
# (nothing, 0) if none. marker_name distinguishes RBM A/B's own "N_ITERS="
# token from the paired model's separate "PAIRED_ITERS=" token.
function find_resumable(model_prefix::AbstractString, suffix_fn, target_iters::Int; marker_name::AbstractString="N_ITERS")
    prefix, after = split_on_niters(suffix_fn(), target_iters; marker_name)
    best_path, best_iter = nothing, 0

    if isdir(OUTPUT_DIR)
        final_re = Regex("^" * escape_regex(model_prefix) * escape_regex(prefix) * "(\\d+)" * escape_regex(after) * "\\.hdf5\$")
        for f in readdir(OUTPUT_DIR)
            m = match(final_re, f)
            isnothing(m) && continue
            reached = parse(Int, m.captures[1])
            if reached <= target_iters && reached > best_iter
                best_iter = reached
                best_path = joinpath(OUTPUT_DIR, f)
            end
        end
    end

    if isdir(CKPT_DIR)
        ckpt_re = Regex("^" * escape_regex(model_prefix) * escape_regex(prefix) * "\\d+" * escape_regex(after) * "_iter=(\\d+)\\.hdf5\$")
        for f in readdir(CKPT_DIR)
            m = match(ckpt_re, f)
            isnothing(m) && continue
            reached = parse(Int, m.captures[1])
            if reached <= target_iters && reached > best_iter
                best_iter = reached
                best_path = joinpath(CKPT_DIR, f)
            end
        end
    end

    return best_path, best_iter
end
# paired_suffix() gains a distinguishing "_NITERSB=..." tag only when N_ITERS_B
# differs from N_ITERS, and a "_PAIREDFRAC=..." tag only when PAIRED_TRAIN_FRAC
# differs from 1.0, so the common case's filenames don't change just because
# these features exist.
n_iters_b_tag()  = N_ITERS_B == N_ITERS ? "" : "_NITERSB=$(N_ITERS_B)"
paired_frac_tag() = PAIRED_TRAIN_N > 0 ? "_PAIREDN=$(PAIRED_TRAIN_N)" :
    (PAIRED_TRAIN_FRAC == 1.0 ? "" : "_PAIREDFRAC=$(PAIRED_TRAIN_FRAC)")
# The paired model's own filename previously carried no trace of REG_B or
# REG_PAIRED (only the always-0 global REG), so every REG_B/REG_PAIRED
# combination silently shared and overwrote the SAME rbm_paired_*.hdf5 file
# -- confirmed by finding only one such file on disk despite three different
# REG_B values having been trained. pair_results_pfam.jl/analyze_results_pfam.jl
# must apply the identical tags (see the matching comments there) so all three
# scripts keep resolving to the same paired-model filename for a given run.
reg_b_tag()      = REG_B == REG ? "" : "_REGB=$(regstr(REG_B))"
reg_paired_tag() = REG_PAIRED == REG ? "" : "_REGPAIRED=$(regstr(REG_PAIRED))"
reg_a_tag()      = REG_A == REG ? "" : "_REGA=$(regstr(REG_A))"
# Deliberately absent from single_suffix_A/B() -- SEED only varies the
# paired model's own training stochasticity (its init + PCD chains), not
# which RBM A/B checkpoint gets loaded, so multiple SEED runs correctly
# reuse the same frozen A/B (see the SEED comment above Random.seed! for
# why this is safe: the split cache and A/B's isfile() checks never touch
# the RNG stream either).
seed_tag()       = SEED == 42 ? "" : "_SEED=$(SEED)"
paired_suffix()  = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(REG)_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)$(n_iters_b_tag())$(paired_frac_tag())$(reg_a_tag())$(reg_b_tag())$(reg_paired_tag())$(seed_tag())_PAIRED_ITERS=$(PAIRED_ITERS)"

path_rbm_A      = joinpath(OUTPUT_DIR, "rbm_A_$(single_suffix_A()).hdf5")
path_rbm_B      = joinpath(OUTPUT_DIR, "rbm_B_$(single_suffix_B()).hdf5")
path_rbm_paired = joinpath(OUTPUT_DIR, "rbm_paired_$(paired_suffix()).hdf5")
path_lpl_A      = joinpath(OUTPUT_DIR, "lpl_A_$(single_suffix_A()).txt")
path_lpl_B      = joinpath(OUTPUT_DIR, "lpl_B_$(single_suffix_B()).txt")
path_vh_A       = joinpath(OUTPUT_DIR, "vh_check_A_$(single_suffix_A()).txt")
path_vh_B       = joinpath(OUTPUT_DIR, "vh_check_B_$(single_suffix_B()).txt")
path_h_A        = joinpath(OUTPUT_DIR, "h_check_A_$(single_suffix_A()).txt")
path_h_B        = joinpath(OUTPUT_DIR, "h_check_B_$(single_suffix_B()).txt")
path_v_A        = joinpath(OUTPUT_DIR, "v_check_A_$(single_suffix_A()).txt")
path_v_B        = joinpath(OUTPUT_DIR, "v_check_B_$(single_suffix_B()).txt")
path_wn_A       = joinpath(OUTPUT_DIR, "wn_check_A_$(single_suffix_A()).txt")
path_wn_B       = joinpath(OUTPUT_DIR, "wn_check_B_$(single_suffix_B()).txt")
path_lpl_paired = joinpath(OUTPUT_DIR, "lpl_paired_$(paired_suffix()).txt")
path_freeze_paired = joinpath(OUTPUT_DIR, "freeze_check_paired_$(paired_suffix()).txt")
path_vh_paired  = joinpath(OUTPUT_DIR, "vh_check_paired_$(paired_suffix()).txt")
path_h_paired   = joinpath(OUTPUT_DIR, "h_check_paired_$(paired_suffix()).txt")
path_ab_paired  = joinpath(OUTPUT_DIR, "ab_check_paired_$(paired_suffix()).txt")
path_ab_val_paired = joinpath(OUTPUT_DIR, "ab_val_check_paired_$(paired_suffix()).txt")
path_firing_paired = joinpath(OUTPUT_DIR, "firing_check_paired_$(paired_suffix()).txt")
path_v_paired   = joinpath(OUTPUT_DIR, "v_check_paired_$(paired_suffix()).txt")
path_wn_paired  = joinpath(OUTPUT_DIR, "wn_check_paired_$(paired_suffix()).txt")
path_lplval_A       = joinpath(OUTPUT_DIR, "lplval_A_$(single_suffix_A()).txt")
path_lplval_B       = joinpath(OUTPUT_DIR, "lplval_B_$(single_suffix_B()).txt")
path_lplval_paired  = joinpath(OUTPUT_DIR, "lplval_paired_$(paired_suffix()).txt")
path_gn_A       = joinpath(OUTPUT_DIR, "gn_check_A_$(single_suffix_A()).txt")
path_gn_B       = joinpath(OUTPUT_DIR, "gn_check_B_$(single_suffix_B()).txt")
path_gn_paired  = joinpath(OUTPUT_DIR, "gn_check_paired_$(paired_suffix()).txt")

# =============================================================================
# DATA
# =============================================================================
# FASTA_PATH stores, per record, the two domains of the same protein
# concatenated end to end (see train_small_pfam.jl). Family A is sites
# 1:SPLIT_SITE, family B is sites SPLIT_SITE+1:end — the protein analogue of
# the synthetic two-family Ising system's XA/XB.
#
# Also captures each record's species code (see species_code below) so the
# train/val split can be grouped by species instead of by individual sequence
# — see grouped_train_val_split.
function load_sequences(path)
    reader = open(FASTA.Reader, path)
    sequences = LongAA[]
    species = String[]
    for record in reader
        push!(sequences, LongAA(FASTA.sequence(record)))
        push!(species, species_code(FASTA.identifier(record)))
    end
    close(reader)
    return sequences, species
end

# UniProt-style headers encode the species as the last underscore-separated
# field of the accession/species token, in either of two layouts seen across
# the FASTA files this pipeline has been pointed at:
#   "A0A010YSD8_9BACL/6-116"                       (bare accession_species/range)
#   "tr|A0A355I9E3|A0A355I9E3_UNCFI/8-112"          (UniProt tr|acc|acc_species/range)
# Splitting on "|" and taking the last field handles both (a no-op when there
# is no "|"), then stripping "/range" and taking the text after the last "_"
# gives the species code ("9BACL", "UNCFI", ...) in either case.
function species_code(id::AbstractString)
    id_part = split(id, "|")[end]
    id_part = split(id_part, "/")[1]
    parts = split(id_part, "_")
    return length(parts) >= 2 ? parts[end] : id_part
end

# UniProt uses digit-prefixed codes ("9BACT", "9GAMM", ...) and "UNC"/"UNK"
# codes ("UNCFI", ...) as generic placeholders for a higher taxonomic rank
# when the exact species isn't in their curated mnemonic list — NOT as an
# identifier for one specific species. In practice a single one of these
# codes can cover thousands of genuinely unrelated organisms (e.g. "9BACT"
# alone tags 3240+ sequences in the PF00072/PF01339 dataset), so grouping by
# them would wrongly force many unrelated sequences into one "paralog" group
# instead of just the true intra-species duplicates grouping is meant to
# catch — see grouped_train_val_split, which treats these as singletons.
function is_ambiguous_species(code::AbstractString)
    return occursin(r"^[0-9]", code) || occursin("UNC", code) || occursin("UNK", code)
end

# Minimal union-find (disjoint-set) used to merge species-code groups and
# exact-duplicate-sequence groups into one final partition below: two
# sequences end up in the same final group if EITHER criterion says so, even
# transitively (e.g. i~j by species code, j~k by duplicate sequence ⇒ i~j~k
# all one group). Path compression only (no union-by-rank) — the sequence
# counts here (tens of thousands) don't warrant the extra bookkeeping.
function dsu_find(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end
function dsu_union!(parent::Vector{Int}, x::Int, y::Int)
    rx, ry = dsu_find(parent, x), dsu_find(parent, y)
    rx != ry && (parent[rx] = ry)
end

# Fractional pairwise sequence identity, computed as a single GEMM: since a
# one-hot encoding has exactly one 1 per site, the dot product of two
# sequences' flattened one-hot vectors equals their number of matching sites,
# so (Xflat' * Xflat) / L gives identity directly — the same trick already
# used elsewhere in this pipeline for correlation-tensor computations. X is
# (q, L, N) onehot data; returns a dense N×N Float32 matrix (materialized
# back to the CPU — both the near-duplicate merge and the weight computation
# below need random access, which a GPU array doesn't give cheaply).
function pairwise_identity_matrix(X::AbstractArray)
    q, L, N = size(X)
    Xflat = dev(reshape(Float32.(X), q * L, N))
    return Array((Xflat' * Xflat) ./ Float32(L))
end

# Merges sequences with identity >= threshold into the same union-find group,
# in addition to whatever merges the caller already made (species code,
# exact duplicates) — catches near-duplicates a few point mutations away
# that exact string matching misses. O(N²) but fast in practice (~1s at
# N≈27,500 on this machine): a plain @inbounds double loop over the upper
# triangle, column-major order to match Julia's array layout.
function merge_near_duplicates!(parent::Vector{Int}, ident::AbstractMatrix, threshold::Real)
    n = size(ident, 1)
    n_merged = 0
    @inbounds for i in 1:n
        for j in (i+1):n
            if ident[j, i] >= threshold && dsu_find(parent, i) != dsu_find(parent, j)
                dsu_union!(parent, i, j)
                n_merged += 1
            end
        end
    end
    return n_merged
end

# Standard DCA/Potts-model "Meff" sequence weight: w_i = 1 / (number of
# sequences, including itself, at identity >= threshold to sequence i).
# A sequence with no close neighbors gets full weight 1; a sequence that's
# one of k near-duplicates gets weight 1/k, so the k of them collectively
# contribute about as much as one independent sample would.
function sequence_weights(ident::AbstractMatrix, threshold::Real)
    n = size(ident, 1)
    return [1.0 / count(>=(threshold), @view ident[:, i]) for i in 1:n]
end

# Weighted bootstrap resample (with replacement) of `idx`, drawn according to
# `weights` (same length as `idx`), returning `n_draws` indices. This is how
# sequence reweighting actually reaches training: StandardizedRBM's pcd! (see
# the RestrictedBoltzmannMachines.jl source) has no wts kwarg — unlike the
# base RBM/CenteredRBM methods, it always shuffles/minibatches its `data`
# argument uniformly — so there's no way to inject continuous per-sample
# weights into its gradient estimate without forking the library. Instead,
# pre-resampling the training indices according to `weights` before slicing
# out XA/XB/X_AB approximates the same thing: pcd! then reshuffles this fixed
# resampled set uniformly across many epochs, which converges to the desired
# weighted distribution in expectation without touching pcd! at all.
function weighted_resample(idx::Vector{Int}, weights::Vector{Float64}, n_draws::Int)
    cw = cumsum(weights)
    total = cw[end]
    return [idx[searchsortedfirst(cw, rand() * total)] for _ in 1:n_draws]
end

# Applies reweighting to `idx` when `ident` is a real matrix, or returns `idx`
# unchanged when it's `nothing` — the USE_REWEIGHTING toggle's single entry
# point. Callers pass `nothing` for `ident` (instead of computing it and
# discarding the result) when the toggle is off, so no wasted identity-matrix
# computation happens either.
function maybe_reweight(idx::Vector{Int}, ident::Union{Nothing,AbstractMatrix}, label::AbstractString)
    isnothing(ident) && return idx
    w = sequence_weights(ident, REWEIGHT_IDENTITY_THRESHOLD)
    println("$label reweighting: Meff=$(round(sum(w); digits=1)) / N=$(length(w)) (effective vs. raw sequence count)")
    return weighted_resample(idx, w, length(idx))
end

# Splits at the GROUP level so no two sequences that are (a) from the same
# genuinely-identified species, (b) exact duplicates of each other regardless
# of species tag, or (c) near-duplicates at >= DEDUP_IDENTITY_THRESHOLD
# identity, ever land on opposite sides — a per-sequence random split (the
# original behavior) can put near-identical paralogs on both sides, and
# species-code + exact-duplicate grouping alone (an earlier version of this
# function) still missed near-duplicates a few point mutations apart, plus
# exact duplicates that carry an ambiguous code or span multiple different
# codes (checked empirically: ~24% of this dataset is involved in an exact
# duplicate, and ~96% of those duplicate clusters were NOT already
# same-species-grouped) — all of these silently leak training information
# into what's supposed to be a held-out test.
#
# Sequences with an ambiguous species code (see is_ambiguous_species) are
# only merged with others via the duplicate/near-duplicate criteria, not
# treated as one group by their shared placeholder code, since a shared
# placeholder is not evidence they're paralogs of each other. `ident` is the
# full N×N pairwise identity matrix (see pairwise_identity_matrix), computed
# once by the caller and reused here — it's also reused later, sliced down to
# paired_train_idx, for the paired model's reweighting (see DATA section).
function grouped_train_val_split(seqs::Vector{LongAA}, species::Vector{String}, ident::AbstractMatrix, n_samples::Int, train_frac::Real, val_frac::Real)
    parent = collect(1:n_samples)

    n_ambiguous = 0
    species_first_seen = Dict{String, Int}()
    for (i, s) in enumerate(species)
        if is_ambiguous_species(s)
            n_ambiguous += 1
        elseif haskey(species_first_seen, s)
            dsu_union!(parent, i, species_first_seen[s])
        else
            species_first_seen[s] = i
        end
    end

    seq_first_seen = Dict{LongAA, Int}()
    n_dup = 0
    for (i, sq) in enumerate(seqs)
        if haskey(seq_first_seen, sq)
            dsu_union!(parent, i, seq_first_seen[sq])
            n_dup += 1
        else
            seq_first_seen[sq] = i
        end
    end
    n_near_dup = merge_near_duplicates!(parent, ident, DEDUP_IDENTITY_THRESHOLD)
    println("Species codes: $n_ambiguous / $n_samples sequences have an ambiguous (9XXX/UNCxx/UNKxx) code")
    println("Exact duplicates: $n_dup / $n_samples sequences are an exact duplicate of an earlier one")
    println("Near-duplicates: $n_near_dup additional merges at >= $(DEDUP_IDENTITY_THRESHOLD) identity")

    groups = Dict{Int, Vector{Int}}()
    for i in 1:n_samples
        push!(get!(groups, dsu_find(parent, i), Int[]), i)
    end
    group_keys = collect(keys(groups))

    n_train_target = floor(Int, train_frac * n_samples)
    n_val_target   = floor(Int, val_frac * n_samples)

    # Balanced greedy (LPT-style) allocation instead of "shuffle then fill
    # train-first". A handful of large species/paralog groups (thousands of
    # sequences each) previously landed entirely in train or entirely in val
    # depending on luck of the shuffle order, which swung the held-out set's
    # composition enough to make downstream cross-family overlap metrics
    # wildly unstable across reruns (observed range 0.28-0.71 across 5 seeds
    # at the same TOP_FRAC). Placing the largest groups first and always
    # routing each group to whichever bucket is proportionally further below
    # its target fraction keeps both splits' composition stable regardless
    # of shuffle order; only the many small/singleton groups (low individual
    # impact) are left to the randomized tie-break.
    shuffled_keys = group_keys[randperm(length(group_keys))]
    sorted_keys = sort(shuffled_keys; by = k -> -length(groups[k]), alg = MergeSort)

    train_idx = Int[]
    val_idx   = Int[]
    for k in sorted_keys
        idxs = groups[k]
        train_open = length(train_idx) < n_train_target
        val_open   = length(val_idx)   < n_val_target
        fill_train = n_train_target > 0 ? length(train_idx) / n_train_target : 1.0
        fill_val   = n_val_target   > 0 ? length(val_idx)   / n_val_target   : 1.0
        if train_open && (!val_open || fill_train <= fill_val)
            append!(train_idx, idxs)
        elseif val_open
            append!(val_idx, idxs)
        end
        # else: both targets already met — this group is left in neither
        # split (matches the old per-sequence split's behavior when
        # TRAIN_FRAC+VAL_FRAC < 1)
    end
    println("Groups (species+duplicate merged): $(length(group_keys)) total, $(length(train_idx)) seqs → train (target $n_train_target), $(length(val_idx)) seqs → val (target $n_val_target)")
    return train_idx, val_idx
end

seqs_raw, seqs_species = load_sequences(FASTA_PATH)
seqs_onehot = LatentAlignedRBMs.onehot(seqs_raw)  # (q, n_sites_total, n_samples) BitArray

q          = size(seqs_onehot, 1)
n_sites    = size(seqs_onehot, 2)
n_samples  = size(seqs_onehot, 3)
N_VIS_A    = SPLIT_SITE
N_VIS_B    = n_sites - SPLIT_SITE

# Computed once over the WHOLE dataset (full concatenated sequence, matching
# what exact-duplicate detection already operates on) — used both for the
# split's near-duplicate leakage merge right below, and reused (sliced down
# to paired_train_idx, no recomputation needed) for the paired model's
# reweighting further down.
#
# Split/identity cache: full_ident + the resulting train/val split are
# entirely determined by FASTA_PATH + TRAIN_FRAC/VAL_FRAC/
# DEDUP_IDENTITY_THRESHOLD (Random.seed!(42) above pins the one randperm()
# grouped_train_val_split uses) -- identical across every REG_B/H_ADD/N_ITERS/
# device combination, and shared with pair_results_pfam.jl (same cache file,
# same key, ported verbatim). full_ident itself is a dense N×N matmul that's
# fast on GPU but slow on CPU with nothing printed while it runs, so cache it
# once instead of repaying that cost every run.
fasta_tag = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
split_cache_path = joinpath(OUTPUT_DIR, "split_cache_$(fasta_tag)_TRAIN=$(TRAIN_FRAC)_VAL=$(VAL_FRAC)_DEDUP=$(DEDUP_IDENTITY_THRESHOLD).hdf5")

if isfile(split_cache_path)
    println("--- Train/val split: found cached split, skipping identity matrix + split computation ---")
    println("Loading ← $split_cache_path")
    full_ident = h5read(split_cache_path, "full_ident")
    train_idx  = h5read(split_cache_path, "train_idx")
    val_idx    = h5read(split_cache_path, "val_idx")
    @assert size(full_ident, 1) == n_samples "cached split's dataset size ($(size(full_ident,1))) doesn't match current FASTA ($n_samples) -- delete $split_cache_path and rerun"
else
    full_ident = pairwise_identity_matrix(seqs_onehot)
    # Held-out validation split, paired sequences never trained on, grouped by
    # species/duplicates so paralogs never straddle the split (see
    # grouped_train_val_split above). Family A/B keep the same sample indices so
    # the pairing between them is preserved in both splits.
    train_idx, val_idx = grouped_train_val_split(seqs_raw, seqs_species, full_ident, n_samples, TRAIN_FRAC, VAL_FRAC)
    h5open(split_cache_path, "w") do io
        write(io, "full_ident", full_ident)
        write(io, "train_idx", train_idx)
        write(io, "val_idx", val_idx)
    end
    println("Saved train/val split + identity matrix → $split_cache_path")
end

# paired_train_idx is determined here, immediately after the split and
# before any reweighting below, deliberately: train_idx/val_idx/
# paired_train_idx are the only index sets that MUST reproduce byte-identically
# in pair_results_pfam.jl (they define hard leakage boundaries / what the
# paired model was and wasn't trained on); the reweighted resampling below
# does not need to match exactly between the two scripts (any fresh draw from
# the same weight distribution is scientifically equivalent), so keeping its
# rand() calls after this point means pair_results_pfam.jl only has to mirror
# the code up to here to stay in sync, not the reweighting steps too.
#
# The paired model trains on paired_train_idx, a (possibly smaller) subset of
# train_idx — a uniform random subsample when PAIRED_TRAIN_FRAC<1, simulating
# scarcer genuinely-paired data; the full train_idx (byte-identical to X_AB in
# every run before this feature existed) when PAIRED_TRAIN_FRAC==1.
n_paired_train = PAIRED_TRAIN_N > 0 ? min(PAIRED_TRAIN_N, length(train_idx)) :
    floor(Int, PAIRED_TRAIN_FRAC * length(train_idx))
paired_train_idx = (PAIRED_TRAIN_N == 0 && PAIRED_TRAIN_FRAC == 1.0) ? train_idx :
    train_idx[randperm(length(train_idx))[1:n_paired_train]]
println("Paired training subset: $(length(paired_train_idx)) / $(length(train_idx)) train sequences " *
        (PAIRED_TRAIN_N > 0 ? "(PAIRED_TRAIN_N=$(PAIRED_TRAIN_N))" : "(PAIRED_TRAIN_FRAC=$(PAIRED_TRAIN_FRAC))"))

# RBM A/B always train on the *entire* training split (see PAIRED_TRAIN_FRAC
# comment above). Sequence reweighting (see maybe_reweight/sequence_weights/
# weighted_resample above; toggled by USE_REWEIGHTING) is computed
# SEPARATELY per model rather than once on the full concatenated sequence:
# two sequences redundant in family B but unique in family A should count as
# one effective sample for RBM B but two for RBM A, so each model's weights
# are computed only over the columns and (for the paired model) the training
# subset it actually sees.
identA = USE_REWEIGHTING ? pairwise_identity_matrix(seqs_onehot[:, 1:SPLIT_SITE, train_idx]) : nothing
resampled_train_idx_A = maybe_reweight(train_idx, identA, "RBM A")
XA = seqs_onehot[:, 1:SPLIT_SITE, resampled_train_idx_A]

identB = USE_REWEIGHTING ? pairwise_identity_matrix(seqs_onehot[:, SPLIT_SITE+1:end, train_idx]) : nothing
resampled_train_idx_B = maybe_reweight(train_idx, identB, "RBM B")
XB = seqs_onehot[:, SPLIT_SITE+1:end, resampled_train_idx_B]

println("XA: $(size(XA))  XB: $(size(XB))")

# Reweighting for the paired model uses the full concatenated sequence (both
# domains, matching what it actually trains on), restricted to
# paired_train_idx — sliced out of full_ident (already computed over every
# sequence) rather than recomputed from scratch.
ident_paired = USE_REWEIGHTING ? full_ident[paired_train_idx, paired_train_idx] : nothing
resampled_paired_idx = maybe_reweight(paired_train_idx, ident_paired, "Paired model")

XA_paired = seqs_onehot[:, 1:SPLIT_SITE, resampled_paired_idx]
XB_paired = seqs_onehot[:, SPLIT_SITE+1:end, resampled_paired_idx]
X_AB = cat(XA_paired, XB_paired; dims=2)

println("X_AB (paired training data): $(size(X_AB))")

XA_val = seqs_onehot[:, 1:SPLIT_SITE, val_idx]
XB_val = seqs_onehot[:, SPLIT_SITE+1:end, val_idx]
X_AB_val = cat(XA_val, XB_val; dims=2)

println("XA_val: $(size(XA_val))  XB_val: $(size(XB_val))  X_AB_val: $(size(X_AB_val))")

# =============================================================================
# PARAMETER PROJECTION
# =============================================================================
function frozen_block_ranges(rbm_A, rbm_B)
    n_vis_A, n_hid_A = size(rbm_A.w, 2), size(rbm_A.w, 3)
    n_vis_B, n_hid_B = size(rbm_B.w, 2), size(rbm_B.w, 3)
    n_vis_total = n_vis_A + n_vis_B

    vis_A = 1:n_vis_A
    vis_B = (n_vis_A + 1):n_vis_total
    hid_A = 1:n_hid_A
    hid_B = (n_hid_A + 1):(n_hid_A + n_hid_B)
    return (; vis_A, vis_B, hid_A, hid_B)
end

function project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)
    (; vis_A, vis_B, hid_A, hid_B) = frozen_block_ranges(rbm_A, rbm_B)

    rbm_paired.visible.par[:, :, vis_A] .= rbm_A.visible.par
    rbm_paired.visible.par[:, :, vis_B] .= rbm_B.visible.par
    rbm_paired.hidden.par[:, hid_A]     .= rbm_A.hidden.par
    rbm_paired.hidden.par[:, hid_B]     .= rbm_B.hidden.par

    rbm_paired.w[:, vis_A, hid_A] .= rbm_A.w
    rbm_paired.w[:, vis_A, hid_B] .= 0
    rbm_paired.w[:, vis_B, hid_A] .= 0
    rbm_paired.w[:, vis_B, hid_B] .= rbm_B.w
end

# =============================================================================
# FREEZE CHECKPOINT
# =============================================================================
# Since `pcd!` runs the Adam update on *all* parameters every iteration,
# freezing rbm_A's/rbm_B's blocks inside rbm_paired is implemented by resetting
# them back to the reference values after each step (see `freeze_callback`
# below), rather than by excluding them from the gradient. `freeze_drift`
# measures, for each frozen block, how far its current value sits from what it
# should be. Called right before the reset it shows how much a single Adam
# step tried to move a supposedly-frozen block (should be small but nonzero);
# called right after the reset it must be exactly zero, or the freeze is
# broken (e.g. wrong index ranges, aliasing, an array not actually written).
const FREEZE_TOL = 1f-6

function freeze_drift(rbm_paired, rbm_A, rbm_B)
    (; vis_A, vis_B, hid_A, hid_B) = frozen_block_ranges(rbm_A, rbm_B)
    return (
        vis_A = maximum(abs, rbm_paired.visible.par[:, :, vis_A] .- rbm_A.visible.par),
        vis_B = maximum(abs, rbm_paired.visible.par[:, :, vis_B] .- rbm_B.visible.par),
        hid_A = maximum(abs, rbm_paired.hidden.par[:, hid_A] .- rbm_A.hidden.par),
        hid_B = maximum(abs, rbm_paired.hidden.par[:, hid_B] .- rbm_B.hidden.par),
        w_AA  = maximum(abs, rbm_paired.w[:, vis_A, hid_A] .- rbm_A.w),
        w_BB  = maximum(abs, rbm_paired.w[:, vis_B, hid_B] .- rbm_B.w),
        w_AB  = maximum(abs, rbm_paired.w[:, vis_A, hid_B]),  # cross block, must stay 0
        w_BA  = maximum(abs, rbm_paired.w[:, vis_B, hid_A]),  # cross block, must stay 0
    )
end

function log_freeze_checkpoint(path_freeze, iter, drift_before, drift_after)
    worst_before = maximum(drift_before)
    worst_after  = maximum(drift_after)
    status = worst_after <= FREEZE_TOL ? "OK" : "VIOLATION"
    open(path_freeze, "a") do f
        println(f, "iter=$iter status=$status drift_before_reset=$drift_before drift_after_reset=$drift_after")
    end
    println("iter=$iter  freeze_check=$status  max_drift_before_reset=$worst_before  max_drift_after_reset=$worst_after")
    if worst_after > FREEZE_TOL
        @warn "Frozen parameters did not reset to their reference values" iter drift_after
    end
end

# =============================================================================
# VH CORRELATION CHECKPOINT (are the added hidden units learning?)
# =============================================================================
# For an RBM, the weight gradient is driven by the discrepancy between the
# data-driven and model-driven visible-hidden correlations, <v h>_data and
# <v h>_model (the sufficient statistics of contrastive divergence). If that
# discrepancy is ~0 throughout training for a hidden unit, its incoming
# weights get essentially no learning signal — either because it has already
# converged, or because it is "dead" (e.g. a ReLU stuck at 0 for both data and
# fantasy particles). We track this specifically for the H_ADD extra hidden
# units in rbm_paired, since those are the only ones actually free to learn
# (see FREEZE CHECKPOINT above).
function added_hidden_range(rbm_A, rbm_B, h_add)
    n_hid_A = size(rbm_A.w, 3)
    n_hid_B = size(rbm_B.w, 3)
    return (n_hid_A + n_hid_B + 1):(n_hid_A + n_hid_B + h_add)
end

# <v h> averaged over the batch: v is (q, n_vis, batch), h is (n_hid, batch);
# returns (q, n_vis, n_hid).
function vh_correlation(v, h)
    q, n_vis, batch = size(v)
    vr = reshape(Float32.(v), q * n_vis, batch)
    C  = (vr * h') ./ batch
    return reshape(C, q, n_vis, size(h, 1))
end

# Ordinary least-squares fit of y ~ slope*x + intercept, plus Pearson r.
# A perfectly-trained RBM has <vh>_model tracking <vh>_data along y=x, i.e.
# r → 1, slope → 1, intercept → 0 — not merely a small |x - y|, which can look
# deceptively good if both are just small and noisy.
function linear_alignment(x, y)
    mx, my = mean(x), mean(y)
    cov_xy = mean((x .- mx) .* (y .- my))
    var_x  = mean((x .- mx) .^ 2)
    var_y  = mean((y .- my) .^ 2)
    r         = cov_xy / sqrt(var_x * var_y)
    slope     = cov_xy / var_x
    intercept = my - slope * mx
    return (; r, slope, intercept)
end

function vh_learning_checkpoint(rbm, vd, vm, hid_add)
    hd = mean_h_from_v(rbm, vd)[hid_add, :]
    hm = mean_h_from_v(rbm, vm)[hid_add, :]

    cd = vh_correlation(vd, hd)
    cm = vh_correlation(vm, hm)
    diff  = cd .- cm
    align = linear_alignment(vec(cd), vec(cm))

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
        mean_h_data   = Array(vec(mean(hd; dims=2))),
        mean_h_model  = Array(vec(mean(hm; dims=2))),
    )
end

function log_vh_checkpoint(path_vh, iter, chk)
    open(path_vh, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept) mean_h_data=$(chk.mean_h_data) mean_h_model=$(chk.mean_h_model)")
    end
    println("iter=$iter  vh_check  mean|<vh>_data-<vh>_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# H CORRELATION CHECKPOINT (single hidden-unit means, data vs model)
# =============================================================================
# Companion to the VH CHECKPOINT above: that one checks the *joint* <v h>
# statistic; this checks the *marginal* <h> alone — one point per hidden
# unit, its mean activation under the data distribution vs under the model —
# Pearson-correlated against each other exactly like vh_learning_checkpoint
# does. This is the training-time analogue of the ⟨h⟩ row in pair_results_pfam.jl's
# post-training scatter validation, so the two can be read side by side: at
# the end of training, this curve's last value should be in the same ballpark
# as the equilibrium-sample ⟨h⟩ scatter's correlation.
function h_learning_checkpoint(rbm, vd, vm, hid_add)
    hd = mean_h_from_v(rbm, vd)[hid_add, :]
    hm = mean_h_from_v(rbm, vm)[hid_add, :]
    mean_hd = vec(mean(hd; dims=2))
    mean_hm = vec(mean(hm; dims=2))
    diff  = mean_hd .- mean_hm
    align = linear_alignment(mean_hd, mean_hm)

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
    )
end

function log_h_checkpoint(path_h, iter, chk)
    open(path_h, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  h_check  mean|<h>_data-<h>_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# V CORRELATION CHECKPOINT (single visible-unit/colour means, data vs model)
# =============================================================================
# Companion to the V-V CHECKPOINT below: that one checks the *pairwise*
# <v_i v_j> statistic across distinct sites; this checks the *marginal* <v>
# alone — one point per (site, colour), mean occupancy under data vs under
# model. The training-time analogue of pair_results_pfam.jl's ⟨v⟩ scatter row.
function v_learning_checkpoint(vd, vm)
    q, n_vis, _ = size(vd)
    mean_vd = vec(mean(reshape(Float32.(vd), q * n_vis, :); dims=2))
    mean_vm = vec(mean(reshape(Float32.(vm), q * n_vis, :); dims=2))
    diff  = mean_vd .- mean_vm
    align = linear_alignment(mean_vd, mean_vm)

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
    )
end

function log_v_checkpoint(path_v, iter, chk)
    open(path_v, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  v_check  mean|<v>_data-<v>_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# A-B CORRELATION CHECKPOINT (is joint training capturing cross-family structure?)
# =============================================================================
# The whole point of rbm_paired's extra hidden units is to let the model
# represent correlations *between* family A and family B — the frozen private
# blocks can't touch these by construction (the A-B weight cross-blocks are
# pinned at 0, see FREEZE CHECKPOINT above). So the metric that actually
# answers "did joint training help" is the connected two-point correlation
# between visible sites of A and visible sites of B, compared between data and
# model samples — not just something about the added hidden units themselves.
#
# For Potts/one-hot visible units, the connected correlation between site i
# (color a) and site j (color b) is
#   C[a,i,b,j] = <v_i=a v_j=b> - <v_i=a><v_j=b>
# computed once under the data distribution (from vd) and once under the
# model (from the persistent fantasy chains vm), restricted to i ∈ family A,
# j ∈ family B. In a well-trained joint model these two tensors should align
# along y=x (r → 1, slope → 1, intercept → 0), exactly like the vh check above.
function vv_connected_correlation(v_i, v_j)
    q, n_i, batch = size(v_i)
    n_j = size(v_j, 2)
    vi = reshape(Float32.(v_i), q * n_i, batch)
    vj = reshape(Float32.(v_j), q * n_j, batch)
    mean_i = vec(mean(vi; dims=2))
    mean_j = vec(mean(vj; dims=2))
    cross  = (vi * vj') ./ batch
    connected = cross .- mean_i * mean_j'
    return reshape(connected, q, n_i, q, n_j)
end

function ab_correlation_checkpoint(vd, vm, vis_A, vis_B)
    c_data  = vv_connected_correlation(vd[:, vis_A, :], vd[:, vis_B, :])
    c_model = vv_connected_correlation(vm[:, vis_A, :], vm[:, vis_B, :])
    diff  = c_data .- c_model
    align = linear_alignment(vec(c_data), vec(c_model))

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
    )
end

function log_ab_checkpoint(path_ab, iter, chk)
    open(path_ab, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  ab_check  mean|C_data-C_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# A-B CORRELATION CHECKPOINT, HELD-OUT (does cross-family learning generalize?)
# =============================================================================
# ab_correlation_checkpoint above compares the *training minibatch*'s
# cross-family connected correlation against the model's (persistent fantasy
# particles) — it can't distinguish genuine cross-family structure from
# something the added units have fit to that particular minibatch. This is
# the intended replacement for tracking lpl_paired/lplval_paired: raw
# pseudolikelihood is dominated by the 250 frozen hidden units and is
# essentially insensitive to what the H_ADD free units do (see the flat
# lplval_paired curves from the K=50/K=100 runs); ab_check/ab_check_val are
# restricted to exactly the connected correlation those free units can move,
# since the frozen blocks' cross terms are pinned at 0 by construction.
#
# c_data_val depends only on data, not on the current model, so it is
# computed once before training starts (see train_rbm! below) and reused at
# every checkpoint; only the model-side statistic (from the current
# persistent fantasy particles vm) is recomputed each time, keeping the added
# cost essentially free — no extra Gibbs sampling, unlike a held-out vv_check
# would require.
function ab_val_correlation_checkpoint(vm, vis_A, vis_B, c_data_val)
    c_model = vv_connected_correlation(vm[:, vis_A, :], vm[:, vis_B, :])
    diff  = c_data_val .- c_model
    align = linear_alignment(vec(c_data_val), vec(c_model))

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
    )
end

function log_ab_val_checkpoint(path_ab_val, iter, chk)
    open(path_ab_val, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  ab_check_val  mean|C_heldout-C_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# Note: a training-time V-V (pairwise ⟨v_i v_j⟩, data vs model) checkpoint
# used to live here, mirroring the A-B check below but for a single machine's
# own visible layer. It was computed every LOG_EVERY iterations via a large
# CPU nested loop over every (site pair, colour pair) — for RBM A alone that's
# q^2 x n_vis x (n_vis-1) ~ 5.4M entries, repeated ~100 times over training —
# and was removed as too computationally costly to run *during* training.
# The equivalent check on real equilibrium samples, run once (not every
# checkpoint), still exists post-training as the ⟨vv⟩ panel in
# pair_results_pfam.jl's moments_validation figure.

# =============================================================================
# FIRING-RATE CHECKPOINT (per added-unit activity, distinct from alignment)
# =============================================================================
# nsReLU units are a dReLU mixture: each draw lands either on the positive
# branch (h > 0) or the negative branch (h < 0), with a mixture weight that
# depends on the input. `r`/`slope` in the vh check can look reasonable for a
# unit that is nevertheless collapsed — e.g. one that fires on almost every
# sample regardless of v, or almost never does — because those alignment
# stats are computed from *mean* activations, which don't reveal how
# concentrated the underlying samples are. Firing rate = fraction of samples
# with h_j > 0, computed separately from actual samples (not mean-field
# values) of data (vd) and model (vm). A unit stuck near 0 or 1 for both,
# and staying that way across checkpoints, isn't discriminating between
# inputs — regardless of what its mean-based correlation stats say.
function firing_rate_checkpoint(rbm, vd, vm, hid_add)
    hd = sample_h_from_v(rbm, vd)[hid_add, :]
    hm = sample_h_from_v(rbm, vm)[hid_add, :]
    return (
        data_rate  = Array(vec(mean(hd .> 0; dims=2))),
        model_rate = Array(vec(mean(hm .> 0; dims=2))),
    )
end

function log_firing_checkpoint(path_firing, iter, chk)
    open(path_firing, "a") do f
        println(f, "iter=$iter data_rate=$(chk.data_rate) model_rate=$(chk.model_rate)")
    end
    println("iter=$iter  firing_check  data_rate=$(chk.data_rate)  model_rate=$(chk.model_rate)")
end

# =============================================================================
# WEIGHT-NORM CHECKPOINT (are the tracked units acquiring structure at all?)
# =============================================================================
# The vh/vv/firing checks all measure *what a unit is doing with its current
# weights*. This instead tracks the weights themselves: the per-unit Frobenius
# norm of the incoming visible weight column, ‖w[:, :, j]‖, for each unit in
# `hid_add`. A unit whose norm stays flat near its (typically near-zero, see
# `initialize!`) starting value isn't acquiring any structure regardless of
# what its correlation/firing stats look like; a growing norm means the
# optimizer is actually shaping that unit's receptive field. Logged at iter=0
# (the pre-training baseline) and every LOG_EVERY iterations after.
function weight_norm_checkpoint(rbm, hid_add)
    w_add = rbm.w[:, :, hid_add]
    q, n_vis, h_add = size(w_add)
    wr = reshape(w_add, q * n_vis, h_add)
    norms = vec(sqrt.(sum(abs2, wr; dims=1)))
    return (
        norms     = Array(norms),
        mean_norm = mean(norms),
        max_norm  = maximum(norms),
        min_norm  = minimum(norms),
    )
end

function log_weight_norm_checkpoint(path_wn, iter, chk)
    open(path_wn, "a") do f
        println(f, "iter=$iter mean_norm=$(chk.mean_norm) max_norm=$(chk.max_norm) min_norm=$(chk.min_norm) norms=$(chk.norms)")
    end
    println("iter=$iter  weight_norm_check  mean=$(chk.mean_norm)  max=$(chk.max_norm)  min=$(chk.min_norm)")
end

# =============================================================================
# HELD-OUT PSEUDOLIKELIHOOD CHECKPOINT (does it generalize, or just memorize?)
# =============================================================================
# `lpl` (already logged) is the pseudolikelihood of the current training
# minibatch `vd` — it can improve throughout training even if the model is
# just memorizing the training set. This computes the same statistic on the
# shuffled-out validation sequences (XA_val/XB_val, never trained on), so a gap
# that opens up between the two — training lpl improving while held-out lpl
# stalls or worsens — is the standard overfitting signal, and is what actually
# determines whether the correlations the model has learned (see V-V/A-B
# checks) can be trusted to generalize rather than being an artifact of this
# particular training set.
function log_heldout_checkpoint(path_lplval, iter, rbm_cpu, v_heldout_cpu)
    lpl_val = mean(log_pseudolikelihood(rbm_cpu, v_heldout_cpu))
    open(path_lplval, "a") do f; println(f, lpl_val); end
    println("iter=$iter  lpl_val=$lpl_val")
end

# =============================================================================
# FREE-BLOCK GRADIENT-NORM CHECKPOINT (is there still a learning signal here?)
# =============================================================================
# FREEZE CHECKPOINT's drift measures how hard Adam pushed on the *frozen*
# blocks before getting reset — useful for catching a broken freeze, but not
# for the free block, since the frozen-vs-reference comparison doesn't apply
# there. This instead recomputes the actual raw contrastive-divergence
# gradient — ∂d - ∂m from `∂free_energy`, the exact same quantity `pcd!`
# computes internally before regularization/optimizer state are applied — and
# reports its per-unit Frobenius norm restricted to the weight columns of
# `hid_add` (the free/added units for rbm_paired; all units for rbm_A/rbm_B,
# since nothing is frozen there). A norm that decays toward ~0 means the
# optimizer has run out of signal for that unit (converged, or stuck); one
# that stays flat-nonzero or grows means there's still real gradient driving
# it, independent of what the weight-norm or correlation checks show.
function free_gradient_checkpoint(rbm, vd, vm, hid_add)
    ∂d = ∂free_energy(rbm, vd)
    ∂m = ∂free_energy(rbm, vm)
    ∂w = (∂d.w .- ∂m.w)[:, :, hid_add]
    q, n_vis, h_add = size(∂w)
    gr = reshape(∂w, q * n_vis, h_add)
    norms = vec(sqrt.(sum(abs2, gr; dims=1)))
    return (
        norms     = Array(norms),
        mean_norm = mean(norms),
        max_norm  = maximum(norms),
        min_norm  = minimum(norms),
    )
end

function log_free_gradient_checkpoint(path_gn, iter, chk)
    open(path_gn, "a") do f
        println(f, "iter=$iter mean_norm=$(chk.mean_norm) max_norm=$(chk.max_norm) min_norm=$(chk.min_norm) norms=$(chk.norms)")
    end
    println("iter=$iter  free_grad_check  mean=$(chk.mean_norm)  max=$(chk.max_norm)  min=$(chk.min_norm)")
end

# =============================================================================
# TRAIN HELPER
# =============================================================================
function train_rbm!(rbm, X, iters, path_lpl; freeze_callback=nothing, path_freeze=nothing, rbm_A=nothing, rbm_B=nothing,
                     path_vh=nothing, hid_add=nothing, path_ab=nothing, vis_A=nothing, vis_B=nothing,
                     path_firing=nothing, path_wn=nothing, path_h=nothing, path_v=nothing,
                     path_lplval=nothing, v_heldout=nothing, path_gn=nothing,
                     ckpt_prefix=nothing, ckpt_every=nothing,
                     path_ab_val=nothing, v_ab_val=nothing, reg=REG, iter_offset::Int=0,
                     batchsize::Int=BATCH_SIZE)
    # pcd!'s minibatch iterator silently produces ZERO training iterations
    # whenever batchsize exceeds the number of available samples (see
    # infinite_minibatches: `iter.batchsize > n && return nothing`) -- with
    # PAIRED_TRAIN_FRAC shrinking the paired model's training set well below
    # BATCH_SIZE, this previously caused a *silent* no-training run whose
    # only symptom was the freeze-check failing (pcd!'s unconditional
    # zerosum!/rescale_weights! gauge-fixing at the top of the function
    # still nudges rbm_paired's frozen A/B blocks even with zero real
    # training steps, since gauge-fixing acts on the whole combined weight
    # matrix, not per-block). Capping here makes the common (BATCH_SIZE-vs-
    # full-data) case a no-op while still training something on tiny data.
    n_samples = size(X)[end]
    if batchsize > n_samples
        println("NOTE: batchsize ($batchsize) > training set size ($n_samples) -- capping to $n_samples")
        batchsize = n_samples
    end
    check_ckpt = !isnothing(ckpt_prefix) && !isnothing(ckpt_every)
    # Checkpoint .txt files are only cleared/reinitialized when this call will
    # actually run >=1 training iteration. A resume call that lands on an
    # already-fully-trained model (iters==0, a legitimate no-op -- e.g. a
    # rerun of train_potts_pfam.jl after the target iteration count was
    # already reached) previously truncated every checkpoint file
    # unconditionally here, permanently destroying the real checkpoint
    # history from whatever earlier call actually did the training, even
    # though it changed nothing about the saved model itself. Confirmed via
    # session notes: this silently emptied analyze_results_pfam.jl's figures
    # for a model that had trained completely normally.
    clear_ckpts = iters > 0
    clear_ckpts && (open(path_lpl, "w") do io end)
    check_freeze = !isnothing(freeze_callback) && !isnothing(path_freeze)
    check_freeze && clear_ckpts && (open(path_freeze, "w") do io end)
    check_vh = !isnothing(path_vh) && !isnothing(hid_add)
    check_vh && clear_ckpts && (open(path_vh, "w") do io end)
    check_h = !isnothing(path_h) && !isnothing(hid_add)
    check_h && clear_ckpts && (open(path_h, "w") do io end)
    check_ab = !isnothing(path_ab) && !isnothing(vis_A) && !isnothing(vis_B)
    check_ab && clear_ckpts && (open(path_ab, "w") do io end)
    check_firing = !isnothing(path_firing) && !isnothing(hid_add)
    check_firing && clear_ckpts && (open(path_firing, "w") do io end)
    check_v = !isnothing(path_v)
    check_v && clear_ckpts && (open(path_v, "w") do io end)
    check_wn = !isnothing(path_wn) && !isnothing(hid_add)
    if check_wn && clear_ckpts
        open(path_wn, "w") do io end
        log_weight_norm_checkpoint(path_wn, 0, weight_norm_checkpoint(rbm, hid_add))
    end
    check_lplval = !isnothing(path_lplval) && !isnothing(v_heldout)
    check_lplval && clear_ckpts && (open(path_lplval, "w") do io end)
    v_heldout_cpu = check_lplval ? cpu(v_heldout) : nothing
    check_gn = !isnothing(path_gn) && !isnothing(hid_add)
    check_gn && clear_ckpts && (open(path_gn, "w") do io end)
    check_ab_val = !isnothing(path_ab_val) && !isnothing(v_ab_val) && !isnothing(vis_A) && !isnothing(vis_B)
    c_data_val = nothing
    if check_ab_val
        clear_ckpts && (open(path_ab_val, "w") do io end)
        c_data_val = vv_connected_correlation(v_ab_val[:, vis_A, :], v_ab_val[:, vis_B, :])
    end
    @time pcd!(
        rbm, X;
        optim       = OptimiserChain(ClipNorm(CLIP_NORM), Adam(LR, (0f0, 999f-3), 1f-6)),
        iters       = iters,
        batchsize   = batchsize,
        steps       = CD_STEPS,
        l2l1_weights = reg,
        ϵv=1f-1, ϵh=0f0, damping=1f-1, rescale_hidden=false,
        callback = function(; iter, vd, vm, kwargs...)
            # true_iter is the CUMULATIVE iteration count across a resumed
            # run (iter_offset + this call's local 1:iters counter), used
            # everywhere for logging/checkpoint-naming/cadence so a resumed
            # run's logs read as a seamless continuation rather than
            # restarting from "iter=1". pcd! itself only ever sees the
            # REMAINING iteration count, and knows nothing about the offset.
            true_iter = iter + iter_offset
            if !isnothing(freeze_callback)
                # Detailed before/after drift is only worth the extra
                # computation (and log volume) at the usual LOG_EVERY cadence,
                # but the correctness guarantee itself — that the reset
                # actually landed exactly on the reference values — is cheap
                # (a handful of `maximum(abs, ...)` reductions) and is checked
                # on *every* iteration, hard-failing immediately rather than
                # just warning if it's ever violated.
                log_this_iter = check_freeze && iszero(true_iter % LOG_EVERY)
                drift_before = log_this_iter ? freeze_drift(rbm, rbm_A, rbm_B) : nothing
                freeze_callback()
                if check_freeze
                    drift_after = freeze_drift(rbm, rbm_A, rbm_B)
                    worst_after = maximum(drift_after)
                    if worst_after > FREEZE_TOL
                        error("Freeze violated at iter=$true_iter: max_drift_after_reset=$worst_after  drift_after=$drift_after")
                    end
                    log_this_iter && log_freeze_checkpoint(path_freeze, true_iter, drift_before, drift_after)
                end
            end
            if check_vh && iszero(true_iter % LOG_EVERY)
                chk = vh_learning_checkpoint(rbm, vd, vm, hid_add)
                log_vh_checkpoint(path_vh, true_iter, chk)
            end
            if check_h && iszero(true_iter % LOG_EVERY)
                chk_h = h_learning_checkpoint(rbm, vd, vm, hid_add)
                log_h_checkpoint(path_h, true_iter, chk_h)
            end
            if check_ab && iszero(true_iter % LOG_EVERY)
                chk_ab = ab_correlation_checkpoint(vd, vm, vis_A, vis_B)
                log_ab_checkpoint(path_ab, true_iter, chk_ab)
            end
            if check_ab_val && iszero(true_iter % LOG_EVERY)
                chk_ab_val = ab_val_correlation_checkpoint(vm, vis_A, vis_B, c_data_val)
                log_ab_val_checkpoint(path_ab_val, true_iter, chk_ab_val)
            end
            if check_firing && iszero(true_iter % LOG_EVERY)
                chk_fr = firing_rate_checkpoint(rbm, vd, vm, hid_add)
                log_firing_checkpoint(path_firing, true_iter, chk_fr)
            end
            if check_v && iszero(true_iter % LOG_EVERY)
                chk_v = v_learning_checkpoint(vd, vm)
                log_v_checkpoint(path_v, true_iter, chk_v)
            end
            if check_wn && iszero(true_iter % LOG_EVERY)
                log_weight_norm_checkpoint(path_wn, true_iter, weight_norm_checkpoint(rbm, hid_add))
            end
            if check_gn && iszero(true_iter % LOG_EVERY)
                log_free_gradient_checkpoint(path_gn, true_iter, free_gradient_checkpoint(rbm, vd, vm, hid_add))
            end
            if iszero(true_iter % LOG_EVERY)
                rbm_cpu = cpu(rbm)
                lpl = mean(log_pseudolikelihood(rbm_cpu, cpu(vd)))
                println("iter=$true_iter  lpl=$lpl")
                open(path_lpl, "a") do f; println(f, lpl); end
                if check_lplval
                    log_heldout_checkpoint(path_lplval, true_iter, rbm_cpu, v_heldout_cpu)
                end
            end
            if check_ckpt && true_iter > 0 && iszero(true_iter % ckpt_every)
                ckpt_path = "$(ckpt_prefix)_iter=$(true_iter).hdf5"
                save_rbm(ckpt_path, cpu(rbm); overwrite=true)
                println("iter=$true_iter  checkpoint saved → $ckpt_path")
            end
            # Long training runs (tens of thousands of iterations) were
            # observed to slowly approach GPU memory exhaustion and stall for
            # several minutes at a time before self-recovering -- same root
            # cause as the fix already applied to pair_results_pfam.jl's
            # gibbs_sample: many small per-iteration GPU allocations (here,
            # notably the l2l1_weights regularization term's sign.(rbm.w) and
            # mean(abs, rbm.w; dims) temporaries, only allocated when REG/REG_B
            # is nonzero) pile up in the CUDA memory pool faster than Julia's
            # GC reclaims them inside a tight, allocation-heavy loop. Periodic
            # GC.gc() + CUDA.reclaim() keeps steady-state memory bounded
            # regardless of iters.
            if iter > 0 && iszero(iter % 500)
                GC.gc()
                USE_GPU && CUDA.reclaim()
            end
            # stdout is block-buffered (not line-buffered) whenever it's
            # redirected to a file rather than a tty -- without this, a whole
            # run's progress can sit invisibly in the buffer and only appear
            # when the process exits, making a genuinely-progressing run look
            # hung to anyone tailing the log file live.
            iszero(true_iter % LOG_EVERY) && flush(stdout)
        end,
    )
    # Final, explicit check after the training loop has fully returned: the
    # per-iteration hard assert above already guarantees every reset landed
    # exactly on target throughout training, but this re-confirms the model
    # hasn't been mutated afterwards (e.g. by code added between this call
    # and the point where the caller saves/uses `rbm`), and leaves an
    # unambiguous record of the final state in the freeze-check log.
    if check_freeze
        drift_final = freeze_drift(rbm, rbm_A, rbm_B)
        worst_final = maximum(drift_final)
        status_final = worst_final <= FREEZE_TOL ? "OK" : "VIOLATION"
        open(path_freeze, "a") do f
            println(f, "FINAL status=$status_final drift_after_reset=$drift_final")
        end
        println("FINAL  freeze_check=$status_final  max_drift_after_reset=$worst_final")
        if worst_final > FREEZE_TOL
            error("Freeze violated at end of training: max_drift_after_reset=$worst_final  drift=$drift_final")
        end
    end
end

# =============================================================================
# TRAIN INDIVIDUAL RBMs
# =============================================================================
if isfile(path_rbm_A)
    println("\n--- RBM A: found existing model, skipping training ---")
    println("Loading RBM A ← $path_rbm_A")
    rbm_A = load_rbm(path_rbm_A)
else
    resume_path_A, resume_iter_A = find_resumable("rbm_A_", single_suffix_A, N_ITERS)
    if resume_iter_A > 0
        println("\n--- RBM A: resuming from iter=$resume_iter_A ← $resume_path_A ---")
        println("(warm start: model parameters carry over; PCD fantasy chain and Adam momentum restart fresh)")
        rbm_A = load_rbm(resume_path_A)
    else
        rbm_A = RBM(PottsGumbel((q, N_VIS_A)), nsReLU((N_HIDDEN_A,)), zeros(q, N_VIS_A, N_HIDDEN_A))
        initialize!(rbm_A, XA)
        rbm_A = standardize(rbm_A)
    end
    remaining_A = N_ITERS - resume_iter_A
    println("\n--- Training RBM A ($remaining_A more iterations to reach N_ITERS=$N_ITERS) ---")
    rbm_A=dev(rbm_A)
    train_rbm!(rbm_A, dev(XA), remaining_A, path_lpl_A; path_vh = path_vh_A, hid_add = 1:N_HIDDEN_A, path_wn = path_wn_A,
        path_h = path_h_A, path_v = path_v_A,
        path_lplval = path_lplval_A, v_heldout = dev(XA_val), path_gn = path_gn_A,
        ckpt_prefix = joinpath(CKPT_DIR, "rbm_A_$(single_suffix_A())"), ckpt_every = CKPT_EVERY_SINGLE,
        reg = REG_A, iter_offset = resume_iter_A)
    rbm_A=cpu(rbm_A)
    save_rbm(path_rbm_A, rbm_A; overwrite=true)
    println("Saved RBM A → $path_rbm_A")
    println("VH-correlation checkpoints (RBM A) → $path_vh_A")
    println("H-correlation checkpoints (RBM A) → $path_h_A")
    println("V-correlation checkpoints (RBM A) → $path_v_A")
    println("Weight-norm checkpoints (RBM A) → $path_wn_A")
    println("Held-out pseudolikelihood checkpoints (RBM A) → $path_lplval_A")
    println("Free-gradient-norm checkpoints (RBM A) → $path_gn_A")
end

if isfile(path_rbm_B)
    println("\n--- RBM B: found existing model, skipping training ---")
    println("Loading RBM B ← $path_rbm_B")
    rbm_B = load_rbm(path_rbm_B)
else
    resume_path_B, resume_iter_B = find_resumable("rbm_B_", single_suffix_B, N_ITERS_B)
    if resume_iter_B > 0
        println("\n--- RBM B: resuming from iter=$resume_iter_B ← $resume_path_B ---")
        println("(warm start: model parameters carry over; PCD fantasy chain and Adam momentum restart fresh)")
        rbm_B = load_rbm(resume_path_B)
    else
        rbm_B = RBM(PottsGumbel((q, N_VIS_B)), nsReLU((N_HIDDEN_B,)), zeros(q, N_VIS_B, N_HIDDEN_B))
        initialize!(rbm_B, XB)
        rbm_B = standardize(rbm_B)
    end
    remaining_B = N_ITERS_B - resume_iter_B
    println("\n--- Training RBM B ($remaining_B more iterations to reach N_ITERS_B=$N_ITERS_B) ---")
    rbm_B=dev(rbm_B)
    train_rbm!(rbm_B, dev(XB), remaining_B, path_lpl_B; path_vh = path_vh_B, hid_add = 1:N_HIDDEN_B, path_wn = path_wn_B,
        path_h = path_h_B, path_v = path_v_B,
        path_lplval = path_lplval_B, v_heldout = dev(XB_val), path_gn = path_gn_B,
        ckpt_prefix = joinpath(CKPT_DIR, "rbm_B_$(single_suffix_B())"), ckpt_every = CKPT_EVERY_SINGLE,
        reg = REG_B, iter_offset = resume_iter_B)
    rbm_B=cpu(rbm_B)
    save_rbm(path_rbm_B, rbm_B; overwrite=true)
    println("Saved RBM B → $path_rbm_B")
    println("VH-correlation checkpoints (RBM B) → $path_vh_B")
    println("H-correlation checkpoints (RBM B) → $path_h_B")
    println("V-correlation checkpoints (RBM B) → $path_v_B")
    println("Weight-norm checkpoints (RBM B) → $path_wn_B")
    println("Held-out pseudolikelihood checkpoints (RBM B) → $path_lplval_B")
    println("Free-gradient-norm checkpoints (RBM B) → $path_gn_B")
end

# =============================================================================
# TRAIN PAIRED RBM
# =============================================================================
n_hid_total = N_HIDDEN_A + N_HIDDEN_B + H_ADD
# PottsGumbel (not Potts) is required for GPU training: `Potts`'s sampling
# routine iterates element-by-element and hits CUDA's "scalar indexing is
# disallowed" error, whereas PottsGumbel uses the GPU-friendly Gumbel-softmax
# trick (same as rbm_A/rbm_B above). Both share the same `.par` layout, so the
# frozen-parameter projection is unaffected by this choice.
resume_path_paired, resume_iter_paired = find_resumable("rbm_paired_", paired_suffix, PAIRED_ITERS; marker_name="PAIRED_ITERS")
if resume_iter_paired > 0
    println("\n--- Paired RBM: resuming from iter=$resume_iter_paired ← $resume_path_paired ---")
    println("(warm start: model parameters carry over; PCD fantasy chain and Adam momentum restart fresh)")
    rbm_paired = load_rbm(resume_path_paired)
else
    rbm_paired = RBM(PottsGumbel((q, N_VIS_A + N_VIS_B)), nsReLU((n_hid_total,)), zeros(q, N_VIS_A + N_VIS_B, n_hid_total))
    initialize!(rbm_paired, X_AB)
    rbm_paired = standardize(rbm_paired)
end
project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)
remaining_paired = PAIRED_ITERS - resume_iter_paired

println("\n--- Training Paired RBM ($remaining_paired more iterations to reach PAIRED_ITERS=$PAIRED_ITERS) ---")
# rbm_A/rbm_B were moved back to cpu after their own training (for saving); the
# freeze reset runs every iteration against rbm_paired, so its reference RBMs
# must live on the same device (gpu) as rbm_paired, otherwise the projection's
# `.=` broadcast mixes CuArray and Array and blows up. Keep the cpu rbm_A/rbm_B
# untouched (already saved) and make separate gpu copies just for this reset.
rbm_A_gpu  = dev(rbm_A)
rbm_B_gpu  = dev(rbm_B)
rbm_paired = dev(rbm_paired)
hid_add    = added_hidden_range(rbm_A_gpu, rbm_B_gpu, H_ADD)
(; vis_A, vis_B) = frozen_block_ranges(rbm_A_gpu, rbm_B_gpu)
X_AB_val_gpu = dev(X_AB_val)
train_rbm!(rbm_paired, dev(X_AB), remaining_paired, path_lpl_paired;
    freeze_callback = () -> project_to_frozen_par!(rbm_paired, rbm_A_gpu, rbm_B_gpu, H_ADD),
    path_freeze = path_freeze_paired, rbm_A = rbm_A_gpu, rbm_B = rbm_B_gpu,
    path_vh = path_vh_paired, hid_add = hid_add, path_h = path_h_paired,
    path_ab = path_ab_paired, vis_A = vis_A, vis_B = vis_B,
    path_firing = path_firing_paired, path_wn = path_wn_paired, path_v = path_v_paired,
    path_lplval = path_lplval_paired, v_heldout = X_AB_val_gpu, path_gn = path_gn_paired,
    ckpt_prefix = joinpath(CKPT_DIR, "rbm_paired_$(paired_suffix())"), ckpt_every = CKPT_EVERY_PAIRED,
    path_ab_val = path_ab_val_paired, v_ab_val = X_AB_val_gpu, iter_offset = resume_iter_paired,
    reg = REG_PAIRED)
rbm_paired = cpu(rbm_paired)
save_rbm(path_rbm_paired, rbm_paired; overwrite=true)
println("Saved paired RBM → $path_rbm_paired")
println("Freeze checkpoints → $path_freeze_paired")
println("VH-correlation checkpoints (added hidden units) → $path_vh_paired")
println("H-correlation checkpoints (added hidden units) → $path_h_paired")
println("V-correlation checkpoints (whole visible layer) → $path_v_paired")
println("Firing-rate checkpoints (added hidden units) → $path_firing_paired")
println("A-B correlation checkpoints (cross-family structure) → $path_ab_paired")
println("A-B correlation checkpoints, held-out (cross-family generalization) → $path_ab_val_paired")
println("Weight-norm checkpoints (added hidden units) → $path_wn_paired")
println("Held-out pseudolikelihood checkpoints (paired) → $path_lplval_paired")
println("Free-gradient-norm checkpoints (added hidden units) → $path_gn_paired")
