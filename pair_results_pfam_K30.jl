using CUDA
using HDF5
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_v, sample_from_inputs, mean_h_from_v
using RestrictedBoltzmannMachines: free_energy, load_rbm, gpu, cpu, Falses
using Statistics: mean, std, cor
using LinearAlgebra: eigen, Symmetric
using Random
using LatentAlignedRBMs
using FASTX
using BioSequences
using Plots

# CPU fallback: see the matching comment in train_potts_pfam.jl. Same
# convention -- PFAM_USE_CPU=1 env var, not an ARGS position or filename tag.
const USE_GPU = lowercase(get(ENV, "PFAM_USE_CPU", "0")) ∉ ("1", "true", "yes")
dev(x) = USE_GPU ? gpu(x) : cpu(x)
println(USE_GPU ? "Device: GPU" : "Device: CPU (PFAM_USE_CPU=1)")
gr()
ENV["GKSwstype"] = "100"

# Parsed early (ahead of Random.seed!) -- see the matching comment in
# train_potts_pfam.jl. Deliberately absent from single_suffix_A/B() so
# multiple SEED variants of the paired model still correctly load the same
# frozen A/B checkpoints.
const SEED = length(ARGS) >= 14 ? parse(Int, ARGS[14]) : 42
Random.seed!(SEED)

# =============================================================================
# CONFIG
# =============================================================================
# Protein analogue of TwoFamilyIsing/pair_results.jl: the "real" validation
# companion to analyze_results_pfam.jl — that script only reads training-time
# logs (short PCD chains); this one loads the actual saved models and draws
# long, properly-equilibrated Gibbs chains from them.
#
# Two substantive differences from the Ising original, both required by using
# real protein data instead of a synthetic model — see the comments at
# `connected_corr_full`/`frob_strength_matrix` and the DATA section below:
#   1. No ground-truth coupling matrix K exists for real sequences, so the
#      A-B "definitive test" instead uses the held-out validation split
#      (XA_val/XB_val, never trained on) as the best available truth proxy.
#   2. Amino-acid identities aren't ordinal, so pair_results.jl's `decode_state`
#      trick (numbering categories 0..q-1 and taking a Pearson correlation) is
#      not meaningful here — site-pair coupling is instead measured by the
#      Frobenius norm of the connected one-hot correlation tensor, the same
#      quantity train_potts_pfam.jl's own vh/vv/ab checkpoints already use.
#
# Same ARGS convention as train_potts_pfam.jl / analyze_results_pfam.jl, plus
# nine trailing sampling-only overrides (train_potts_pfam.jl/analyze_results_pfam.jl
# don't sample, so they stop at ARGS[13]):
# N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN_A=30] [N_HIDDEN_B=30] [FASTA_PATH=./PF00072_PF00512_paired.fasta] [SPLIT_SITE=111] [N_ITERS_B=N_ITERS] [PAIRED_TRAIN_FRAC=1.0] [USE_REWEIGHTING=true] [REG_B=0] [REG_PAIRED=0] [REG_A=0] [SEED=42] [GIBBS_STRIDE=50] [N_GIBBS_STEPS_A=600] [N_GIBBS_STEPS_B=600] [N_GIBBS_STEPS_PAIRED=600] [CONVERGENCE_LOG_EVERY=10] [N_SAMPLES_A=5000] [N_SAMPLES_B=5000] [N_SAMPLES_PAIRED=5000]
const OUTPUT_DIR = "./results_pfam"
# Patched for the older K=30/H_ADD=20 model, whose checkpoint filenames
# predate the CLIP/SPLIT/RW tagging convention -- see single_suffix_A/B()
# and paired_suffix() below for the matching filename patch.
const CD_STEPS   = 30
const CLIP_NORM  = 1.0   # must match train_potts_pfam.jl's CLIP_NORM to reconstruct the same filenames

const N_ITERS      = parse(Int, ARGS[1])
const PAIRED_ITERS = parse(Int, ARGS[2])
const H_ADD        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
const N_HIDDEN_A   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const N_HIDDEN_B   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 30
const REG          = 0
const BATCH_SIZE   = 256
const LR           = 1f-4
const FASTA_PATH   = length(ARGS) >= 6 ? ARGS[6] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE   = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 111
const N_ITERS_B    = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : N_ITERS
const PAIRED_TRAIN_FRAC = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0
const USE_REWEIGHTING = length(ARGS) >= 10 ? parse(Bool, ARGS[10]) : true
# RBM B-specific l2l1_weights override -- must match train_potts_pfam.jl's
# constant of the same name to reconstruct the right checkpoint path.
const REG_B        = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : REG
# Paired-model-specific l2l1_weights override -- must match train_potts_pfam.jl's
# constant of the same name (see the comment there) to reconstruct the right
# checkpoint path.
const REG_PAIRED   = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : REG
# RBM A-specific l2l1_weights override -- must match train_potts_pfam.jl's
# constant of the same name to reconstruct the right checkpoint path.
const REG_A        = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : REG
const DATASET_TAG  = "$(replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => ""))_l$(SPLIT_SITE)"

const TRAIN_FRAC = 0.7
const VAL_FRAC   = 0.15

# Must match train_potts_pfam.jl's constants of the same name — see the
# comments there for the reasoning (two different identity thresholds for
# two different purposes: split leakage-prevention vs. DCA-style reweighting).
const DEDUP_IDENTITY_THRESHOLD    = 0.97
const REWEIGHT_IDENTITY_THRESHOLD = 0.8

# Independently settable per model (rather than one shared N_SAMPLES) so an
# experiment can e.g. reuse a cached RBM A sample set (see FANTASY_A_CACHE
# below) while sweeping B/Paired's sample count independently.
const N_SAMPLES_A       = length(ARGS) >= 20 ? parse(Int, ARGS[20]) : 5000
const N_SAMPLES_B       = length(ARGS) >= 21 ? parse(Int, ARGS[21]) : 5000
const N_SAMPLES_PAIRED  = length(ARGS) >= 22 ? parse(Int, ARGS[22]) : 5000
# Sampling-time Gibbs stride, decoupled from training's CD_STEPS: 100
# elementary sweeps between checkpoints, regardless of what K was used during
# training.
#
# Earlier versions of this script escalated N_GIBBS_STEPS_A up to 10,000 (1M
# sweeps) chasing what looked like an under-equilibration problem (RBM A's
# free energy still visibly decreasing at shorter budgets). A set of targeted
# diagnostics (see conversation/session notes) ruled out save/load corruption,
# a GPU-specific sampling bug, and a many-small-calls-vs-one-big-call artifact
# as explanations — what's actually happening is that the *model itself*
# (trained with a much shorter CD_STEPS) is only reliably calibrated within
# roughly that horizon; sampling far beyond it walks the chain away from
# where the model is trustworthy, and correlation with the data degrades the
# longer you run, with no plateau observed out to 31,000 sweeps in ad hoc
# testing. So: longer sampling was making things *worse*, not better.
#
# Rather than guess a budget, `gibbs_sample` below now also logs a
# correlation-vs-sweep-count trace (⟨h⟩/⟨v⟩/⟨hv⟩ against real data) alongside
# free energy at every checkpoint, so the actual peak/degradation point is
# measured directly per model (A, B, Paired can behave differently) instead
# of assumed. The final moments_validation figure still uses the last
# checkpoint's samples — use the new convergence_check figure/logs to judge
# whether N_GIBBS_STEPS below should be tightened further for a future run.
# GIBBS_STRIDE matches CD_STEPS=50 here (finer convergence-trace resolution);
# N_GIBBS_STEPS_* doubled from the 100-stride default (300) to 600 so the
# total default exploratory budget stays 30,000 sweeps either way.
# All five overridable via trailing ARGS (see convention comment above) so the
# H_ADD-sweep driver can run the default-budget -> fine-grained-probe ->
# tuned-final-resample workflow per sweep point without maintaining N copies
# of this file with hand-edited consts.
const GIBBS_STRIDE          = length(ARGS) >= 15 ? parse(Int, ARGS[15]) : 50
const N_GIBBS_STEPS_A       = length(ARGS) >= 16 ? parse(Int, ARGS[16]) : 600   # 30,000 sweeps by default -- upper end of the range explored in diagnostics
const N_GIBBS_STEPS_B       = length(ARGS) >= 17 ? parse(Int, ARGS[17]) : 600
const N_GIBBS_STEPS_PAIRED  = length(ARGS) >= 18 ? parse(Int, ARGS[18]) : 600
const CONVERGENCE_LOG_EVERY = length(ARGS) >= 19 ? parse(Int, ARGS[19]) : 10   # log correlation trace every N checkpoints

# Use the *entire* available train/val split for the "data" moment/coupling
# comparisons (matching what pair_results.jl does for the Ising dataset),
# rather than an arbitrary subsample -- `min(N_DATA_SAMPLES, length(...))`
# below clips to whatever's actually available, so a large constant here
# just means "use everything".
const N_DATA_SAMPLES = typemax(Int)

# Fraction of A×B site pairs treated as "putative strongest couplings" when
# computing top-K overlap against the held-out reference (see TOP_FRAC below).
const TOP_FRAC = 0.05

const ARCHS = [("", "Potts+nsReLU")]

# Canonicalizes an integer-valued Float64 (e.g. from `parse(Float64, ARGS[n])`)
# to the same string an untyped Int 0 literal would produce ("0", not "0.0"),
# so a REG_A/REG_B/REG_PAIRED explicitly passed as "0" on the command line
# resolves to the exact same filename as the default (untyped Int) fallback --
# every existing pre-regularization model file was saved with the bare "REG=0"
# spelling, so any mismatch here means silently failing to find it (see
# "Missing model file(s)" session notes).
regstr(r) = isinteger(r) ? string(Int(r)) : string(r)

# Only gains a distinguishing "_NITERSB=..." / "_PAIREDFRAC=..." tag when
# those parameters actually differ from their defaults, so the common case's
# directory names are unchanged from before these features existed.
n_iters_b_tag() = N_ITERS_B == N_ITERS ? "" : "_NITERSB=$(N_ITERS_B)"
paired_frac_tag() = PAIRED_TRAIN_FRAC == 1.0 ? "" : "_PAIREDFRAC=$(PAIRED_TRAIN_FRAC)"
# reg_b_tag() included unconditionally would break the common REG_B==REG case's
# directory names; only add a distinguishing suffix when RBM B is actually
# regularized differently from the default, matching the n_iters_b_tag()/
# paired_frac_tag() convention.
reg_b_tag() = REG_B == REG ? "" : "_REGB=$(regstr(REG_B))"
# Same convention, for the paired model's own regularization -- see the
# matching comment on paired_suffix() below.
reg_paired_tag() = REG_PAIRED == REG ? "" : "_REGPAIRED=$(regstr(REG_PAIRED))"
reg_a_tag() = REG_A == REG ? "" : "_REGA=$(regstr(REG_A))"
# Deliberately absent from single_suffix_A/B() -- see the matching comment
# in train_potts_pfam.jl.
seed_tag() = SEED == 42 ? "" : "_SEED=$(SEED)"
const PARAMTAG = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_H_ADD=$(H_ADD)_N_ITERS=$(N_ITERS)_PAIRED_ITERS=$(PAIRED_ITERS)_REG=$(REG)_BS=$(BATCH_SIZE)_LR=$(LR)_DATA=$(DATASET_TAG)$(n_iters_b_tag())$(paired_frac_tag())$(reg_a_tag())$(reg_b_tag())$(reg_paired_tag())"
const REPORT_DIR = joinpath(OUTPUT_DIR, "pair_results_$(PARAMTAG)")
isdir(REPORT_DIR) || mkpath(REPORT_DIR)

figpath(arch, name) = joinpath(REPORT_DIR, "$(arch)$(name)_$(PARAMTAG).png")

# H_ADD is deliberately absent from single_suffix_A/B() — see train_potts_pfam.jl's
# comment above its own definition: RBM A/B don't depend on H_ADD, so their
# saved files are shared across an H_ADD sweep, and only paired_suffix()
# (built independently below) varies by it. single_suffix_A/B() are separate
# (not one shared single_suffix()) so N_ITERS_B can differ from N_ITERS
# without affecting RBM A's filename — see the matching comment in
# train_potts_pfam.jl.
single_suffix_A() = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG_A))_BS=$(BATCH_SIZE)_LR=$(LR)_DATA=$(DATASET_TAG)"
single_suffix_B() = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS_B)_REG=$(regstr(REG_B))_BS=$(BATCH_SIZE)_LR=$(LR)_DATA=$(DATASET_TAG)"
# Must include reg_b_tag()/reg_paired_tag() to match train_potts_pfam.jl's
# paired_suffix() -- see the detailed comment there: without these, every
# REG_B/REG_PAIRED combination silently shared and overwrote the same
# rbm_paired_*.hdf5 file, so this script would have loaded whichever
# combination was trained *last*, not the one matching this run's own REG_B/
# REG_PAIRED.
paired_suffix()  = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(REG)_BS=$(BATCH_SIZE)_LR=$(LR)_DATA=$(DATASET_TAG)$(n_iters_b_tag())$(paired_frac_tag())$(reg_a_tag())$(reg_b_tag())$(reg_paired_tag())$(seed_tag())_PAIRED_ITERS=$(PAIRED_ITERS)"
hdf5path(arch, name, suffix) = joinpath(OUTPUT_DIR, "$(arch)$(name)_$(suffix).hdf5")

# =============================================================================
# DATA
# =============================================================================
# Reconstructs the exact train/val split train_potts_pfam.jl used (same seed,
# same order of operations: seed set once at top, no other RNG draws before
# `randperm` below), so the "data" and "held-out" arrays here line up with
# what the loaded models were actually fit on / never saw.
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

# Ported verbatim from train_potts_pfam.jl — must stay identical so the two
# scripts draw the exact same train_idx/val_idx/paired_train_idx from the
# same seed. See the matching comments there for the reasoning.
function species_code(id::AbstractString)
    id_part = split(id, "|")[end]
    id_part = split(id_part, "/")[1]
    parts = split(id_part, "_")
    return length(parts) >= 2 ? parts[end] : id_part
end

function is_ambiguous_species(code::AbstractString)
    return occursin(r"^[0-9]", code) || occursin("UNC", code) || occursin("UNK", code)
end

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

# Ported verbatim from train_potts_pfam.jl — see the comments there.
function pairwise_identity_matrix(X::AbstractArray)
    q, L, N = size(X)
    Xflat = dev(reshape(Float32.(X), q * L, N))
    return Array((Xflat' * Xflat) ./ Float32(L))
end

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

function sequence_weights(ident::AbstractMatrix, threshold::Real)
    n = size(ident, 1)
    return [1.0 / count(>=(threshold), @view ident[:, i]) for i in 1:n]
end

function weighted_resample(idx::Vector{Int}, weights::Vector{Float64}, n_draws::Int)
    cw = cumsum(weights)
    total = cw[end]
    return [idx[searchsortedfirst(cw, rand() * total)] for _ in 1:n_draws]
end

# Ported verbatim from train_potts_pfam.jl — see the comment there.
function maybe_reweight(idx::Vector{Int}, ident::Union{Nothing,AbstractMatrix}, label::AbstractString)
    isnothing(ident) && return idx
    w = sequence_weights(ident, REWEIGHT_IDENTITY_THRESHOLD)
    println("$label reweighting: Meff=$(round(sum(w); digits=1)) / N=$(length(w))")
    return weighted_resample(idx, w, length(idx))
end

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

q         = size(seqs_onehot, 1)
n_sites   = size(seqs_onehot, 2)
n_samples = size(seqs_onehot, 3)
N_VIS_A   = SPLIT_SITE
N_VIS_B   = n_sites - SPLIT_SITE

# Split/identity cache: full_ident + the resulting train/val split are
# entirely determined by FASTA_PATH + TRAIN_FRAC/VAL_FRAC/
# DEDUP_IDENTITY_THRESHOLD (Random.seed!(42) above pins the one randperm()
# grouped_train_val_split uses) -- identical across every REG_B/H_ADD/N_ITERS/
# device/sampling-budget combination, and shared with train_potts_pfam.jl
# (same cache file, same key, ported verbatim). full_ident itself is a dense
# N×N matmul that's fast on GPU but slow on CPU with nothing printed while it
# runs, so cache it once instead of repaying that cost every run.
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
    train_idx, val_idx = grouped_train_val_split(seqs_raw, seqs_species, full_ident, n_samples, TRAIN_FRAC, VAL_FRAC)
    h5open(split_cache_path, "w") do io
        write(io, "full_ident", full_ident)
        write(io, "train_idx", train_idx)
        write(io, "val_idx", val_idx)
    end
    println("Saved train/val split + identity matrix → $split_cache_path")
end

# PCA basis cache: like full_ident above, this is a property of the WHOLE
# dataset only (FASTA_PATH + q/n_sites), not of any particular REG_A/REG_B/
# REG_PAIRED/PAIRED_TRAIN_FRAC/SEED combination, so it's computed once and
# reused by every run against this FASTA -- used by the pca_scatter figure
# below to show where a given run's training subset and generated samples
# sit relative to the full population. Eigendecomposing the (q*n_sites)x
# (q*n_sites) covariance matrix (~6000x6000 here) rather than doing SVD on
# the full (q*n_sites)x(n_samples) matrix directly, since q*n_sites << n_samples
# and this is much cheaper that way; entirely deterministic (no RNG use), so
# safe to compute at any point relative to Random.seed!(SEED) above.
const PCA_NCOMP = 2
pca_cache_path = joinpath(OUTPUT_DIR, "pca_cache_$(fasta_tag).hdf5")
if isfile(pca_cache_path)
    println("--- PCA basis: found cached basis, skipping eigendecomposition ---")
    println("Loading ← $pca_cache_path")
    pca_mean = h5read(pca_cache_path, "mean")
    pca_components = h5read(pca_cache_path, "components")
    @assert length(pca_mean) == q * n_sites "cached PCA basis dimension mismatch -- delete $pca_cache_path and rerun"
else
    println("--- Computing PCA basis over full dataset ($n_samples sequences) ---")
    Xflat = reshape(Float32.(seqs_onehot), q * n_sites, n_samples)
    pca_mean = vec(mean(Xflat; dims=2))
    Xc = Xflat .- pca_mean
    C = Symmetric((Xc * Xc') ./ Float32(n_samples - 1))
    ev = eigen(C)
    top = sortperm(ev.values; rev=true)[1:PCA_NCOMP]
    pca_components = ev.vectors[:, top]
    h5open(pca_cache_path, "w") do io
        write(io, "mean", pca_mean)
        write(io, "components", pca_components)
    end
    println("Saved PCA basis → $pca_cache_path")
end
pca_project(X) = pca_components' * (reshape(Float32.(X), q * n_sites, size(X)[end]) .- pca_mean)
const PCA_BG_SUBSAMPLE = 20_000
# Uses its own independent RNG (not the global stream) so this subsampling
# can't shift the global RNG position -- paired_train_idx below MUST land on
# the exact same draw pair_results_pfam.jl and train_potts_pfam.jl would
# both make from a freshly-seeded stream (see the comment there), and this
# figure is purely cosmetic background-point selection, not something that
# needs to reproduce across scripts.
pca_bg_idx = n_samples > PCA_BG_SUBSAMPLE ? randperm(MersenneTwister(0), n_samples)[1:PCA_BG_SUBSAMPLE] : (1:n_samples)
pca_full_proj = pca_project(seqs_onehot[:, :, pca_bg_idx])

function pca_scatter_panel(proj_train, proj_gen, title_str)
    plt = Plots.scatter(pca_full_proj[1, :], pca_full_proj[2, :];
        color=:gainsboro, alpha=0.3, markersize=2, markerstrokewidth=0, label="full dataset",
        xlabel="PC1", ylabel="PC2", title=title_str, titlefontsize=10,
        framestyle=:box, grid=false, legend=:outerright)
    Plots.scatter!(plt, proj_train[1, :], proj_train[2, :];
        color=:dodgerblue, alpha=0.5, markersize=3, markerstrokewidth=0, label="training data")
    Plots.scatter!(plt, proj_gen[1, :], proj_gen[2, :];
        color=:firebrick, alpha=0.5, markersize=3, markerstrokewidth=0, label="generated samples")
    return plt
end

# Same subsampling as train_potts_pfam.jl, at the same RNG position (right
# after the split, before anything reweighting-related draws random numbers)
# so this reproduces exactly the subset the paired model actually trained on
# — see the matching comment in train_potts_pfam.jl for why only train_idx/
# val_idx/paired_train_idx need to reproduce byte-identically, not the
# reweighted resampling below.
n_paired_train = floor(Int, PAIRED_TRAIN_FRAC * length(train_idx))
paired_train_idx = PAIRED_TRAIN_FRAC == 1.0 ? train_idx :
    train_idx[randperm(length(train_idx))[1:n_paired_train]]

n_data_train        = min(N_DATA_SAMPLES, length(train_idx))
n_data_val          = min(N_DATA_SAMPLES, length(val_idx))
n_data_paired_train = min(N_DATA_SAMPLES, length(paired_train_idx))
data_train_idx        = train_idx[1:n_data_train]
data_val_idx           = val_idx[1:n_data_val]
data_paired_train_idx = paired_train_idx[1:n_data_paired_train]

# XA/XB: reference for RBM A/B's own moments (they always train on the full
# training split, regardless of PAIRED_TRAIN_FRAC) — reweighted the same way
# train_potts_pfam.jl reweights what it actually trains on (toggled by
# USE_REWEIGHTING, same as there), so this "train data" reference reflects
# the distribution the model was actually fit to.
identA = USE_REWEIGHTING ? pairwise_identity_matrix(seqs_onehot[:, 1:SPLIT_SITE, data_train_idx]) : nothing
resampled_train_idx_A = maybe_reweight(data_train_idx, identA, "RBM A")
XA = seqs_onehot[:, 1:SPLIT_SITE, resampled_train_idx_A]

identB = USE_REWEIGHTING ? pairwise_identity_matrix(seqs_onehot[:, SPLIT_SITE+1:end, data_train_idx]) : nothing
resampled_train_idx_B = maybe_reweight(data_train_idx, identB, "RBM B")
XB = seqs_onehot[:, SPLIT_SITE+1:end, resampled_train_idx_B]

# X_AB: reference for the PAIRED model specifically — built from
# paired_train_idx (its actual, possibly-smaller training subset), not the
# full train_idx, so the "train data" baseline reported below reflects what
# that model really saw rather than the full data it didn't, and reweighted
# via full_ident sliced to this subset (reused, not recomputed).
ident_paired = USE_REWEIGHTING ? full_ident[data_paired_train_idx, data_paired_train_idx] : nothing
resampled_paired_idx = maybe_reweight(data_paired_train_idx, ident_paired, "Paired model")
XA_paired = seqs_onehot[:, 1:SPLIT_SITE, resampled_paired_idx]
XB_paired = seqs_onehot[:, SPLIT_SITE+1:end, resampled_paired_idx]
X_AB = cat(XA_paired, XB_paired; dims=2)

XA_val = seqs_onehot[:, 1:SPLIT_SITE, data_val_idx]
XB_val = seqs_onehot[:, SPLIT_SITE+1:end, data_val_idx]
X_AB_val = cat(XA_val, XB_val; dims=2)

println("XA: $(size(XA))  XB: $(size(XB))  (subsampled from $(length(train_idx)) train rows)")
println("X_AB: $(size(X_AB))  (paired model's training subset: $(length(paired_train_idx)) of $(length(train_idx)) train rows)")
println("XA_val: $(size(XA_val))  XB_val: $(size(XB_val))  X_AB_val: $(size(X_AB_val))  (subsampled from $(length(val_idx)) held-out rows)")

# =============================================================================
# ARCHITECTURE-GENERIC HELPERS
# =============================================================================
# Potts arrays are (q, n_vis, batch) one-hot. Only one architecture exists on
# the protein side so far, but keep the ndims-dispatch shape from
# pair_results.jl so a future flat (n_vis, batch) architecture just slots in.
flatten(x::AbstractArray{<:Any,2}) = x
flatten(x::AbstractArray{<:Any,3}) = reshape(x, size(x, 1) * size(x, 2), size(x, 3))

vis_slice(x::AbstractArray{<:Any,2}, r) = x[r, :]
vis_slice(x::AbstractArray{<:Any,3}, r) = x[:, r, :]

visible_means(x) = vec(mean(flatten(x); dims=2))
function hv_moments(h, x)
    xf = flatten(x)
    return h * xf' / size(xf, 2)
end

# =============================================================================
# SITE-PAIR COUPLING STRENGTH (Potts-appropriate replacement for pair_results.jl's decode_state+cor)
# =============================================================================
# C[a,i,b,j] = <v_i=a v_j=b> - <v_i=a><v_j=b> — the connected correlation
# between site i of block 1 (color a) and site j of block 2 (color b), same
# quantity as train_potts_pfam.jl's vv/ab-check checkpoints. Reducing this
# q×q-per-pair tensor to one non-negative "how strongly do these two sites
# covary" scalar per (i,j) via its Frobenius norm is the standard way to get a
# site-level coupling-strength matrix out of a Potts model (cf. DCA
# Frobenius-norm/APC scores) — unlike decoding each one-hot site down to a
# single numeric category index and taking an ordinary Pearson correlation,
# which implicitly assumes the categories are ordered (true for Ising ±1
# spins, false for amino acid identities).
function connected_corr_full(v_i, v_j)
    q, n_i, batch = size(v_i)
    n_j = size(v_j, 2)
    vi = reshape(Float64.(Array(v_i)), q * n_i, batch)
    vj = reshape(Float64.(Array(v_j)), q * n_j, batch)
    mean_i = vec(mean(vi; dims=2))
    mean_j = vec(mean(vj; dims=2))
    cross  = (vi * vj') ./ batch
    connected = cross .- mean_i * mean_j'
    return reshape(connected, q, n_i, q, n_j)
end

function frob_strength_matrix(v_i, v_j)
    C = connected_corr_full(v_i, v_j)
    q, n_i, _, n_j = size(C)
    S = Matrix{Float64}(undef, n_i, n_j)
    for j in 1:n_j, i in 1:n_i
        S[i, j] = sqrt(sum(abs2, @view C[:, i, :, j]))
    end
    return S
end

function zero_diag!(M)
    for i in 1:minimum(size(M))
        M[i, i] = 0
    end
    return M
end

mean_offdiag(M) = mean(M[i, j] for i in 1:size(M, 1), j in 1:size(M, 2) if i != j)

# Off-diagonal (site i != site j) entries of the connected correlation tensor,
# flattened across every colour pair — the exact same population
# train_potts_pfam.jl's vv_self_correlation_checkpoint correlates for the
# training-time vv_check curve (same-site self-correlation pairs excluded for
# the same reason: they're basically guaranteed to align via the marginals
# and would inflate the correlation with a trivial signal). This is the
# post-training analogue at the same granularity — unlike
# individual_correlations'/family_correlation_preservation's heatmaps, which
# aggregate each site pair down to a single Frobenius score for
# visualizability, this keeps every (site pair, colour pair) as its own point.
function vv_offdiag_vec(C)
    q, n_vis, _, _ = size(C)
    vals = Vector{Float64}(undef, q * q * n_vis * (n_vis - 1))
    k = 0
    for j in 1:n_vis, i in 1:n_vis
        i == j && continue
        for b in 1:q, a in 1:q
            k += 1
            vals[k] = C[a, i, b, j]
        end
    end
    return vals
end

# The full off-diagonal population above can run into the tens of millions of
# points for the larger machines (q^2 x n_vis x (n_vis-1), e.g. ~13.7M for the
# paired model's 177-site visible layer) — too many to usefully render as a
# scatter. Correlation rho is computed on the FULL population (so it matches
# what the training-time vv_check curve would report), but only a random
# subsample of points is actually drawn.
const VV_SCATTER_SUBSAMPLE = 20_000

function scatter_panel_vv(v_data, v_model, title_str; color=:dodgerblue)
    x = vv_offdiag_vec(connected_corr_full(v_data, v_data))
    y = vv_offdiag_vec(connected_corr_full(v_model, v_model))
    ρ = round(cor(x, y), digits=4)
    n = length(x)
    idx = n > VV_SCATTER_SUBSAMPLE ? randperm(n)[1:VV_SCATTER_SUBSAMPLE] : 1:n
    plt = Plots.scatter(x[idx], y[idx];
        xlabel="data", ylabel="model", title="$title_str (ρ=$ρ)", titlefontsize=10,
        alpha=0.3, markersize=2, markerstrokewidth=0, label="",
        color=color, framestyle=:box, grid=false)
    lo, hi = extrema(vcat(x[idx], y[idx]))
    Plots.plot!(plt, [lo, hi], [lo, hi]; color=:red, linestyle=:dash, label="")
    return plt, ρ
end

# Long, properly-equilibrated Gibbs chain (architecture-agnostic): initialize
# from the layer's own marginal (same convention pcd! itself uses for
# persistent chains), then track free energy over the run so burn-in can be
# checked visually rather than assumed.
function gibbs_sample(rbm, n_samples, n_steps, stride;
                       ref_h=nothing, ref_x=nothing, hid_range=nothing,
                       log_every=1, path_conv=nothing)
    x = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., n_samples))
    F = zeros(n_samples, n_steps)
    F[:, 1] .= Array(free_energy(rbm, x))

    # Correlation-vs-sweep-count trace: at every `log_every`-th checkpoint,
    # compute ⟨h⟩/⟨v⟩/⟨hv⟩ correlation of the *current* chain state against
    # real data (ref_h/ref_x, precomputed once by the caller), the same
    # quantities the final moments_validation scatter panels report on the
    # very last checkpoint. This exists because escalating the sampling
    # budget (chasing what looked like under-equilibration) turned out to
    # make correlation worse, not better — see the CONFIG section's note —
    # so rather than guess a budget, this measures where it actually peaks.
    check_conv = !isnothing(ref_h) && !isnothing(ref_x)
    conv_iters, conv_rh, conv_rv, conv_rhv = Int[], Float64[], Float64[], Float64[]
    hr = isnothing(hid_range) ? Colon() : hid_range
    ref_h_r = check_conv ? ref_h[hr, :] : nothing
    check_conv && !isnothing(path_conv) && (open(path_conv, "w") do io end)

    function log_conv!(cum_sweeps, x_now)
        hm    = Array(Float64.(sample_h_from_v(rbm, x_now)))[hr, :]
        x_now_c = Array(x_now)
        r_h  = cor(vec(mean(ref_h_r; dims=2)), vec(mean(hm; dims=2)))
        r_v  = cor(visible_means(ref_x), visible_means(x_now_c))
        r_hv = cor(vec(hv_moments(ref_h_r, ref_x)), vec(hv_moments(hm, x_now_c)))
        push!(conv_iters, cum_sweeps); push!(conv_rh, r_h); push!(conv_rv, r_v); push!(conv_rhv, r_hv)
        if !isnothing(path_conv)
            open(path_conv, "a") do f
                println(f, "sweeps=$cum_sweeps r_h=$r_h r_v=$r_v r_hv=$r_hv")
            end
        end
        println("    sweeps=$cum_sweeps  r_h=$(round(r_h,digits=3))  r_v=$(round(r_v,digits=3))  r_hv=$(round(r_hv,digits=3))")
        # stdout is block-buffered (not line-buffered) whenever it's redirected
        # to a file rather than a tty -- without this, a whole run's progress
        # can sit invisibly in the buffer and only appear when the process
        # exits, making a genuinely-progressing run look hung to anyone
        # tailing the log file live.
        flush(stdout)
    end

    check_conv && log_conv!(0, x)
    for t in 2:n_steps
        x = sample_v_from_v(rbm, x; steps=stride)
        F[:, t] .= Array(free_energy(rbm, x))
        if check_conv && (t % log_every == 0 || t == n_steps)
            log_conv!((t - 1) * stride, x)
        end
        # Long runs (large n_steps) were observed to slowly fill the CUDA
        # memory pool and stall/thrash near exhaustion: each iteration
        # reassigns `x` to a freshly-allocated GPU array, but Julia's GC is
        # tuned for CPU memory pressure and doesn't reliably keep up inside a
        # tight, allocation-heavy GPU loop, so old buffers pile up uncollected
        # faster than the pool reclaims them. Periodic GC.gc() + CUDA.reclaim()
        # forces collection of the now-orphaned previous `x`/intermediate
        # buffers every 500 iterations, keeping steady-state memory bounded
        # regardless of n_steps.
        if t % 500 == 0
            GC.gc()
            USE_GPU && CUDA.reclaim()
        end
    end
    return x, F, (; iters=conv_iters, r_h=conv_rh, r_v=conv_rv, r_hv=conv_rhv)
end

# =============================================================================
# PLOT HELPERS (Plots.jl — this environment has Plots, not CairoMakie; see
# train_small_pfam.jl, which uses the same backend)
# =============================================================================
const CELL = 380

function scatter_panel(xdata, ydata, title_str, xl, yl; color=:dodgerblue)
    ρ = round(cor(vec(xdata), vec(ydata)), digits=4)
    plt = Plots.scatter(vec(xdata), vec(ydata);
        xlabel=xl, ylabel=yl, title="$title_str (ρ=$ρ)", titlefontsize=10,
        alpha=0.5, markersize=3, markerstrokewidth=0, label="",
        color=color, framestyle=:box, grid=false)
    lo, hi = extrema(vcat(vec(xdata), vec(ydata)))
    Plots.plot!(plt, [lo, hi], [lo, hi]; color=:red, linestyle=:dash, label="")
    return plt, ρ
end

function heat_panel(M, title_str)
    return Plots.heatmap(M'; yflip=true, title=title_str, color=:viridis, framestyle=:box)
end

# Categorizes every site-pair by top-K membership: 0=background (neither),
# 1=true positive (in both top_truth and top_model), 2=false negative (real
# strong pair the model missed), 3=false positive (model claims strong,
# truth doesn't). top_truth/top_model are linear indices into a
# column-major-flattened (N_VIS_A, N_VIS_B) matrix, matching what
# `sortperm(vec(C_AB))` already produces above -- shared by both the spatial
# heatmap and the scatter panel below so the same top-K sets drive both
# views of the same question ("which specific pairs did the model get
# right?"), not two independently-computed answers.
function hit_miss_categories(top_truth, top_model, n::Int)
    truth_set, model_set = Set(top_truth), Set(top_model)
    cats = zeros(Int, n)
    for i in 1:n
        int, inm = i in truth_set, i in model_set
        cats[i] = int && inm ? 1 : int ? 2 : inm ? 3 : 0
    end
    return cats
end

const HITMISS_COLORS = [:gainsboro, :seagreen, :firebrick, :dodgerblue]  # bg, TP, FN, FP
const HITMISS_LABELS = ["background", "hit", "missed", "false-alarm"]

# Spatial view: where on the (site_A, site_B) grid do hits/misses/false
# alarms fall? Same grid/orientation as heat_panel above so it lines up
# visually with cross_correlation_vs_heldout's raw-strength panels.
#
# Heatmaps don't get a discrete legend for free -- the colorbar is
# disabled (a continuous 0-3 gradient bar is a confusing way to label 4
# categories) and replaced with the same NaN-point idiom used below: an
# invisible scatter point per category, present only so Plots.jl adds a
# proper labeled legend entry for it.
function hit_miss_heat_panel(cats_flat, shape, title_str)
    M = reshape(cats_flat, shape)
    cg = cgrad(HITMISS_COLORS, 4; categorical=true)
    plt = Plots.heatmap(M'; yflip=true, title=title_str, titlefontsize=10, color=cg,
        clims=(-0.5, 3.5), colorbar=false, framestyle=:box, legend=:outerright)
    for (color, lbl) in zip(HITMISS_COLORS, HITMISS_LABELS)
        Plots.scatter!(plt, [NaN], [NaN]; color=color, markersize=6, markerstrokewidth=0, label=lbl)
    end
    return plt
end

# Magnitude view: a hit/miss heatmap alone can't distinguish a false
# negative that narrowly missed the threshold from one the model ranked
# near zero -- this scatter (held-out strength vs. model strength, same
# top-K categories, threshold lines at each axis's Kn-th-largest value)
# carries that "how close" information the spatial panel drops. All
# TP/FN/FP points are kept; only the (typically ~95% of all pairs)
# background points are subsampled, matching the existing
# scatter_panel_vv subsampling convention -- otherwise this is just a wall
# of uninformative gray dots. Plotted as 4 separate labeled series (rather
# than one series with a per-point color vector) specifically so each
# category gets its own legend entry.
function hit_miss_scatter_panel(C_truth, C_model, top_truth, top_model, Kn, title_str; bg_subsample=3000)
    x, y = vec(C_truth), vec(C_model)
    cats = hit_miss_categories(top_truth, top_model, length(x))
    bg_idx = findall(==(0), cats)
    bg_kept = length(bg_idx) > bg_subsample ? bg_idx[randperm(length(bg_idx))[1:bg_subsample]] : bg_idx
    thresh_truth = sort(x; rev=true)[Kn]
    thresh_model = sort(y; rev=true)[Kn]
    plt = Plots.scatter(x[bg_kept], y[bg_kept];
        xlabel="held-out strength", ylabel="model strength", title=title_str, titlefontsize=10,
        color=HITMISS_COLORS[1], alpha=0.3, markersize=3, markerstrokewidth=0, label=HITMISS_LABELS[1],
        framestyle=:box, grid=false, legend=:outerright)
    for cat_val in 1:3
        idx = findall(==(cat_val), cats)
        Plots.scatter!(plt, x[idx], y[idx]; color=HITMISS_COLORS[cat_val+1], alpha=0.6,
            markersize=3, markerstrokewidth=0, label=HITMISS_LABELS[cat_val+1])
    end
    Plots.vline!(plt, [thresh_truth]; color=:black, linestyle=:dash, label="")
    Plots.hline!(plt, [thresh_model]; color=:black, linestyle=:dash, label="")
    return plt
end

function free_energy_panel(F, title_str)
    μ = vec(mean(F; dims=1))
    σ = vec(std(F; dims=1))
    return Plots.plot(1:length(μ), μ; ribbon=σ / 2, color=:dodgerblue, fillalpha=0.3,
        xlabel="sampling step", ylabel="free energy", title=title_str, label="", framestyle=:box, grid=false)
end

# =============================================================================
# PER-ARCHITECTURE ANALYSIS
# =============================================================================
summaries = Dict{String,Any}()

for (arch, label) in ARCHS
    println("\n" * "="^60)
    println(label)
    println("="^60)

    pA, pB, pP = hdf5path(arch, "rbm_A", single_suffix_A()), hdf5path(arch, "rbm_B", single_suffix_B()), hdf5path(arch, "rbm_paired", paired_suffix())
    if !(isfile(pA) && isfile(pB) && isfile(pP))
        println("  Missing model file(s) for this config — skipping $label.")
        println("    expected: $pA")
        println("              $pB")
        println("              $pP")
        continue
    end

    rbm_A      = dev(load_rbm(pA))
    rbm_B      = dev(load_rbm(pB))
    rbm_paired = dev(load_rbm(pP))

    XA_g, XB_g, X_AB_g = dev(XA), dev(XB), dev(X_AB)
    XA_c, XB_c, X_AB_c = Array(XA_g), Array(XB_g), Array(X_AB_g)

    n_flat_A    = q * N_VIS_A
    n_flat_B    = q * N_VIS_B
    vis_A_range = 1:n_flat_A
    vis_B_range = (n_flat_A + 1):(n_flat_A + n_flat_B)

    n_hid_A       = size(rbm_A.w)[end]
    n_hid_B       = size(rbm_B.w)[end]
    hid_A_range   = 1:n_hid_A
    hid_B_range   = (n_hid_A + 1):(n_hid_A + n_hid_B)
    hid_add_range = (n_hid_A + n_hid_B + 1):(n_hid_A + n_hid_B + H_ADD)

    # Reference data-side <h>, computed up front (before sampling) so
    # gibbs_sample can track a correlation-vs-sweep-count trace as it runs.
    h_data_A = Array(Float64.(sample_h_from_v(rbm_A, XA_g)))
    h_data_B = Array(Float64.(sample_h_from_v(rbm_B, XB_g)))
    h_data_P = Array(Float64.(sample_h_from_v(rbm_paired, X_AB_g)))

    path_conv_A = joinpath(REPORT_DIR, "convergence_check_A_$(PARAMTAG).txt")
    path_conv_B = joinpath(REPORT_DIR, "convergence_check_B_$(PARAMTAG).txt")
    path_conv_P = joinpath(REPORT_DIR, "convergence_check_paired_$(PARAMTAG).txt")

    # RBM A fantasy-sample cache: keyed on single_suffix_A() (already encodes
    # N_HIDDEN_A/B, K, N_ITERS, etc.) plus A's own sampling params, so any
    # change to what A was trained or sampled with invalidates the cache
    # automatically -- same "tag everything, skip if cached" convention
    # train_potts_pfam.jl uses for the RBM A/B model files themselves. Lets
    # an experiment vary B/Paired's sampling params without repaying RBM A's
    # (often the most expensive) sampling cost every run.
    path_fantasy_A_cache = joinpath(OUTPUT_DIR, "fantasy_A_cache_$(single_suffix_A())_NSAMPLESA=$(N_SAMPLES_A)_STRIDE=$(GIBBS_STRIDE)_STEPSA=$(N_GIBBS_STEPS_A).hdf5")
    if isfile(path_fantasy_A_cache)
        println("--- RBM A fantasy samples: found cached samples, skipping sampling ---")
        println("Loading ← $path_fantasy_A_cache")
        fantasy_x_A = dev(h5read(path_fantasy_A_cache, "fantasy_x_A"))
        F_A = h5read(path_fantasy_A_cache, "F_A")
        conv_A = (; iters=h5read(path_fantasy_A_cache, "conv_iters"),
                    r_h=h5read(path_fantasy_A_cache, "conv_rh"),
                    r_v=h5read(path_fantasy_A_cache, "conv_rv"),
                    r_hv=h5read(path_fantasy_A_cache, "conv_rhv"))
    else
        println("Sampling RBM A ($N_SAMPLES_A samples, $(N_GIBBS_STEPS_A*GIBBS_STRIDE) Gibbs sweeps)...")
        fantasy_x_A, F_A, conv_A = gibbs_sample(rbm_A, N_SAMPLES_A, N_GIBBS_STEPS_A, GIBBS_STRIDE;
            ref_h=h_data_A, ref_x=XA_c, log_every=CONVERGENCE_LOG_EVERY, path_conv=path_conv_A)
        h5open(path_fantasy_A_cache, "w") do io
            write(io, "fantasy_x_A", Array(fantasy_x_A))
            write(io, "F_A", F_A)
            write(io, "conv_iters", collect(conv_A.iters))
            write(io, "conv_rh", collect(conv_A.r_h))
            write(io, "conv_rv", collect(conv_A.r_v))
            write(io, "conv_rhv", collect(conv_A.r_hv))
        end
        println("Saved fantasy RBM A samples → $path_fantasy_A_cache")
    end
    println("Sampling RBM B ($N_SAMPLES_B samples, $(N_GIBBS_STEPS_B*GIBBS_STRIDE) Gibbs sweeps)...")
    fantasy_x_B, F_B, conv_B = gibbs_sample(rbm_B, N_SAMPLES_B, N_GIBBS_STEPS_B, GIBBS_STRIDE;
        ref_h=h_data_B, ref_x=XB_c, log_every=CONVERGENCE_LOG_EVERY, path_conv=path_conv_B)
    println("Sampling Paired RBM ($N_SAMPLES_PAIRED samples, $(N_GIBBS_STEPS_PAIRED*GIBBS_STRIDE) Gibbs sweeps)...")
    fantasy_x_paired, F_P, conv_P = gibbs_sample(rbm_paired, N_SAMPLES_PAIRED, N_GIBBS_STEPS_PAIRED, GIBBS_STRIDE;
        ref_h=h_data_P, ref_x=X_AB_c, hid_range=hid_add_range, log_every=CONVERGENCE_LOG_EVERY, path_conv=path_conv_P)
    fantasy_x_paired_A = vis_slice(fantasy_x_paired, 1:N_VIS_A)
    fantasy_x_paired_B = vis_slice(fantasy_x_paired, (N_VIS_A + 1):(N_VIS_A + N_VIS_B))

    # ---- Free energy (burn-in sanity check) ----
    p1, p2, p3 = free_energy_panel(F_A, "Free energy — RBM A"), free_energy_panel(F_B, "Free energy — RBM B"), free_energy_panel(F_P, "Free energy — Paired RBM")
    savefig(Plots.plot(p1, p2, p3; layout=(1, 3), size=(3 * CELL, CELL)), figpath(arch, "free_energy"))

    # ---- PCA: where does this run's training subset and generated samples
    # sit relative to the full population? Same PCA basis (cached above,
    # shared across every run against this FASTA) for every panel, so
    # different runs' figures are directly visually comparable. ----
    pca_train_proj = pca_project(X_AB)
    pca_gen_proj   = pca_project(Array(fantasy_x_paired))
    savefig(pca_scatter_panel(pca_train_proj, pca_gen_proj, "Paired RBM: training data + generated samples"),
        figpath(arch, "pca_scatter"))

    # ---- Convergence check: correlation vs cumulative sweep count. This is
    # the direct, per-model measurement of where sampling quality peaks and
    # starts degrading, replacing the earlier guess-a-big-budget approach —
    # see the CONFIG section's note on N_GIBBS_STEPS above. ----
    function conv_panel(conv, key, title_str)
        Plots.plot(conv.iters, getfield(conv, key); xlabel="cumulative sweeps", ylabel="r",
            title=title_str, titlefontsize=10, label="", marker=:circle, markersize=3,
            color=:seagreen, framestyle=:box, grid=false)
    end
    cp_h  = [conv_panel(c, :r_h, "⟨h⟩ vs sweeps — $lbl") for (c, lbl) in zip((conv_A, conv_B, conv_P), ("RBM A", "RBM B", "Paired (added)"))]
    cp_v  = [conv_panel(c, :r_v, "⟨v⟩ vs sweeps — $lbl") for (c, lbl) in zip((conv_A, conv_B, conv_P), ("RBM A", "RBM B", "Paired"))]
    cp_hv = [conv_panel(c, :r_hv, "⟨hv⟩ vs sweeps — $lbl") for (c, lbl) in zip((conv_A, conv_B, conv_P), ("RBM A", "RBM B", "Paired (added)"))]
    savefig(Plots.plot(cp_h..., cp_v..., cp_hv...; layout=(3, 3), size=(3 * CELL, 3 * CELL)), figpath(arch, "convergence_check"))

    # ---- Individual RBM validation (one-hot-level moments — matches what the model was trained on) ----
    # h_data_A/h_data_B/XA_c/XB_c were already computed above, before sampling.
    h_model_A = Array(Float64.(sample_h_from_v(rbm_A, fantasy_x_A)))
    h_model_B = Array(Float64.(sample_h_from_v(rbm_B, fantasy_x_B)))
    fantasy_x_A_c, fantasy_x_B_c = Array(fantasy_x_A), Array(fantasy_x_B)

    ph1, _ = scatter_panel(vec(mean(h_data_A; dims=2)), vec(mean(h_model_A; dims=2)), "⟨h⟩ — RBM A", "data", "model"; color=:dodgerblue)
    ph2, _ = scatter_panel(vec(mean(h_data_B; dims=2)), vec(mean(h_model_B; dims=2)), "⟨h⟩ — RBM B", "data", "model"; color=:orange)

    # Diagnostic: the ⟨h⟩ panels above average actual *stochastic* h samples
    # (sample_h_from_v) over the batch, whereas the training-time h_check
    # curve compares *mean-field* activations (mean_h_from_v — the exact
    # conditional expectation E[h|v], with no sampling noise of its own). If
    # ⟨h⟩ here looks much worse than h_check's last value, this panel — using
    # the same long-equilibrated data/fantasy batches but mean-field instead
    # of sampled — isolates how much of that gap is just stochastic-sampling
    # noise stacked on top of the mean-field signal, vs. a genuine difference
    # between the short training-time chains and these long ones.
    hmf_data_A  = Array(Float64.(mean_h_from_v(rbm_A, XA_g)))
    hmf_model_A = Array(Float64.(mean_h_from_v(rbm_A, fantasy_x_A)))
    hmf_data_B  = Array(Float64.(mean_h_from_v(rbm_B, XB_g)))
    hmf_model_B = Array(Float64.(mean_h_from_v(rbm_B, fantasy_x_B)))
    phmf1, _ = scatter_panel(vec(mean(hmf_data_A; dims=2)), vec(mean(hmf_model_A; dims=2)), "⟨h⟩ mean-field — RBM A", "data", "model"; color=:dodgerblue)
    phmf2, _ = scatter_panel(vec(mean(hmf_data_B; dims=2)), vec(mean(hmf_model_B; dims=2)), "⟨h⟩ mean-field — RBM B", "data", "model"; color=:orange)
    pv1, _ = scatter_panel(visible_means(XA_c), visible_means(fantasy_x_A_c), "⟨v⟩ — RBM A", "data", "model"; color=:dodgerblue)
    pv2, _ = scatter_panel(visible_means(XB_c), visible_means(fantasy_x_B_c), "⟨v⟩ — RBM B", "data", "model"; color=:orange)
    phv1, _ = scatter_panel(vec(hv_moments(h_data_A, XA_c)), vec(hv_moments(h_model_A, fantasy_x_A_c)), "⟨hv⟩ — RBM A", "data", "model"; color=:dodgerblue)
    phv2, _ = scatter_panel(vec(hv_moments(h_data_B, XB_c)), vec(hv_moments(h_model_B, fantasy_x_B_c)), "⟨hv⟩ — RBM B", "data", "model"; color=:orange)
    pvv1, _ = scatter_panel_vv(XA_c, fantasy_x_A_c, "⟨vv⟩ — RBM A"; color=:dodgerblue)
    pvv2, _ = scatter_panel_vv(XB_c, fantasy_x_B_c, "⟨vv⟩ — RBM B"; color=:orange)
    # Combined into one 4x3 grid (rows h/v/hv/vv, columns A/B/Paired) together
    # with the paired-model panels below, once those are computed too — see
    # `savefig(... "moments_validation")` after phvp/pvvp.

    # ---- Individual RBM pairwise coupling strength (site-level, Frobenius-norm — see connected_corr_full) ----
    CA_data  = zero_diag!(frob_strength_matrix(XA_c, XA_c))
    CB_data  = zero_diag!(frob_strength_matrix(XB_c, XB_c))
    CA_model = zero_diag!(frob_strength_matrix(fantasy_x_A_c, fantasy_x_A_c))
    CB_model = zero_diag!(frob_strength_matrix(fantasy_x_B_c, fantasy_x_B_c))

    hA1, hA2 = heat_panel(CA_data, "Coupling strength data — A"), heat_panel(CA_model, "Coupling strength model — A")
    hB1, hB2 = heat_panel(CB_data, "Coupling strength data — B"), heat_panel(CB_model, "Coupling strength model — B")
    savefig(Plots.plot(hA1, hA2, hB1, hB2; layout=(2, 2), size=(2 * CELL, 2 * CELL)), figpath(arch, "individual_correlations"))

    # ---- Paired RBM — full model moments ----
    # h_data_P/X_AB_c were already computed above, before sampling.
    h_model_P = Array(Float64.(sample_h_from_v(rbm_paired, fantasy_x_paired)))
    fantasy_P_c = Array(fantasy_x_paired)
    X_AB_flat, fantasy_P_flat = flatten(X_AB_c), flatten(fantasy_P_c)
    n_data, n_samp = size(X_AB_flat, 2), size(fantasy_P_flat, 2)

    # ⟨h⟩/⟨hv⟩ restricted to the added units only (hid_add_range, defined
    # above), matching the scope of the training-time h_check_paired/
    # vh_check_paired curves (and of wn_check_paired/gn_check_paired/
    # firing_check_paired) — these are the only free parameters, so including
    # the frozen A/B blocks here would contaminate the comparison with
    # whatever those blocks' own already-known behaviour is (see the A/B
    # panels), rather than isolating what the added units themselves learned.
    php, _  = scatter_panel(vec(mean(h_data_P[hid_add_range, :]; dims=2)), vec(mean(h_model_P[hid_add_range, :]; dims=2)), "⟨h⟩ — Paired RBM (added units)", "data", "model"; color=:purple)
    pvp, _  = scatter_panel(visible_means(X_AB_c), visible_means(fantasy_P_c), "⟨v⟩ — Paired RBM", "data", "model"; color=:purple)
    hv_data_P  = h_data_P[hid_add_range, :]  * X_AB_flat' / n_data
    hv_model_P = h_model_P[hid_add_range, :] * fantasy_P_flat' / n_samp
    phvp, _ = scatter_panel(vec(hv_data_P), vec(hv_model_P), "⟨hv⟩ — Paired RBM (added units)", "data", "model"; color=:purple)
    pvvp, _ = scatter_panel_vv(X_AB_c, fantasy_P_c, "⟨vv⟩ — Paired RBM (all)"; color=:purple)

    hmf_data_P  = Array(Float64.(mean_h_from_v(rbm_paired, X_AB_g)))[hid_add_range, :]
    hmf_model_P = Array(Float64.(mean_h_from_v(rbm_paired, fantasy_x_paired)))[hid_add_range, :]
    phmfp, _ = scatter_panel(vec(mean(hmf_data_P; dims=2)), vec(mean(hmf_model_P; dims=2)), "⟨h⟩ mean-field — Paired (added)", "data", "model"; color=:purple)
    savefig(Plots.plot(phmf1, phmf2, phmfp; layout=(1, 3), size=(3 * CELL, CELL)), figpath(arch, "h_meanfield_check"))

    # One consolidated 4x3 grid, all three machines side by side: rows are
    # ⟨h⟩/⟨v⟩/⟨hv⟩/⟨vv⟩ (matching the training-time h_check/v_check/vh_check/
    # vv_check curves in analyze_results_pfam.jl), columns are RBM A / RBM B /
    # Paired. At the end of training these post-sampling correlations should
    # land in the same ballpark as the corresponding curve's last value — this
    # is the figure to check that against. ⟨vv⟩ uses the same raw
    # (site-pair, colour-pair) granularity as the vv_check curve, unlike the
    # individual_correlations/family_correlation_preservation heatmaps, which
    # aggregate each site pair down to one Frobenius score for visualizability.
    savefig(Plots.plot(ph1, ph2, php, pv1, pv2, pvp, phv1, phv2, phvp, pvv1, pvv2, pvvp;
                        layout=(4, 3), size=(3 * CELL, 4 * CELL)), figpath(arch, "moments_validation"))

    # ---- Paired RBM — block-wise ⟨hv⟩ moments: the direct test of what the ----
    # ---- added hidden units learned, versus what stayed correctly frozen. ----
    hv_data_AA  = h_data_P[hid_A_range, :]  * X_AB_flat[vis_A_range, :]' / n_data
    hv_model_AA = h_model_P[hid_A_range, :] * fantasy_P_flat[vis_A_range, :]' / n_samp
    hv_data_BB  = h_data_P[hid_B_range, :]  * X_AB_flat[vis_B_range, :]' / n_data
    hv_model_BB = h_model_P[hid_B_range, :] * fantasy_P_flat[vis_B_range, :]' / n_samp
    hv_data_addA  = h_data_P[hid_add_range, :]  * X_AB_flat[vis_A_range, :]' / n_data
    hv_model_addA = h_model_P[hid_add_range, :] * fantasy_P_flat[vis_A_range, :]' / n_samp
    hv_data_addB  = h_data_P[hid_add_range, :]  * X_AB_flat[vis_B_range, :]' / n_data
    hv_model_addB = h_model_P[hid_add_range, :] * fantasy_P_flat[vis_B_range, :]' / n_samp
    hv_data_addfull  = h_data_P[hid_add_range, :]  * X_AB_flat' / n_data
    hv_model_addfull = h_model_P[hid_add_range, :] * fantasy_P_flat' / n_samp

    pAA, ρ_AA = scatter_panel(hv_data_AA, hv_model_AA, "⟨h_A v_A⟩", "data", "model"; color=:dodgerblue)
    pBB, ρ_BB = scatter_panel(hv_data_BB, hv_model_BB, "⟨h_B v_B⟩", "data", "model"; color=:orange)
    pAF, ρ_addfull = scatter_panel(hv_data_addfull, hv_model_addfull, "⟨h_add v_full⟩", "data", "model"; color=:seagreen)
    pAA2, ρ_addA = scatter_panel(hv_data_addA, hv_model_addA, "⟨h_add v_A⟩", "data", "model"; color=:seagreen)
    pAB2, ρ_addB = scatter_panel(hv_data_addB, hv_model_addB, "⟨h_add v_B⟩", "data", "model"; color=:seagreen)
    savefig(Plots.plot(pAA, pBB, pAF, pAA2, pAB2; layout=(2, 3), size=(3 * CELL, 2 * CELL)), figpath(arch, "paired_block_hv"))

    # ---- Cross-family coupling strength vs held-out reference (the definitive test) ----
    # No synthetic K exists for real sequences — XA_val/XB_val (never trained
    # on) stand in as the best available "ground truth" for what real
    # cross-family covariation looks like. Comparing training-set XA/XB against
    # this same reference (C_AB_train) gives a finite-sample-noise baseline:
    # even a perfect model can't beat how well *raw data* generalizes from
    # train to held-out.
    C_AB_val    = frob_strength_matrix(XA_val, XB_val)
    C_AB_train  = frob_strength_matrix(XA_c, XB_c)
    C_AB_paired = frob_strength_matrix(fantasy_x_paired_A, fantasy_x_paired_B)
    # frob_strength_matrix pairs samples positionally (needs matching batch
    # dims) -- fantasy_x_A_c/fantasy_x_B_c can now come from independently
    # -sized N_SAMPLES_A/N_SAMPLES_B (e.g. a cached RBM A run vs a fresh RBM B
    # run), so truncate both to their common size before this comparison.
    n_indep = min(size(fantasy_x_A_c, 3), size(fantasy_x_B_c, 3))
    C_AB_indep  = frob_strength_matrix(view(fantasy_x_A_c, :, :, 1:n_indep), view(fantasy_x_B_c, :, :, 1:n_indep))  # independent RBMs = zero-coupling baseline

    hv1, hv2 = heat_panel(C_AB_val, "Held-out data (A-B)"), heat_panel(C_AB_train, "Train data (A-B)")
    hv3, hv4 = heat_panel(C_AB_paired, "Paired RBM (A-B)"), heat_panel(C_AB_indep, "Independent RBMs (A-B)")
    savefig(Plots.plot(hv1, hv2, hv3, hv4; layout=(2, 2), size=(2 * CELL, 2 * CELL)), figpath(arch, "cross_correlation_vs_heldout"))

    Kn         = max(1, round(Int, TOP_FRAC * N_VIS_A * N_VIS_B))
    top_val    = sortperm(vec(C_AB_val),    rev=true)[1:Kn]
    top_train  = sortperm(vec(C_AB_train),  rev=true)[1:Kn]
    top_paired = sortperm(vec(C_AB_paired), rev=true)[1:Kn]
    top_indep  = sortperm(vec(C_AB_indep),  rev=true)[1:Kn]

    cor_paired_vs_val = cor(vec(C_AB_paired), vec(C_AB_val))
    cor_train_vs_val  = cor(vec(C_AB_train),  vec(C_AB_val))
    cor_indep_vs_val  = cor(vec(C_AB_indep),  vec(C_AB_val))
    overlap_paired = length(intersect(top_val, top_paired)) / Kn
    overlap_train  = length(intersect(top_val, top_train))  / Kn
    overlap_indep  = length(intersect(top_val, top_indep))  / Kn

    # Which SPECIFIC top-K pairs does the paired model get right, not just
    # the aggregate overlap fraction? Spatial view (where on the protein)
    # and magnitude view (how close were the misses) side by side, both
    # driven by the same top_val/top_paired sets computed above.
    cats_paired = hit_miss_categories(top_val, top_paired, length(vec(C_AB_val)))
    hm_spatial = hit_miss_heat_panel(cats_paired, size(C_AB_val), "Paired vs held-out top-$Kn")
    hm_scatter = hit_miss_scatter_panel(C_AB_val, C_AB_paired, top_val, top_paired, Kn,
        "Paired vs held-out top-$Kn (ρ=$(round(cor_paired_vs_val; digits=3)))")
    # Extra width vs. the other 2-panel figures above: each panel now carries
    # its own outer-right legend (see hit_miss_heat_panel/hit_miss_scatter_panel),
    # which needs room beyond the plain plot area or it gets clipped.
    # Extra height (not just width) vs. the plain size=(2*CELL, CELL) used
    # elsewhere: this figure's title + legend + x-axis label stack up enough
    # that the plain CELL height clips "held-out strength" off the bottom
    # edge -- confirmed by cropping and inspecting the rendered PNG directly,
    # not just eyeballing the inline preview.
    savefig(Plots.plot(hm_spatial, hm_scatter; layout=(1, 2), size=(2 * CELL + 240, CELL + 60),
        bottom_margin=6Plots.mm), figpath(arch, "top_k_hit_miss"))

    # ---- Family preservation: does pairing corrupt each family's own stats? ----
    h_data_A_pres  = Array(Float64.(sample_h_from_v(rbm_A, XA_g)))
    h_model_A_pres = Array(Float64.(sample_h_from_v(rbm_A, fantasy_x_paired_A)))
    h_data_B_pres  = Array(Float64.(sample_h_from_v(rbm_B, XB_g)))
    h_model_B_pres = Array(Float64.(sample_h_from_v(rbm_B, fantasy_x_paired_B)))
    fantasy_x_paired_A_c, fantasy_x_paired_B_c = Array(fantasy_x_paired_A), Array(fantasy_x_paired_B)

    pp1, _ = scatter_panel(vec(mean(h_data_A_pres; dims=2)), vec(mean(h_model_A_pres; dims=2)), "⟨h⟩ family A", "data", "model"; color=:dodgerblue)
    pp2, _ = scatter_panel(visible_means(XA_c), visible_means(fantasy_x_paired_A_c), "⟨v⟩ family A", "data", "model"; color=:dodgerblue)
    pp3, _ = scatter_panel(vec(hv_moments(h_data_A_pres, XA_c)), vec(hv_moments(h_model_A_pres, fantasy_x_paired_A_c)), "⟨hv⟩ family A", "data", "model"; color=:dodgerblue)
    pp4, _ = scatter_panel(vec(mean(h_data_B_pres; dims=2)), vec(mean(h_model_B_pres; dims=2)), "⟨h⟩ family B", "data", "model"; color=:orange)
    pp5, _ = scatter_panel(visible_means(XB_c), visible_means(fantasy_x_paired_B_c), "⟨v⟩ family B", "data", "model"; color=:orange)
    pp6, _ = scatter_panel(vec(hv_moments(h_data_B_pres, XB_c)), vec(hv_moments(h_model_B_pres, fantasy_x_paired_B_c)), "⟨hv⟩ family B", "data", "model"; color=:orange)
    savefig(Plots.plot(pp1, pp2, pp3, pp4, pp5, pp6; layout=(2, 3), size=(3 * CELL, 2 * CELL)), figpath(arch, "family_preservation"))

    CA_paired_model = zero_diag!(frob_strength_matrix(fantasy_x_paired_A_c, fantasy_x_paired_A_c))
    CB_paired_model = zero_diag!(frob_strength_matrix(fantasy_x_paired_B_c, fantasy_x_paired_B_c))

    hc1, hc2 = heat_panel(CA_data, "Coupling strength data — A"), heat_panel(CA_paired_model, "Coupling strength paired — A")
    hc3, hc4 = heat_panel(CB_data, "Coupling strength data — B"), heat_panel(CB_paired_model, "Coupling strength paired — B")
    savefig(Plots.plot(hc1, hc2, hc3, hc4; layout=(2, 2), size=(2 * CELL, 2 * CELL)), figpath(arch, "family_correlation_preservation"))

    summaries[label] = (;
        ρ_AA, ρ_BB, ρ_addA, ρ_addB, ρ_addfull,
        Kn, overlap_paired, overlap_train, overlap_indep,
        cor_paired_vs_val, cor_train_vs_val, cor_indep_vs_val,
        preservation_A_data = mean_offdiag(CA_data),
        preservation_A_paired = mean_offdiag(CA_paired_model),
        preservation_B_data = mean_offdiag(CB_data),
        preservation_B_paired = mean_offdiag(CB_paired_model),
    )

    println("\nBlock-wise ⟨hv⟩ correlations (data vs model, from real equilibrium samples):")
    println("  ⟨h_A  v_A⟩    ρ = $ρ_AA   (frozen block — should be ≈1)")
    println("  ⟨h_B  v_B⟩    ρ = $ρ_BB   (frozen block — should be ≈1)")
    println("  ⟨h_add v_A⟩   ρ = $ρ_addA (learned cross-family signal)")
    println("  ⟨h_add v_B⟩   ρ = $ρ_addB (learned cross-family signal)")
    println("  ⟨h_add v_full⟩ ρ = $ρ_addfull (learned cross-family signal, whole visible layer at once)")
    println("\nCross-family coupling strength vs held-out reference:")
    println("  top-$Kn overlap — paired RBM vs held-out: $(round(overlap_paired; digits=3))")
    println("  top-$Kn overlap — train data vs held-out: $(round(overlap_train; digits=3))")
    println("  top-$Kn overlap — independent vs held-out: $(round(overlap_indep; digits=3))")
    println("  corr(paired, held-out) = $(round(cor_paired_vs_val; digits=3))  corr(train, held-out) = $(round(cor_train_vs_val; digits=3))  corr(indep, held-out) = $(round(cor_indep_vs_val; digits=3))")
end

# =============================================================================
# WRITTEN REPORT
# =============================================================================
io = IOBuffer()
rp(x...) = println(io, x...)
rp("# Sampled-model validation (protein pair $(DATASET_TAG)) — N_HIDDEN_A=$(N_HIDDEN_A), N_HIDDEN_B=$(N_HIDDEN_B), H_ADD=$(H_ADD), N_ITERS=$(N_ITERS), PAIRED_ITERS=$(PAIRED_ITERS)")
rp()
rp("Real Gibbs samples ($N_SAMPLES_A/$N_SAMPLES_B/$N_SAMPLES_PAIRED samples for A/B/Paired) from the actual saved models — $(N_GIBBS_STEPS_A*GIBBS_STRIDE) sweeps burn-in for RBM A, $(N_GIBBS_STEPS_B*GIBBS_STRIDE) for RBM B, $(N_GIBBS_STEPS_PAIRED*GIBBS_STRIDE) for the paired model (A gets far more: see the free-energy diagnostic) —")
rp("not the short training-time chains `analyze_results_pfam.jl` reports on. Figures in `$(REPORT_DIR)/`.")
rp()
rp("No synthetic ground-truth coupling matrix exists for real sequences, so the")
rp("\"definitive test\" below uses the held-out validation split (never trained on) as the")
rp("best available truth proxy, and the *training-set* data's own agreement with that")
rp("held-out split as a finite-sample-noise baseline (see DATA / connected_corr_full comments in the script).")
rp()
for (label, s) in summaries
    rp("## $label")
    rp()
    rp("**Block-wise ⟨hv⟩ correlation (data vs model, real samples):**")
    rp("- frozen blocks: ⟨h_A v_A⟩ ρ=$(round(s.ρ_AA;digits=3)), ⟨h_B v_B⟩ ρ=$(round(s.ρ_BB;digits=3)) — should be ≈1")
    rp("- **added units — cross-family signal**: ⟨h_add v_A⟩ ρ=$(round(s.ρ_addA;digits=3)), ⟨h_add v_B⟩ ρ=$(round(s.ρ_addB;digits=3)), ⟨h_add v_full⟩ ρ=$(round(s.ρ_addfull;digits=3))")
    rp()
    rp("**Cross-family coupling strength vs held-out reference (the definitive test):**")
    rp("- top-$(s.Kn) strongest-pair overlap: paired RBM=$(round(s.overlap_paired;digits=3)), train data=$(round(s.overlap_train;digits=3)), independent-RBM baseline=$(round(s.overlap_indep;digits=3))")
    rp("- correlation of full coupling-strength matrix against held-out data: paired=$(round(s.cor_paired_vs_val;digits=3)), train data=$(round(s.cor_train_vs_val;digits=3)), independent=$(round(s.cor_indep_vs_val;digits=3))")
    gap = s.overlap_paired - s.overlap_indep
    rp("  → paired model beats the zero-coupling (independent-RBM) baseline by $(round(gap;digits=3)) in top-K overlap" *
       (gap > 0.1 ? " — real cross-family structure recovered." : " — little to no improvement over having learned nothing about family coupling."))
    rp()
    rp("**Family preservation (mean off-diagonal coupling strength, closer to data value is tighter):**")
    rp("- A: data=$(round(s.preservation_A_data;digits=4)), within paired model=$(round(s.preservation_A_paired;digits=4))")
    rp("- B: data=$(round(s.preservation_B_data;digits=4)), within paired model=$(round(s.preservation_B_paired;digits=4))")
    rp()
end

if length(summaries) == 2
    labels = collect(keys(summaries))
    rp("## Potts vs Binary — head-to-head (real samples)")
    rp()
    rp("| metric | $(labels[1]) | $(labels[2]) |")
    rp("|---|---|---|")
    rp("| ⟨h_add v_A⟩ ρ | $(round(summaries[labels[1]].ρ_addA;digits=3)) | $(round(summaries[labels[2]].ρ_addA;digits=3)) |")
    rp("| ⟨h_add v_B⟩ ρ | $(round(summaries[labels[1]].ρ_addB;digits=3)) | $(round(summaries[labels[2]].ρ_addB;digits=3)) |")
    rp("| ⟨h_add v_full⟩ ρ | $(round(summaries[labels[1]].ρ_addfull;digits=3)) | $(round(summaries[labels[2]].ρ_addfull;digits=3)) |")
    rp("| top-K overlap vs held-out | $(round(summaries[labels[1]].overlap_paired;digits=3)) | $(round(summaries[labels[2]].overlap_paired;digits=3)) |")
    rp("| corr(paired coupling strength, held-out) | $(round(summaries[labels[1]].cor_paired_vs_val;digits=3)) | $(round(summaries[labels[2]].cor_paired_vs_val;digits=3)) |")
end

report = String(take!(io))
open(joinpath(REPORT_DIR, "report.md"), "w") do f
    write(f, report)
end
println("\n" * report)
println("Report + figures written to $(REPORT_DIR)/")
