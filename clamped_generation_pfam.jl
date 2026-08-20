using CUDA
using HDF5
using RestrictedBoltzmannMachines: load_rbm, gpu, cpu
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_h, sample_from_inputs, Falses
using LatentAlignedRBMs
using FASTX
using BioSequences
using Statistics
using Random

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
# Generalization test #2: clamp one family's domain to a real, held-out
# sequence and ask the model to *generate* the other family's domain, then
# compare to the true cognate partner. Unlike the paralog-pairing test (which
# only ranks a small set of real candidates by free energy), this asks the
# model to produce a sequence unconstrained, which is a strictly harder and
# more direct test of what the cross-family (H_ADD) coupling has learned.
#
# There is no built-in clamped/conditional sampling in
# RestrictedBoltzmannMachines -- sample_v_from_v bundles a full h-then-v
# sweep with no hook to fix a subset of visible sites. So this hand-rolls the
# same two primitives sample_v_from_v itself composes internally
# (sample_h_from_v, sample_v_from_h), re-overwriting the clamped block back
# to its fixed value after every v-update.
#
# Run on train_potts_pfam.jl's held-out validation split specifically (not
# the paralog-test species pool), for the cleanest possible generalization
# claim: rbm_paired never saw these sequences during training. Both
# directions (A->B and B->A) are run, since N_HIDDEN_A=150 != N_HIDDEN_B=100
# means the two directions are not guaranteed to be symmetric.
#
# Four numbers per direction disentangle what's actually being measured:
#   1. true-partner recovery   : generated(A_i) vs. true B_i        (main result)
#   2. wrong-partner control   : generated(A_i) vs. true B_j, j!=i  (did clamping
#                                 A_i specifically matter, or is this just "a
#                                 plausible generic B" regardless of which A?)
#   3. consensus baseline      : per-site most common residue (over the
#                                 training set) vs. true B_i        (how much
#                                 is explained by conservation alone, no model)
#   4. real-to-real similarity : true B_i vs. true B_j, j!=i        (context:
#                                 how conserved is this family already?)
const FASTA_PATH  = length(ARGS) >= 2 ? ARGS[2] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 111
const TRAIN_FRAC  = 0.7
const VAL_FRAC    = 0.15
const DEDUP_IDENTITY_THRESHOLD = 0.97   # must match train_potts_pfam.jl, to load its split cache
const OUTPUT_DIR  = "./results_pfam"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

const PATH_RBM_PAIRED = ARGS[1]
run_tag    = replace(basename(PATH_RBM_PAIRED), r"^rbm_paired_" => "", r"\.hdf5$" => "")
REPORTPATH = joinpath(OUTPUT_DIR, "clamped_generation_$(run_tag).txt")

const N_HELDOUT     = 2000   # capped subsample of the validation split, for tractable runtime
const N_GIBBS_STEPS = 300    # same protocol as pair_results_pfam.jl / generation_coverage_pfam.jl
const GIBBS_STRIDE  = 100
const LOG_EVERY     = 20     # checkpoints for the convergence trace
# A single stochastic Gibbs sample is not a fair comparison against
# consensus's deterministic "always guess the mode" strategy -- even a
# perfectly-calibrated model's one sample will underperform a point estimate
# whenever its conditional distribution isn't degenerate, and that gap grows
# specifically wherever the true distribution is more peaked (i.e. more
# conserved sites), which is exactly the pattern the first version of this
# script showed. N_REPLICAS independent chains per held-out sequence gives a
# per-site majority vote -- a point estimate comparable to consensus.
const N_REPLICAS    = 10

println("rbm_paired : $PATH_RBM_PAIRED")
println("dataset    : $FASTA_PATH (split at $SPLIT_SITE)")
println("report     -> $REPORTPATH")

# =============================================================================
# DATA
# =============================================================================
function load_records(path)
    reader = open(FASTA.Reader, path)
    seqs = LongAA[]
    for record in reader
        push!(seqs, LongAA(FASTA.sequence(record)))
    end
    close(reader)
    return seqs
end

seqs_raw    = load_records(FASTA_PATH)
seqs_onehot = LatentAlignedRBMs.onehot(seqs_raw)   # (q, n_sites, n_samples) BitArray
q, n_sites, n_samples = size(seqs_onehot)
n_vis_A = SPLIT_SITE
n_vis_B = n_sites - SPLIT_SITE
println("q=$q  n_sites=$n_sites  n_vis_A=$n_vis_A  n_vis_B=$n_vis_B  n_samples=$n_samples")

# train_potts_pfam.jl now uses a species/duplicate/near-duplicate-grouped
# split (grouped_train_val_split) and caches the result -- loading that cache
# directly (rather than reimplementing the grouping logic here) guarantees
# train_idx/val_idx match what rbm_paired actually did/didn't train on.
fasta_tag = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
split_cache_path = joinpath(OUTPUT_DIR, "split_cache_$(fasta_tag)_TRAIN=$(TRAIN_FRAC)_VAL=$(VAL_FRAC)_DEDUP=$(DEDUP_IDENTITY_THRESHOLD).hdf5")
isfile(split_cache_path) || error("No split cache at $split_cache_path -- run train_potts_pfam.jl on this FASTA_PATH first (it creates this cache).")
train_idx = h5read(split_cache_path, "train_idx")
val_idx   = h5read(split_cache_path, "val_idx")
println("loaded train/val split ← $split_cache_path ($(length(train_idx)) train, $(length(val_idx)) val)")

n_heldout = min(N_HELDOUT, length(val_idx))
heldout_idx = val_idx[randperm(length(val_idx))[1:n_heldout]]
println("held-out sequences used: $n_heldout (of $(length(val_idx)) available in the validation split)")

XA_heldout = seqs_onehot[:, 1:n_vis_A, heldout_idx]
XB_heldout = seqs_onehot[:, n_vis_A+1:end, heldout_idx]
XA_train   = seqs_onehot[:, 1:n_vis_A, train_idx]
XB_train   = seqs_onehot[:, n_vis_A+1:end, train_idx]

# =============================================================================
# MODEL
# =============================================================================
rbm = gpu(load_rbm(PATH_RBM_PAIRED))

# =============================================================================
# HELPERS
# =============================================================================
# Category index per site (argmax over the one-hot/q dimension), for
# comparing generated vs. true sequences site-by-site rather than cell-by-cell.
cat_indices(x) = dropdims(map(ci -> ci[1], argmax(Array(x); dims=1)); dims=1)  # (n_sites, batch)

mean_identity(a_idx, b_idx) = mean(a_idx .== b_idx)
# Per-site match rate (average over the batch dimension only), so sites can
# be stratified by conservation afterwards -- a single scalar `mean_identity`
# hides whether a method wins because it's actually predicting the variable,
# informative sites, or just because most sites are trivially conserved.
site_match_rate(a_idx, b_idx) = vec(mean(a_idx .== b_idx; dims=2))   # (n_sites,)

# Per-site consensus (most common category) over a reference set, plus its
# conservation (the winning category's frequency, i.e. how easy that site is
# for a method that ignores A entirely).
function consensus_stats(ref_onehot)
    q, n, m = size(ref_onehot)
    freqs = sum(Float32.(ref_onehot); dims=3) ./ m           # (q, n, 1)
    idx   = dropdims(map(ci -> ci[1], argmax(freqs; dims=1)); dims=(1, 3))  # (n,)
    conservation = vec(maximum(freqs; dims=1))               # (n,)
    return idx, conservation
end

# Quartile breakdown by site conservation (Q1 = most variable/informative
# sites, Q4 = most conserved/trivial-for-consensus sites), reporting the mean
# of each per-site metric within each bucket. This is the fairer test of
# whether clamped generation adds value beyond the consensus baseline: that
# baseline is unbeatable by construction on conserved sites, so any real
# effect of conditioning on A has to show up specifically in the low
# conservation buckets.
function conservation_breakdown(conservation, metrics::NamedTuple)
    edges = quantile(conservation, [0.25, 0.5, 0.75])
    bucket(c) = c <= edges[1] ? 1 : c <= edges[2] ? 2 : c <= edges[3] ? 3 : 4
    labels = ["Q1 (most variable)", "Q2", "Q3", "Q4 (most conserved)"]
    rows = NamedTuple[]
    for b in 1:4
        mask = bucket.(conservation) .== b
        metric_means = NamedTuple(k => mean(v[mask]) for (k, v) in pairs(metrics))
        row = merge((; label = labels[b], n_sites = count(mask),
                       conservation_range = (minimum(conservation[mask]), maximum(conservation[mask]))),
                    metric_means)
        push!(rows, row)
    end
    return rows
end

# Per-site majority vote across replica chains. idx_array: (n_sites, batch,
# n_replicas) category indices -> (n_sites, batch) mode category. Vectorized
# via per-category counts (q is small, ~20-30 amino acid categories) rather
# than a manual mode-finding loop over sites/batch.
function mode_over_replicas(idx_array, q)
    n, b, r = size(idx_array)
    counts = zeros(Int32, n, b, q)
    for k in 1:q
        counts[:, :, k] .= dropdims(sum(idx_array .== k; dims=3); dims=3)
    end
    return dropdims(map(ci -> ci[3], argmax(counts; dims=3)); dims=3)  # (n, b)
end

function clamped_generate(rbm, clamp_block, clamp_range, gen_range, gen_truth_idx)
    batch = size(clamp_block, 3)
    v = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batch))
    v[:, clamp_range, :] .= gpu(clamp_block)
    trace = Float64[]
    for t in 1:N_GIBBS_STEPS
        for _ in 1:GIBBS_STRIDE
            h = sample_h_from_v(rbm, v)
            v = sample_v_from_h(rbm, h)
            v[:, clamp_range, :] .= gpu(clamp_block)
        end
        if iszero(t % LOG_EVERY)
            gen_idx = cat_indices(v[:, gen_range, :])
            acc = mean_identity(gen_idx, gen_truth_idx)
            push!(trace, acc)
            println("    sweep $(t*GIBBS_STRIDE)/$(N_GIBBS_STEPS*GIBBS_STRIDE)  recovery=$(round(acc, digits=4))")
        end
    end
    return v[:, gen_range, :], trace
end

function run_direction(name, clamp_block, clamp_range, gen_range, gen_truth_block, ref_train_gen_block)
    println("\n--- $name ---")
    n_heldout_here = size(clamp_block, 3)
    gen_truth_idx  = cat_indices(gen_truth_block)   # (n_sites_gen, n_heldout)

    # Tile the batch N_REPLICAS-fold: `repeat(x,1,1,r)` concatenates r whole
    # copies along the batch dimension (not interleaved), so columns
    # 1:n_heldout are replica 1 for every original sequence, the next
    # n_heldout columns are replica 2, etc. -- each replica's randomness is
    # independent (per-element GPU sampling), giving N_REPLICAS independent
    # posterior draws per held-out sequence from one batched Gibbs run.
    clamp_tiled = repeat(clamp_block, 1, 1, N_REPLICAS)
    truth_tiled = repeat(gen_truth_idx, 1, N_REPLICAS)
    v_gen, trace = clamped_generate(rbm, clamp_tiled, clamp_range, gen_range, truth_tiled)
    gen_idx_tiled = cat_indices(v_gen)   # (n_sites_gen, n_heldout*N_REPLICAS)

    # Single-sample metrics (replica 1 only), as in the first version.
    gen_idx_single = gen_idx_tiled[:, 1:n_heldout_here]
    true_partner_recovery = mean_identity(gen_idx_single, gen_truth_idx)
    shift = circshift(1:n_heldout_here, 1)
    wrong_partner_control = mean_identity(gen_idx_single, gen_truth_idx[:, shift])

    consensus_idx, conservation = consensus_stats(ref_train_gen_block)
    consensus_baseline = mean_identity(repeat(consensus_idx, 1, n_heldout_here), gen_truth_idx)
    real_to_real = mean_identity(gen_truth_idx, gen_truth_idx[:, shift])

    # Majority-vote point estimate across N_REPLICAS independent chains --
    # the fair comparison against consensus's own deterministic point
    # estimate (see N_REPLICAS comment above).
    n_sites_gen = size(gen_idx_tiled, 1)
    gen_idx_3d  = reshape(gen_idx_tiled, n_sites_gen, n_heldout_here, N_REPLICAS)
    mode_idx    = mode_over_replicas(gen_idx_3d, q)
    true_partner_mode  = mean_identity(mode_idx, gen_truth_idx)
    wrong_partner_mode = mean_identity(mode_idx, gen_truth_idx[:, shift])

    println("true-partner recovery (single sample)        : $(round(true_partner_recovery, digits=4))")
    println("wrong-partner control (single sample)        : $(round(wrong_partner_control, digits=4))")
    println("true-partner recovery (majority vote, R=$N_REPLICAS)  : $(round(true_partner_mode, digits=4))")
    println("wrong-partner control (majority vote)        : $(round(wrong_partner_mode, digits=4))")
    println("consensus baseline                           : $(round(consensus_baseline, digits=4))")
    println("real-to-real similarity                      : $(round(real_to_real, digits=4))")

    # Site-conservation breakdown, now including the majority-vote point
    # estimate alongside the single-sample and consensus numbers: does the
    # model's *point estimate* beat consensus specifically on the sites where
    # consensus should struggle (low conservation)?
    per_site = (
        true_single    = site_match_rate(gen_idx_single, gen_truth_idx),
        true_mode       = site_match_rate(mode_idx, gen_truth_idx),
        wrong_single    = site_match_rate(gen_idx_single, gen_truth_idx[:, shift]),
        wrong_mode      = site_match_rate(mode_idx, gen_truth_idx[:, shift]),
        consensus       = site_match_rate(repeat(consensus_idx, 1, n_heldout_here), gen_truth_idx),
        real_to_real    = site_match_rate(gen_truth_idx, gen_truth_idx[:, shift]),
    )
    breakdown = conservation_breakdown(conservation, per_site)
    println("\n  site-conservation breakdown:")
    println("  " * rpad("bucket", 22) * rpad("n_sites", 9) * rpad("consv_range", 16) *
            rpad("true_single", 13) * rpad("true_mode", 11) * rpad("wrong_single", 14) *
            rpad("wrong_mode", 12) * rpad("consensus", 11) * "real_to_real")
    for r in breakdown
        rng = "($(round(r.conservation_range[1],digits=2)),$(round(r.conservation_range[2],digits=2)))"
        println("  " * rpad(r.label, 22) * rpad(string(r.n_sites), 9) * rpad(rng, 16) *
                rpad(string(round(r.true_single, digits=3)), 13) *
                rpad(string(round(r.true_mode, digits=3)), 11) *
                rpad(string(round(r.wrong_single, digits=3)), 14) *
                rpad(string(round(r.wrong_mode, digits=3)), 12) *
                rpad(string(round(r.consensus, digits=3)), 11) *
                string(round(r.real_to_real, digits=3)))
    end

    return (; name, trace, true_partner_recovery, wrong_partner_control,
              true_partner_mode, wrong_partner_mode, consensus_baseline, real_to_real, breakdown)
end

# =============================================================================
# RUN BOTH DIRECTIONS
# =============================================================================
range_A = 1:n_vis_A
range_B = (n_vis_A + 1):(n_vis_A + n_vis_B)

results = [
    run_direction("A -> B  (clamp A, generate B)", XA_heldout, range_A, range_B, XB_heldout, XB_train),
    run_direction("B -> A  (clamp B, generate A)", XB_heldout, range_B, range_A, XA_heldout, XA_train),
]

open(REPORTPATH, "w") do f
    println(f, "rbm_paired=$PATH_RBM_PAIRED")
    println(f, "n_heldout=$n_heldout")
    for r in results
        println(f, "direction=$(r.name)")
        println(f, "  true_partner_recovery_single=$(r.true_partner_recovery)")
        println(f, "  wrong_partner_control_single=$(r.wrong_partner_control)")
        println(f, "  true_partner_recovery_mode=$(r.true_partner_mode)")
        println(f, "  wrong_partner_control_mode=$(r.wrong_partner_mode)")
        println(f, "  consensus_baseline=$(r.consensus_baseline)")
        println(f, "  real_to_real_similarity=$(r.real_to_real)")
        println(f, "  convergence_trace=$(r.trace)")
        for row in r.breakdown
            println(f, "  bucket=$(row.label) n_sites=$(row.n_sites) conservation_range=$(row.conservation_range) true_single=$(row.true_single) true_mode=$(row.true_mode) wrong_single=$(row.wrong_single) wrong_mode=$(row.wrong_mode) consensus=$(row.consensus) real_to_real=$(row.real_to_real)")
        end
    end
end
println("\nsaved report -> $REPORTPATH")
