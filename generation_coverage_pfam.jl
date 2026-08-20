using CUDA
using HDF5
using RestrictedBoltzmannMachines: load_rbm, free_energy, gpu, cpu
using RestrictedBoltzmannMachines: sample_v_from_v, sample_from_inputs, Falses
using LatentAlignedRBMs
using FASTX
using BioSequences
using LinearAlgebra
using Statistics
using Random
using Plots
gr()
ENV["GKSwstype"] = "100"

# =============================================================================
# CONFIG
# =============================================================================
# Generalization test #1: does rbm_paired's *generative* distribution cover
# the same region of sequence space as the real data, or has it collapsed
# onto a narrower subset? The paralog-pairing test only checks free-energy
# *ranking* among real candidates; this instead asks the model to actually
# produce sequences and compares their spread to the real data's, in the
# same PCA coordinates pca_species_pfam.jl already established.
#
# Two things are checked together, since either alone is misleading:
#   1. Coverage -- do generated samples span the same PCA region as real
#      data, or are they collapsed into a small sub-region (mode collapse)?
#   2. Novelty -- a model could "cover" the space by literally resampling a
#      diverse subset of training sequences without generalizing at all, so
#      generated samples are also checked for near-duplication against the
#      training set (nearest-neighbor identity).
#
# Long-run Gibbs sampling budget (300 steps x stride 100 = 30,000 sweeps)
# reuses pair_results_pfam.jl's established protocol -- that script's own
# diagnostics found *longer* sampling actively degrades quality for a model
# trained with a short CD_STEPS, so this is deliberately not pushed further.
const FASTA_PATH  = length(ARGS) >= 2 ? ARGS[2] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 111
# SEED lets the same model be resampled independently -- a fresh Gibbs
# trajectory and a fresh choice of PCA-fit/background subsample -- to check
# whether a coverage finding (e.g. mode collapse) is a real model property or
# an artifact of one unlucky/lucky sampling run.
const SEED        = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 42
Random.seed!(SEED)
const TRAIN_FRAC  = 0.7
const VAL_FRAC    = 0.15
const DEDUP_IDENTITY_THRESHOLD = 0.97   # must match train_potts_pfam.jl, to load its split cache
const OUTPUT_DIR  = "./results_pfam"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

const PATH_RBM_PAIRED = ARGS[1]
const DATASET_TAG = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
run_tag   = replace(basename(PATH_RBM_PAIRED), r"^rbm_paired_" => "", r"\.hdf5$" => "")
seed_tag  = SEED == 42 ? "" : "_SEED=$(SEED)"
FIGPATH   = joinpath(OUTPUT_DIR, "generation_coverage_$(run_tag)$(seed_tag).png")
REPORTPATH = joinpath(OUTPUT_DIR, "generation_coverage_$(run_tag)$(seed_tag).txt")

const N_FIT        = 15000   # random subsample used to fit the PCA directions
const N_BACKGROUND = 6000    # random real-data subsample shown for visual density parity
const N_SAMPLES    = 5000    # fantasy particles drawn from the model
const N_GIBBS_STEPS = 300
const GIBBS_STRIDE  = 100
const IDENTITY_THRESHOLD = 0.95  # "near-duplicate of a training sequence" cutoff

println("rbm_paired : $PATH_RBM_PAIRED")
println("dataset    : $FASTA_PATH (split at $SPLIT_SITE)")
println("figure     -> $FIGPATH")
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
println("q=$q  n_sites=$n_sites  n_samples=$n_samples")

# train_potts_pfam.jl now uses a species/duplicate/near-duplicate-grouped
# split (grouped_train_val_split) and caches the result -- loading that cache
# directly (rather than reimplementing the grouping logic here) guarantees
# "training sequences" below matches what rbm_paired actually trained on.
fasta_tag = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
split_cache_path = joinpath(OUTPUT_DIR, "split_cache_$(fasta_tag)_TRAIN=$(TRAIN_FRAC)_VAL=$(VAL_FRAC)_DEDUP=$(DEDUP_IDENTITY_THRESHOLD).hdf5")
isfile(split_cache_path) || error("No split cache at $split_cache_path -- run train_potts_pfam.jl on this FASTA_PATH first (it creates this cache).")
train_idx = h5read(split_cache_path, "train_idx")
println("loaded train split ← $split_cache_path ($(length(train_idx)) train sequences)")

# =============================================================================
# PCA (fit on a random subsample, exactly mirroring pca_species_pfam.jl so
# real vs. generated live in the same coordinate system)
# =============================================================================
function onehot_flat(seqs_onehot, idx)
    q, n_sites, _ = size(seqs_onehot)
    sub = seqs_onehot[:, :, idx]
    return permutedims(Float32.(reshape(sub, q * n_sites, length(idx))))
end

n_fit   = min(N_FIT, n_samples)
fit_idx = randperm(n_samples)[1:n_fit]
Xfit    = onehot_flat(seqs_onehot, fit_idx)
mu      = mean(Xfit; dims=1)
F       = svd(Xfit .- mu)
V2      = F.V[:, 1:2]
var_explained = F.S .^ 2 ./ sum(F.S .^ 2)
println("PC1 explains $(round(100*var_explained[1], digits=1))%, PC2 explains $(round(100*var_explained[2], digits=1))%")

project(idx) = (onehot_flat(seqs_onehot, idx) .- mu) * V2

background_idx = length(1:n_samples) > N_BACKGROUND ?
    randperm(n_samples)[1:N_BACKGROUND] : collect(1:n_samples)
real_scores = project(background_idx)

# =============================================================================
# DRAW GENERATIVE SAMPLES (long-run equilibrated Gibbs chain)
# =============================================================================
rbm = gpu(load_rbm(PATH_RBM_PAIRED))
x = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., N_SAMPLES))
println("\n--- equilibrating $N_SAMPLES fantasy particles ($N_GIBBS_STEPS x $GIBBS_STRIDE = $(N_GIBBS_STEPS*GIBBS_STRIDE) sweeps) ---")
for t in 1:N_GIBBS_STEPS
    global x = sample_v_from_v(rbm, x; steps=GIBBS_STRIDE)
    if iszero(t % 50)
        println("  sweep $(t*GIBBS_STRIDE)/$(N_GIBBS_STEPS*GIBBS_STRIDE)")
    end
end
x_cpu = Array(cpu(x))   # (q, n_sites, N_SAMPLES) Bool/Float array

gen_flat = permutedims(Float32.(reshape(x_cpu, q * n_sites, N_SAMPLES)))
gen_scores = (gen_flat .- mu) * V2

# =============================================================================
# QUANTITATIVE COVERAGE SUMMARY
# =============================================================================
real_std  = vec(std(real_scores; dims=1))
gen_std   = vec(std(gen_scores; dims=1))
real_rng  = [extrema(real_scores[:, k]) for k in 1:2]
gen_rng   = [extrema(gen_scores[:, k]) for k in 1:2]

println("\nPC1 std:  real=$(round(real_std[1],digits=3))  generated=$(round(gen_std[1],digits=3))  ratio=$(round(gen_std[1]/real_std[1],digits=3))")
println("PC2 std:  real=$(round(real_std[2],digits=3))  generated=$(round(gen_std[2],digits=3))  ratio=$(round(gen_std[2]/real_std[2],digits=3))")
println("PC1 range: real=$(round.(real_rng[1],digits=2))  generated=$(round.(gen_rng[1],digits=2))")
println("PC2 range: real=$(round.(real_rng[2],digits=2))  generated=$(round.(gen_rng[2],digits=2))")

# Simple coverage metric: for each real (background) point, is there a
# generated sample within its k-th nearest-real-neighbor distance? A cheap
# proxy for "does the generative cloud reach into this part of real space,"
# without a full precision/recall-for-generative-models machinery.
function nn_dist(query, reference)
    d2 = sum(abs2, query; dims=2) .+ sum(abs2, reference; dims=2)' .- 2 .* (query * reference')
    return vec(sqrt.(max.(d2, 0)))
end
# Typical real-to-real nearest-neighbor spacing, as the distance scale.
real_sub = real_scores[randperm(size(real_scores,1))[1:min(1000,end)], :]
d_real_real = [minimum(nn_dist(real_sub[i:i, :], real_sub[setdiff(1:size(real_sub,1), i), :])) for i in 1:size(real_sub,1)]
typical_spacing = median(d_real_real)
d_real_to_gen = [minimum(nn_dist(real_sub[i:i, :], gen_scores)) for i in 1:size(real_sub,1)]
coverage_frac = mean(d_real_to_gen .<= 3 * typical_spacing)
println("\ncoverage: $(round(100*coverage_frac, digits=1))% of a 1000-point real subsample has a generated sample within 3x the typical real-to-real spacing ($(round(typical_spacing,digits=3)))")

# =============================================================================
# NOVELTY / MEMORIZATION CHECK
# =============================================================================
# Nearest-neighbor identity of each generated sample against the *training*
# sequences only (not the whole dataset) -- the question is specifically
# whether the model is regurgitating what it trained on.
Xtrain = onehot_flat(seqs_onehot, train_idx)
gen_onehot_flat = permutedims(Float32.(reshape(x_cpu, q * n_sites, N_SAMPLES)))
# Chunked to bound memory (n_train x N_SAMPLES similarity blocks).
chunk = 500
best_identity = zeros(Float32, N_SAMPLES)
for lo in 1:chunk:N_SAMPLES
    hi = min(lo + chunk - 1, N_SAMPLES)
    S = (gen_onehot_flat[lo:hi, :] * Xtrain') ./ Float32(n_sites)   # (chunk, n_train)
    best_identity[lo:hi] .= vec(maximum(S; dims=2))
end
frac_near_dup = mean(best_identity .>= IDENTITY_THRESHOLD)
println("\nnear-duplicate fraction: $(round(100*frac_near_dup, digits=1))% of generated samples are >= $(Int(100*IDENTITY_THRESHOLD))% identical to some training sequence")
println("mean nearest-training-neighbor identity of generated samples: $(round(mean(best_identity), digits=3))")

# =============================================================================
# PLOT
# =============================================================================
plt = scatter(
    real_scores[:, 1], real_scores[:, 2];
    color = :gray70, markersize = 1.8, markerstrokewidth = 0, alpha = 0.35,
    label = "real data (n=$(length(background_idx)))",
    legend = :outerright, size = (1150, 750), frame = :box,
    left_margin = 6Plots.mm, top_margin = 8Plots.mm, bottom_margin = 6Plots.mm,
    xlabel = "PC1 ($(round(100*var_explained[1], digits=1))%)",
    ylabel = "PC2 ($(round(100*var_explained[2], digits=1))%)",
    title = "Generated vs. real sequences -- $(replace(DATASET_TAG, "_" => "+"))",
)
scatter!(
    plt, gen_scores[:, 1], gen_scores[:, 2];
    color = :crimson, markersize = 2.2, markerstrokewidth = 0, alpha = 0.55,
    label = "generated (n=$N_SAMPLES)",
)
savefig(plt, FIGPATH)

open(REPORTPATH, "w") do f
    println(f, "rbm_paired=$PATH_RBM_PAIRED")
    println(f, "PC1_explained=$(var_explained[1]) PC2_explained=$(var_explained[2])")
    println(f, "PC1_std_real=$(real_std[1]) PC1_std_generated=$(gen_std[1]) ratio=$(gen_std[1]/real_std[1])")
    println(f, "PC2_std_real=$(real_std[2]) PC2_std_generated=$(gen_std[2]) ratio=$(gen_std[2]/real_std[2])")
    println(f, "coverage_frac_within_3x_typical_spacing=$coverage_frac")
    println(f, "frac_generated_near_duplicate_of_training(>=$(IDENTITY_THRESHOLD))=$frac_near_dup")
    println(f, "mean_nearest_training_identity=$(mean(best_identity))")
end
println("\nsaved figure -> $FIGPATH")
println("saved report -> $REPORTPATH")
