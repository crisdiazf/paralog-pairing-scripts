using CUDA
using HDF5
using RestrictedBoltzmannMachines: load_rbm, gpu, cpu
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_h, sample_from_inputs, Falses
using LatentAlignedRBMs
using FASTX
using BioSequences
using Statistics
using Random
using Plots
gr()
ENV["GKSwstype"] = "100"

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
# Per-site profile + concrete example alignments for the A->B clamped
# generation test (see clamped_generation_pfam.jl for the full
# single-sample/majority-vote/consensus/conservation-quartile analysis, which
# this reuses the same methodology from). This adds two views not visible in
# that aggregate report:
#   1. A per-site accuracy profile along the *natural sequence position*
#      (not sorted by conservation, unlike the quartile breakdown) -- does
#      the model's advantage over consensus concentrate in a particular
#      region of the domain, rather than spreading uniformly?
#   2. Concrete example alignments: the actual true vs. majority-vote
#      generated residue at every site, for a best/median/worst-case
#      held-out sequence, so the accuracy numbers are grounded in real output.
const ALPHABET = collect("ACDEFGHIKLMNPQRSTVWY-")   # LatentAlignedRBMs.onehot's fixed category order

const FASTA_PATH  = length(ARGS) >= 2 ? ARGS[2] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 111
const TRAIN_FRAC  = 0.7
const VAL_FRAC    = 0.15
const DEDUP_IDENTITY_THRESHOLD = 0.97   # must match train_potts_pfam.jl, to load its split cache
const OUTPUT_DIR  = "./results_pfam"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

const PATH_RBM_PAIRED = ARGS[1]
run_tag    = replace(basename(PATH_RBM_PAIRED), r"^rbm_paired_" => "", r"\.hdf5$" => "")
FIGPATH    = joinpath(OUTPUT_DIR, "clamped_generation_profile_$(run_tag).png")
REPORTPATH = joinpath(OUTPUT_DIR, "clamped_generation_profile_$(run_tag).txt")

const N_HELDOUT     = 2000
const N_GIBBS_STEPS = 300
const GIBBS_STRIDE  = 100
const N_REPLICAS    = 10

println("rbm_paired : $PATH_RBM_PAIRED")
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
seqs_onehot = LatentAlignedRBMs.onehot(seqs_raw)
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

n_heldout = min(N_HELDOUT, length(val_idx))
heldout_idx = val_idx[randperm(length(val_idx))[1:n_heldout]]
println("held-out sequences used: $n_heldout (of $(length(val_idx)) available)")

XA_heldout = seqs_onehot[:, 1:n_vis_A, heldout_idx]
XB_heldout = seqs_onehot[:, n_vis_A+1:end, heldout_idx]
XB_train   = seqs_onehot[:, n_vis_A+1:end, train_idx]

# =============================================================================
# MODEL + CLAMPED GENERATION (A -> B, majority vote over N_REPLICAS)
# =============================================================================
rbm = gpu(load_rbm(PATH_RBM_PAIRED))

cat_indices(x) = dropdims(map(ci -> ci[1], argmax(Array(x); dims=1)); dims=1)

function consensus_stats(ref_onehot)
    q, n, m = size(ref_onehot)
    freqs = sum(Float32.(ref_onehot); dims=3) ./ m
    idx   = dropdims(map(ci -> ci[1], argmax(freqs; dims=1)); dims=(1, 3))
    conservation = vec(maximum(freqs; dims=1))
    return idx, conservation
end

function mode_over_replicas(idx_array, q)
    n, b, r = size(idx_array)
    counts = zeros(Int32, n, b, q)
    for k in 1:q
        counts[:, :, k] .= dropdims(sum(idx_array .== k; dims=3); dims=3)
    end
    return dropdims(map(ci -> ci[3], argmax(counts; dims=3)); dims=3)
end

range_A = 1:n_vis_A
range_B = (n_vis_A + 1):(n_vis_A + n_vis_B)

gen_truth_idx = cat_indices(XB_heldout)
clamp_tiled = repeat(XA_heldout, 1, 1, N_REPLICAS)
truth_tiled = repeat(gen_truth_idx, 1, N_REPLICAS)

v = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., size(clamp_tiled, 3)))
v[:, range_A, :] .= gpu(clamp_tiled)
println("\n--- equilibrating A -> B clamped chains ($N_GIBBS_STEPS x $GIBBS_STRIDE = $(N_GIBBS_STEPS*GIBBS_STRIDE) sweeps, batch=$(size(clamp_tiled,3))) ---")
for t in 1:N_GIBBS_STEPS
    for _ in 1:GIBBS_STRIDE
        h = sample_h_from_v(rbm, v)
        global v = sample_v_from_h(rbm, h)
        v[:, range_A, :] .= gpu(clamp_tiled)
    end
    if iszero(t % 50)
        acc = mean(cat_indices(v[:, range_B, :]) .== truth_tiled)
        println("  sweep $(t*GIBBS_STRIDE)/$(N_GIBBS_STEPS*GIBBS_STRIDE)  recovery=$(round(acc, digits=4))")
    end
end
gen_idx_tiled = cat_indices(v[:, range_B, :])
mode_idx = mode_over_replicas(reshape(gen_idx_tiled, n_vis_B, n_heldout, N_REPLICAS), q)

consensus_idx, conservation = consensus_stats(XB_train)

# =============================================================================
# PER-SITE PROFILE (natural sequence order, not conservation-sorted)
# =============================================================================
site_true_mode = vec(mean(mode_idx .== gen_truth_idx; dims=2))
site_consensus = vec(mean(repeat(consensus_idx, 1, n_heldout) .== gen_truth_idx; dims=2))
shift = circshift(1:n_heldout, 1)
site_real_real = vec(mean(gen_truth_idx .== gen_truth_idx[:, shift]; dims=2))

# =============================================================================
# EXAMPLE SEQUENCES (best / median / worst per-sequence accuracy)
# =============================================================================
seq_acc  = vec(mean(mode_idx .== gen_truth_idx; dims=1))
best_i   = argmax(seq_acc)
worst_i  = argmin(seq_acc)
median_i = sortperm(seq_acc)[cld(length(seq_acc), 2)]
examples = [("Best", best_i), ("Median", median_i), ("Worst", worst_i)]
for (label, i) in examples
    println("$label example: heldout_idx=$(heldout_idx[i])  accuracy=$(round(seq_acc[i], digits=3))")
end

# =============================================================================
# PLOT
# =============================================================================
p1 = plot(
    1:n_vis_B, 100 .* site_true_mode;
    label = "true-partner (majority vote)", color = :crimson, lw = 2,
    ylabel = "Accuracy (%)", ylim = (0, 100), legend = :outerright,
    title = "Per-site profile, A -> B (n=$n_heldout held-out sequences)",
    size = (1600, 900), left_margin = 10Plots.mm,
)
plot!(p1, 1:n_vis_B, 100 .* site_consensus; label = "consensus", color = :gray40, lw = 2)
plot!(p1, 1:n_vis_B, 100 .* site_real_real; label = "real-to-real", color = :gray80, lw = 1, ls = :dash)

row_labels = String[]
n_rows = 2 * length(examples)
color_matrix  = zeros(Int, n_rows, n_vis_B)
letter_matrix = Matrix{Char}(undef, n_rows, n_vis_B)
for (k, (label, i)) in enumerate(examples)
    true_row, gen_row = 2k - 1, 2k
    push!(row_labels, "$label truth"); push!(row_labels, "$label gen.")
    for s in 1:n_vis_B
        true_cat, gen_cat = gen_truth_idx[s, i], mode_idx[s, i]
        letter_matrix[true_row, s] = ALPHABET[true_cat]
        letter_matrix[gen_row, s]  = ALPHABET[gen_cat]
        color_matrix[true_row, s]  = 0                          # neutral (truth row)
        color_matrix[gen_row, s]   = gen_cat == true_cat ? 1 : 2 # match / mismatch
    end
end

p2 = heatmap(
    1:n_vis_B, 1:n_rows, color_matrix;
    color = cgrad([:gray85, :mediumseagreen, :indianred], 3, categorical = true),
    colorbar = false, yflip = true,
    yticks = (1:n_rows, row_labels),
    xlabel = "Site position (B domain)",
    legend = false, size = (1600, 900), left_margin = 10Plots.mm,
)
for r in 1:n_rows, s in 1:n_vis_B
    annotate!(p2, s, r, text(string(letter_matrix[r, s]), 6, :black))
end

plt = plot(p1, p2; layout = @layout([a; b{0.6h}]), size = (1600, 1100))
savefig(plt, FIGPATH)
println("\nsaved figure -> $FIGPATH")

open(REPORTPATH, "w") do f
    println(f, "rbm_paired=$PATH_RBM_PAIRED")
    println(f, "n_heldout=$n_heldout  n_replicas=$N_REPLICAS")
    println(f, "site_true_mode=$site_true_mode")
    println(f, "site_consensus=$site_consensus")
    println(f, "site_real_to_real=$site_real_real")
    for (label, i) in examples
        println(f, "$label: heldout_idx=$(heldout_idx[i]) accuracy=$(seq_acc[i])")
        println(f, "  true:      $(join(ALPHABET[gen_truth_idx[:, i]]))")
        println(f, "  generated: $(join(ALPHABET[mode_idx[:, i]]))")
    end
end
println("saved report -> $REPORTPATH")
