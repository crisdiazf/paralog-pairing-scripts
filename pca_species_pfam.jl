using FASTX
using BioSequences
using LatentAlignedRBMs
using LinearAlgebra
using Statistics
using Random
using Plots
gr()
ENV["GKSwstype"] = "100"

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
# Preliminary look at the raw data itself: a PCA of the one-hot encoded
# PF00072+PF00512 paired sequences, with the species used in
# paralog_pairing_pfam.jl's true-pairs test highlighted in color against the
# rest of the dataset in gray. Purpose: sanity-check that "species" (the
# UniProt mnemonic suffix, after dropping 9xxxx/UNCxx placeholder codes) is a
# meaningful grouping in sequence space -- paralogs of one species should sit
# closer to each other than to the bulk of the data if the grouping reflects
# real phylogenetic/compositional structure, which is the implicit
# assumption behind treating within-species free-energy comparisons as
# meaningful in the pairing test.
#
# Uses the exact same species-grouping logic (real mnemonics only, dedup,
# k>=2) as paralog_pairing_pfam.jl, duplicated here rather than shared via
# a module so this stays a single self-contained script.
const FASTA_PATH = length(ARGS) >= 1 ? ARGS[1] : "./PF00072_PF00512_paired.fasta"
const OUTPUT_DIR = "./results_pfam"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)
const DATASET_TAG = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
const FIGPATH = joinpath(OUTPUT_DIR, "pca_species_$(DATASET_TAG).png")

const N_TOP_SPECIES  = 10     # individually colored/labeled; all other qualifying
                              # species are pooled into one "other test species" color
const N_FIT          = 15000  # random subsample used to fit the PCA directions
const N_BACKGROUND   = 3500   # random subsample of excluded points, for legibility
const N_OTHER_DISPLAY = 3500  # random subsample of "other test species" points

# =============================================================================
# DATA + SPECIES GROUPING (mirrors paralog_pairing_pfam.jl)
# =============================================================================
function load_records(path)
    reader = open(FASTA.Reader, path)
    ids  = String[]
    seqs = LongAA[]
    for record in reader
        push!(ids, FASTA.identifier(record))
        push!(seqs, LongAA(FASTA.sequence(record)))
    end
    close(reader)
    return ids, seqs
end

species_code(id) = String(split(split(id, "/")[1], "_")[end])
is_placeholder_code(code) = isdigit(first(code)) || startswith(code, "UNC")

function build_species_groups(ids)
    groups = Dict{String, Vector{Int}}()
    for (i, id) in enumerate(ids)
        code = species_code(id)
        is_placeholder_code(code) && continue
        push!(get!(groups, code, Int[]), i)
    end
    return groups
end

function dedup_group(indices, seqs_raw)
    seen = Set{eltype(seqs_raw)}()
    kept = Int[]
    for i in indices
        s = seqs_raw[i]
        if !(s in seen)
            push!(seen, s)
            push!(kept, i)
        end
    end
    return kept
end

ids, seqs_raw = load_records(FASTA_PATH)
seqs_onehot   = LatentAlignedRBMs.onehot(seqs_raw)   # (q, n_sites, n_samples) BitArray
q, n_sites, n_samples = size(seqs_onehot)
println("q=$q  n_sites=$n_sites  n_samples=$n_samples")

groups = build_species_groups(ids)
species_codes   = String[]
species_indices = Vector{Vector{Int}}()
for (code, idxs) in groups
    kept = dedup_group(idxs, seqs_raw)
    length(kept) >= 2 || continue
    push!(species_codes, code)
    push!(species_indices, kept)
end
println("usable species (k>=2 after dedup, placeholder codes excluded): $(length(species_codes))")

order      = sortperm(length.(species_indices); rev=true)
top_order  = order[1:min(N_TOP_SPECIES, length(order))]
top_codes  = Set(species_codes[top_order])

labels = fill("excluded", n_samples)
for (code, idxs) in zip(species_codes, species_indices)
    lbl = code in top_codes ? code : "other test species"
    for i in idxs
        labels[i] = lbl
    end
end

# =============================================================================
# PCA (fit on a random subsample, project only the points we plot)
# =============================================================================
function onehot_flat(seqs_onehot, idx)
    q, n_sites, _ = size(seqs_onehot)
    sub = seqs_onehot[:, :, idx]
    return permutedims(Float32.(reshape(sub, q * n_sites, length(idx))))  # (length(idx), features)
end

n_fit   = min(N_FIT, n_samples)
fit_idx = randperm(n_samples)[1:n_fit]
Xfit    = onehot_flat(seqs_onehot, fit_idx)
mu      = mean(Xfit; dims=1)
F       = svd(Xfit .- mu)
V2      = F.V[:, 1:2]
var_explained = F.S .^ 2 ./ sum(F.S .^ 2)
println("PC1 explains $(round(100*var_explained[1], digits=1))%, PC2 explains $(round(100*var_explained[2], digits=1))% (fit on a $(n_fit)-sequence random subsample)")

project(idx) = (onehot_flat(seqs_onehot, idx) .- mu) * V2

subsample(idx, n) = length(idx) > n ? idx[randperm(length(idx))[1:n]] : idx

excluded_idx_all = findall(==("excluded"), labels)
other_idx_all    = findall(==("other test species"), labels)
top_idx_all      = findall(l -> l != "excluded" && l != "other test species", labels)

background_idx = subsample(excluded_idx_all, N_BACKGROUND)
other_idx      = subsample(other_idx_all, N_OTHER_DISPLAY)
plot_idx       = vcat(background_idx, other_idx, top_idx_all)

scores = project(plot_idx)
pc1, pc2 = scores[:, 1], scores[:, 2]
plot_labels = labels[plot_idx]

# =============================================================================
# PLOT
# =============================================================================
# Explicit per-group styling and draw order (background first, individually
# highlighted species last/on top) -- Plots' automatic `group=` coloring picks
# palette colors independent of category size, so "excluded" (the vast
# majority of points) can end up in a loud color that visually buries the
# individually highlighted species; drawing in this order with muted
# background colors and vivid, opaque top-species colors avoids that.
top_codes_ordered = species_codes[top_order]  # preserves k-descending order
vivid_colors(n) = [convert(RGB, HSV(360 * (i - 1) / n, 0.85, 0.9)) for i in 1:n]
top_colors = vivid_colors(length(top_codes_ordered))
other_ks   = [length(idx) for (code, idx) in zip(species_codes, species_indices) if !(code in top_codes)]

is_excluded = plot_labels .== "excluded"
is_other    = plot_labels .== "other test species"

plt = scatter(
    pc1[is_excluded], pc2[is_excluded];
    color = :gray88, markersize = 1.3, markerstrokewidth = 0, alpha = 0.25,
    label = "excluded (placeholder code / k<2)",
    legend = :outerright,
    legendfontsize = 8,
    size = (1300, 850),
    frame = :box,
    left_margin = 6Plots.mm,
    top_margin = 8Plots.mm,
    bottom_margin = 6Plots.mm,
    titlefontsize = 12,
    xlabel = "PC1 ($(round(100*var_explained[1], digits=1))%)",
    ylabel = "PC2 ($(round(100*var_explained[2], digits=1))%)",
    title = "PCA of paired $(replace(DATASET_TAG, "_" => "+")) sequences -- colored by species used in the pairing test",
)
scatter!(
    plt, pc1[is_other], pc2[is_other];
    color = :gray45, markersize = 1.8, markerstrokewidth = 0, alpha = 0.35,
    label = "other test species (k=$(minimum(other_ks))-$(maximum(other_ks)))",
)
for (ci, code) in enumerate(top_codes_ordered)
    mask = plot_labels .== code
    k_this = length(species_indices[findfirst(==(code), species_codes)])
    scatter!(
        plt, pc1[mask], pc2[mask];
        color = top_colors[ci], markersize = 5.5, markerstrokewidth = 0.4,
        markerstrokecolor = :black, alpha = 1.0, label = "$code (k=$k_this)",
    )
end
savefig(plt, FIGPATH)
println("saved -> $FIGPATH")
