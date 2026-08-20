using FASTX
using BioSequences
using LatentAlignedRBMs
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
# Reanalysis only -- no retraining, no rerun of the GPU pairing test. Applies
# DCA-style sequence reweighting ("effective sequence count", M_eff) to the
# *existing* paralog_pairing_pfam.jl results, to check whether the headline
# accuracy numbers are skewed by clusters of near-identical sequences, the
# same underlying concern as the species-count skew already handled by
# dropping 9xxxx/UNCxx placeholder codes -- except this time the redundancy
# being corrected can occur *between* formally distinct species (e.g.
# several close relatives all carrying near-identical domains), which the
# species-code filter can't catch.
#
# Weighting scheme: for each test sequence i, n_i = number of test sequences
# (including itself) at or above IDENTITY_THRESHOLD fractional identity to
# it; weight w_i = 1/n_i, so a cluster of m near-identical sequences
# contributes total weight 1 instead of m. M_eff = sum(w_i) is the
# redundancy-corrected effective sample size. Restricted to the ~15k
# sequences that actually feed the pairing test (not the full 77,880-record
# dataset) since that's what the reweighted metric aggregates over; a global
# M_eff over the whole dataset would matter for reweighting *training*
# itself, a separate, bigger step not attempted here.
const REPORT_PATH = length(ARGS) >= 1 ? ARGS[1] :
    "./results_pfam/paralog_pairing_N_HIDDEN_A=150_N_HIDDEN_B=100_H_ADD=20_K=50_N_ITERS=10000_REG=0_BS=256_LR=0.0001_DATA=PF00072_PF00512_l111_PAIRED_ITERS=5000.txt"
const FASTA_PATH  = length(ARGS) >= 2 ? ARGS[2] : "./PF00072_PF00512_paired.fasta"
const IDENTITY_THRESHOLD = 0.8   # standard DCA default

# =============================================================================
# SPECIES GROUPING (mirrors paralog_pairing_pfam.jl exactly, so weights line
# up with the correct/k counts already computed there)
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

groups = build_species_groups(ids)
species_codes   = String[]
species_indices = Vector{Vector{Int}}()
for (code, idxs) in groups
    kept = dedup_group(idxs, seqs_raw)
    length(kept) >= 2 || continue
    push!(species_codes, code)
    push!(species_indices, kept)
end

all_test_idx = sort(unique(vcat(species_indices...)))
n_test = length(all_test_idx)
println("test sequences: $n_test across $(length(species_codes)) species")

# =============================================================================
# PAIRWISE IDENTITY + WEIGHTS
# =============================================================================
# Fractional identity between one-hot rows i,j = (dot product)/n_sites, since
# a match at a site contributes exactly 1 to the dot product (same category
# in both) and 0 otherwise -- this is a single matrix multiply, not an
# explicit O(n_test^2) loop.
X = permutedims(Float32.(reshape(seqs_onehot[:, :, all_test_idx], q * n_sites, n_test)))  # n_test x features
S = (X * X') ./ Float32(n_sites)          # n_test x n_test identity matrix
n_i = vec(sum(S .>= Float32(IDENTITY_THRESHOLD); dims = 2))
w = 1.0 ./ n_i
M_eff = sum(w)
println("M_eff = $(round(M_eff, digits=1)) out of $n_test raw test sequences (identity threshold=$IDENTITY_THRESHOLD)")
println("  i.e. redundancy-corrected sample size is $(round(100*M_eff/n_test, digits=1))% of the raw count")

idx_to_w = Dict(all_test_idx[i] => w[i] for i in 1:n_test)

# =============================================================================
# PER-SPECIES EFFECTIVE WEIGHT
# =============================================================================
species_weight = Dict{String, Float64}()
for (code, idxs) in zip(species_codes, species_indices)
    species_weight[code] = sum(idx_to_w[i] for i in idxs)
end

# =============================================================================
# PARSE EXISTING PAIRING-TEST REPORT (code, k, correct)
# =============================================================================
function parse_report(path)
    rows = NamedTuple[]
    for line in eachline(path)
        parts = split(line)
        length(parts) == 8 || continue
        code = parts[1]
        k = tryparse(Int, parts[2])
        isnothing(k) && continue
        correct = parse(Int, parts[3])
        push!(rows, (; code, k, correct))
    end
    return rows
end

rows = parse_report(REPORT_PATH)
println("parsed $(length(rows)) species rows from $REPORT_PATH")

# =============================================================================
# POOLED vs. UNWEIGHTED vs. Meff-WEIGHTED ACCURACY
# =============================================================================
pooled_acc = sum(r.correct for r in rows) / sum(r.k for r in rows)
mean_acc   = mean(r.correct / r.k for r in rows)

total_w      = sum(species_weight[r.code] for r in rows)
weighted_acc = sum(species_weight[r.code] * (r.correct / r.k) for r in rows) / total_w

println()
println("pooled accuracy (weighted by pair count)  : $(round(100*pooled_acc, digits=1))%")
println("mean per-species accuracy (species-equal) : $(round(100*mean_acc, digits=1))%")
println("Meff-weighted per-species accuracy         : $(round(100*weighted_acc, digits=1))%")

# =============================================================================
# MOST / LEAST REDUNDANT SPECIES (avg effective weight per paralog)
# =============================================================================
per_paralog_weight = [(r.code, species_weight[r.code] / r.k) for r in rows]
sort!(per_paralog_weight; by = x -> x[2])

println("\nMost redundant test species (lowest avg. weight per paralog -- clusters of near-duplicates):")
for (code, wv) in per_paralog_weight[1:min(10, end)]
    println("  $code : avg_weight_per_paralog=$(round(wv, digits=3))")
end
println("\nLeast redundant test species (avg. weight per paralog ~= 1, i.e. phylogenetically distinct):")
for (code, wv) in per_paralog_weight[max(1, end-9):end]
    println("  $code : avg_weight_per_paralog=$(round(wv, digits=3))")
end
