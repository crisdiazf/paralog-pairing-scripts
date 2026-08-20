using CUDA
using HDF5
using RestrictedBoltzmannMachines: load_rbm, free_energy, gpu, cpu
using LatentAlignedRBMs
using FASTX
using BioSequences
using Hungarian
using Random
using Statistics: mean

# =============================================================================
# CONFIG
# =============================================================================
# First real test of the paired-RBM architecture: paralog assignment.
# Many records in PF00072_PF00512_paired.fasta come from species with more
# than one paralogous copy of the fused A+B protein. For a species with k
# paralogs we know the k *true* A<->B pairings (each fasta record already is
# a correct pair, since both halves were sliced from the same accession --
# see train_potts_pfam.jl's DATA section). This script hides that pairing,
# scores every candidate (A_i, B_j) combination with rbm_paired's free
# energy, and asks the Hungarian algorithm to recover the best one-to-one
# assignment, then checks it against the true diagonal.
#
# Why free_energy alone is a valid, sufficient score: rbm_paired's cross
# blocks (hidden_A<->vis_B and hidden_B<->vis_A) are frozen at exactly zero
# (see project_to_frozen_par! in train_potts_pfam.jl), so free energy
# decomposes exactly as F(v_A,v_B) = f_A(v_A) + f_B(v_B) + f_added(v_A,v_B).
# Assignment problems are invariant to per-row/per-column additive
# constants, so the optimal permutation depends *only* on f_added -- i.e.
# this test measures exactly what the H_ADD units learned, nothing else.
#
# Species handle: only the UniProt mnemonic suffix is available in the
# header (e.g. ".../A0A010R279_9PEZI/1641-1768"). Codes with a leading digit
# ("9xxxx") are UniProt placeholders for organisms without a dedicated
# mnemonic and lump together many unrelated species (one code alone, 9BACT,
# covers 10,862 of this dataset's 77,880 records) -- those are dropped
# rather than treated as a single "species". The remaining ~20% of records
# carry real per-species mnemonics and give a clean population: about 1,349
# species with >=2 paralogs, up to k~183, ~400k total pairwise evaluations
# -- no arbitrary paralog-count cap needed.
#
# train_potts_pfam.jl's train/val split is per-species (grouped), so no
# species tested here ever straddles train and val -- but a species entirely
# inside train_idx is NOT necessarily one rbm_paired's cross-family H_ADD
# units actually learned a correct pairing from: when PAIRED_TRAIN_N/
# PAIRED_TRAIN_FRAC < the full training split, the paired model itself only
# trains on paired_train_idx, a subsample of train_idx (see train_potts_pfam.jl's
# DATA section) -- RBM A/B still see the whole train_idx (their own single-family
# fit doesn't need pairing information), but the H_ADD units, the only part of
# the model this test can actually probe, don't. So membership here is reported
# two ways: train_idx/val_idx (:train/:val/:unused, the RBM A/B split -- kept
# for backward compatibility) AND paired_train_idx-based (:paired_train/
# :paired_other), the latter being the one that actually determines whether a
# species is a fair generalization test of what H_ADD learned. Reproducing
# paired_train_idx exactly requires replicating train_potts_pfam.jl's RNG
# state byte-for-byte: seed with the same SEED, then draw immediately after
# loading the (cached) split, before anything else touches the RNG stream.
const TRAIN_FRAC = 0.7
const VAL_FRAC   = 0.15
const DEDUP_IDENTITY_THRESHOLD = 0.97   # must match train_potts_pfam.jl, to load its split cache
const OUTPUT_DIR = "./results_pfam"

const PATH_RBM_PAIRED = ARGS[1]
# FASTA_PATH/SPLIT_SITE default to the PF00072/PF00512 dataset for backward
# compatibility, but any similarly-formatted "two domains concatenated"
# dataset works (e.g. PF00072/PF01339) by passing it as extra arguments --
# same convention as train_potts_pfam.jl's own FASTA_PATH/SPLIT_SITE args.
const FASTA_PATH = length(ARGS) >= 4 ? ARGS[4] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 111
# Must match the PAIRED_TRAIN_FRAC/PAIRED_TRAIN_N/SEED that rbm_paired was
# actually trained with (see train_potts_pfam.jl) -- these don't affect which
# checkpoint file gets loaded (that's already fixed by PATH_RBM_PAIRED), only
# which species this script can fairly call "seen by the paired model".
const PAIRED_TRAIN_FRAC = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1.0
const PAIRED_TRAIN_N    = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 0
const SEED               = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 42
Random.seed!(SEED)

# rbm_A/rbm_B were saved as "rbm_A_<single_suffix>.hdf5" /
# "rbm_B_<single_suffix>.hdf5", and rbm_paired as
# "rbm_paired_<single_suffix>_PAIRED_ITERS=<n>.hdf5" (see train_potts_pfam.jl).
# Deriving the two single-model paths from the paired path avoids having to
# re-supply N_ITERS/H_ADD/CD_STEPS/etc. by hand -- which would be fragile now
# that CD_STEPS is being swept (K=30/50/100 all reconstruct that suffix
# differently).
function derive_single_model_path(paired_path, tag)
    dir  = dirname(paired_path)
    base = basename(paired_path)
    m = match(r"^rbm_paired_(.*)_PAIRED_ITERS=\d+\.hdf5$", base)
    isnothing(m) && error("Could not derive rbm_$(tag) path from $paired_path -- pass it explicitly as an extra argument.")
    # H_ADD (and, when N_ITERS_B != N_ITERS, a "_NITERSB=..." tag) appear in
    # paired_suffix() but deliberately NOT in single_suffix_A()/B() -- A/B
    # training doesn't depend on H_ADD (see train_potts_pfam.jl) -- so strip
    # them if present rather than assuming one fixed naming convention.
    middle = replace(m.captures[1], r"_H_ADD=\d+" => "", r"_NITERSB=\d+" => "")
    candidate = joinpath(dir, "rbm_$(tag)_$(middle).hdf5")
    isfile(candidate) || error("Derived rbm_$(tag) path $candidate does not exist -- pass the correct path explicitly as an extra argument.")
    return candidate
end

const PATH_RBM_A = length(ARGS) >= 2 ? ARGS[2] : derive_single_model_path(PATH_RBM_PAIRED, "A")
const PATH_RBM_B = length(ARGS) >= 3 ? ARGS[3] : derive_single_model_path(PATH_RBM_PAIRED, "B")

isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)
run_tag = replace(basename(PATH_RBM_PAIRED), r"^rbm_paired_" => "", r"\.hdf5$" => "")
path_report = joinpath(OUTPUT_DIR, "paralog_pairing_$(run_tag).txt")

println("rbm_paired : $PATH_RBM_PAIRED")
println("rbm_A      : $PATH_RBM_A")
println("rbm_B      : $PATH_RBM_B")
println("report     -> $path_report")

# =============================================================================
# LOAD MODELS
# =============================================================================
rbm_paired = gpu(load_rbm(PATH_RBM_PAIRED))
rbm_A      = gpu(load_rbm(PATH_RBM_A))
rbm_B      = gpu(load_rbm(PATH_RBM_B))

# =============================================================================
# DATA
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

ids, seqs_raw = load_records(FASTA_PATH)
seqs_onehot   = LatentAlignedRBMs.onehot(seqs_raw)  # (q, n_sites_total, n_samples) BitArray
n_samples     = size(seqs_onehot, 3)
XA = seqs_onehot[:, 1:SPLIT_SITE, :]
XB = seqs_onehot[:, SPLIT_SITE+1:end, :]

# train_potts_pfam.jl now uses a species/duplicate/near-duplicate-grouped
# split (grouped_train_val_split), not a per-sequence randperm, and caches
# the result -- loading that cache directly (rather than reimplementing the
# grouping logic here) guarantees the train/val membership below matches
# what rbm_paired actually saw, byte for byte. Note TRAIN_FRAC+VAL_FRAC < 1,
# so a third bucket ("unused") exists too.
fasta_tag = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
split_cache_path = joinpath(OUTPUT_DIR, "split_cache_$(fasta_tag)_TRAIN=$(TRAIN_FRAC)_VAL=$(VAL_FRAC)_DEDUP=$(DEDUP_IDENTITY_THRESHOLD).hdf5")
isfile(split_cache_path) || error("No split cache at $split_cache_path -- run train_potts_pfam.jl on this FASTA_PATH first (it creates this cache), or check TRAIN_FRAC/VAL_FRAC/DEDUP_IDENTITY_THRESHOLD match.")
# Keep train_idx as the raw Vector (HDF5-stored order) until after the
# paired_train_idx draw below -- randperm(length(train_idx))[1:n] indexes
# into this specific order, so wrapping it in a Set first (order-losing)
# would silently break reproducibility against train_potts_pfam.jl.
train_idx_vec = h5read(split_cache_path, "train_idx")
val_idx_vec   = h5read(split_cache_path, "val_idx")
train_idx = Set(train_idx_vec)
val_idx   = Set(val_idx_vec)
membership(i) = i in train_idx ? :train : (i in val_idx ? :val : :unused)
println("loaded train/val split ← $split_cache_path ($(length(train_idx)) train, $(length(val_idx)) val)")

# paired_train_idx: replicate train_potts_pfam.jl's exact RNG draw -- same
# SEED (seeded at the top of this file), drawn immediately after the split
# load with nothing else touching the RNG stream in between, exactly
# mirroring train_potts_pfam.jl's own DATA section. This is the actual
# training set the paired model's H_ADD units saw; train_idx above is only
# what RBM A/B saw (see the CONFIG section comment).
n_paired_train = PAIRED_TRAIN_N > 0 ? min(PAIRED_TRAIN_N, length(train_idx_vec)) :
    floor(Int, PAIRED_TRAIN_FRAC * length(train_idx_vec))
paired_train_idx = (PAIRED_TRAIN_N == 0 && PAIRED_TRAIN_FRAC == 1.0) ? train_idx_vec :
    train_idx_vec[randperm(length(train_idx_vec))[1:n_paired_train]]
paired_train_idx_set = Set(paired_train_idx)
paired_membership(i) = i in paired_train_idx_set ? :paired_train : :paired_other
println("Paired training subset: $(length(paired_train_idx)) / $(length(train_idx_vec)) train sequences " *
        (PAIRED_TRAIN_N > 0 ? "(PAIRED_TRAIN_N=$(PAIRED_TRAIN_N))" : "(PAIRED_TRAIN_FRAC=$(PAIRED_TRAIN_FRAC))"))

println("XA: $(size(XA))  XB: $(size(XB))  n_samples=$n_samples")

# =============================================================================
# SPECIES GROUPING
# =============================================================================
# Matches train_potts_pfam.jl's species_code exactly: handles both header
# forms seen across this pipeline's FASTA files -- bare
# "A0A010YSD8_9BACL/6-116" and UniProt "tr|A0A355I9E3|A0A355I9E3_UNCFI/8-112"
# (splitting on "|" and taking the last field is a no-op for the bare form).
function species_code(id::AbstractString)
    id_part = split(id, "|")[end]
    id_part = split(id_part, "/")[1]
    parts = split(id_part, "_")
    return length(parts) >= 2 ? String(parts[end]) : String(id_part)
end
# "9xxxx" codes are UniProt placeholders for organisms without a dedicated
# mnemonic; "UNCxx"/"UNKxx" codes are its placeholders for uncultured/unknown
# organisms -- all lump many unrelated species under one code rather than
# identifying a single one, so all are excluded from species grouping (same
# is_ambiguous_species criteria as train_potts_pfam.jl).
is_placeholder_code(code) = occursin(r"^[0-9]", code) || occursin("UNC", code) || occursin("UNK", code)

function build_species_groups(ids)
    groups = Dict{String, Vector{Int}}()
    n_dropped_placeholder = 0
    for (i, id) in enumerate(ids)
        code = species_code(id)
        if is_placeholder_code(code)
            n_dropped_placeholder += 1
            continue
        end
        push!(get!(groups, code, Int[]), i)
    end
    return groups, n_dropped_placeholder
end

# Exact-duplicate paralogs (identical full A+B sequence) tie every column of
# their row, making Hungarian's pick between them arbitrary rather than a
# real test of the model -- drop repeats, keeping the first occurrence.
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

groups, n_dropped_placeholder = build_species_groups(ids)
println("species groups (real mnemonics only): $(length(groups))  (dropped $n_dropped_placeholder records under placeholder codes)")

species_codes   = String[]
species_indices = Vector{Vector{Int}}()
n_dropped_dedup = 0
for (code, idxs) in groups
    kept = dedup_group(idxs, seqs_raw)
    global n_dropped_dedup += length(idxs) - length(kept)
    length(kept) >= 2 || continue
    push!(species_codes, code)
    push!(species_indices, kept)
end
println("usable species (k>=2 after dedup): $(length(species_codes))  (dropped $n_dropped_dedup exact-duplicate records)")

# =============================================================================
# PAIRING TEST
# =============================================================================
function candidate_batch(XA, XB, indices)
    k  = length(indices)
    q  = size(XA, 1)
    nA = size(XA, 2)
    nB = size(XB, 2)
    batch = falses(q, nA + nB, k * k)
    idx = 1
    for i in indices, j in indices
        batch[:, 1:nA, idx]     .= @view XA[:, :, i]
        batch[:, nA+1:end, idx] .= @view XB[:, :, j]
        idx += 1
    end
    return batch
end

function pairing_accuracy(rbm, XA, XB, indices)
    k = length(indices)
    batch = candidate_batch(XA, XB, indices)
    E = reshape(Array(free_energy(rbm, gpu(batch))), k, k)
    assignment, _ = hungarian(E)
    correct = sum(assignment[i] == i for i in 1:k)
    return correct
end

# Null-model baseline: score with the two frozen single-family models alone
# (no cross term at all). Because that cost is additively separable in i and
# j, every permutation ties in total cost -- Hungarian's pick on it is a
# solver tie-break artifact, not a meaningful "no-signal" number, so it is
# reported only as a pipeline sanity check (if this came out suspiciously
# high, that would flag a bug elsewhere in the harness, not a real result).
# The rigorous chance baseline is the expected accuracy of a uniformly
# random permutation, E[hits] = 1, i.e. 1/k per species -- computed
# analytically instead, no sampling needed.
function null_model_correct(rbm_A, rbm_B, XA, XB, indices)
    k = length(indices)
    fa = Array(free_energy(rbm_A, gpu(XA[:, :, indices])))
    fb = Array(free_energy(rbm_B, gpu(XB[:, :, indices])))
    E = fa .+ fb'
    assignment, _ = hungarian(E)
    return sum(assignment[i] == i for i in 1:k)
end

results = NamedTuple[]
for (code, indices) in zip(species_codes, species_indices)
    k             = length(indices)
    correct       = pairing_accuracy(rbm_paired, XA, XB, indices)
    correct_null  = null_model_correct(rbm_A, rbm_B, XA, XB, indices)
    comp          = [membership(i) for i in indices]
    n_train_i     = count(==(:train), comp)
    n_val_i       = count(==(:val), comp)
    n_unused_i    = count(==(:unused), comp)
    comp_p        = [paired_membership(i) for i in indices]
    n_paired_train_i = count(==(:paired_train), comp_p)
    n_paired_other_i = count(==(:paired_other), comp_p)
    push!(results, (; code, k, correct, correct_null,
                      chance = 1.0 / k,
                      n_train = n_train_i, n_val = n_val_i, n_unused = n_unused_i,
                      n_paired_train = n_paired_train_i, n_paired_other = n_paired_other_i))
    println("species=$code  k=$k  correct=$correct/$k  null=$correct_null/$k  chance=1/$k  " *
            "train=$n_train_i val=$n_val_i unused=$n_unused_i  paired_train=$n_paired_train_i paired_other=$n_paired_other_i")
end

total_correct      = sum(r.correct for r in results)
total_correct_null = sum(r.correct_null for r in results)
total_pairs        = sum(r.k for r in results)
mean_chance        = mean(r.chance for r in results)
overall_accuracy      = total_correct / total_pairs
overall_null_accuracy = total_correct_null / total_pairs

# The metric that actually answers "did the paired model's H_ADD units learn
# to generalize": restrict to species where EVERY sequence is paired_other
# (i.e. entirely outside what the paired model's cross-family units trained
# on -- see the CONFIG section comment). At PAIRED_TRAIN_N=0/PAIRED_TRAIN_FRAC=1.0
# this coincides with the plain train/val numbers above (paired_train_idx ==
# train_idx), so nothing here changes existing full-training-set results.
paired_heldout   = filter(r -> r.n_paired_train == 0, results)
paired_trainonly = filter(r -> r.n_paired_other == 0, results)
n_ph_correct = sum(r.correct for r in paired_heldout; init=0)
n_ph_pairs   = sum(r.k for r in paired_heldout; init=0)
n_pt_correct = sum(r.correct for r in paired_trainonly; init=0)
n_pt_pairs   = sum(r.k for r in paired_trainonly; init=0)
paired_heldout_accuracy   = n_ph_pairs > 0 ? n_ph_correct / n_ph_pairs : NaN
paired_heldout_chance     = length(paired_heldout) > 0 ? mean(r.chance for r in paired_heldout) : NaN
paired_trainonly_accuracy = n_pt_pairs > 0 ? n_pt_correct / n_pt_pairs : NaN

open(path_report, "w") do f
    println(f, "rbm_paired=$PATH_RBM_PAIRED")
    println(f, "rbm_A=$PATH_RBM_A")
    println(f, "rbm_B=$PATH_RBM_B")
    println(f, "PAIRED_TRAIN_FRAC=$PAIRED_TRAIN_FRAC PAIRED_TRAIN_N=$PAIRED_TRAIN_N SEED=$SEED")
    println(f, "species_tested=$(length(results)) total_pairs=$total_pairs")
    println(f, "overall_accuracy=$overall_accuracy")
    println(f, "overall_null_model_accuracy=$overall_null_accuracy  # degenerate cost matrix, tie-break artifact -- not a rigorous baseline")
    println(f, "mean_per_species_chance_baseline=$mean_chance  # analytic 1/k average -- the rigorous baseline")
    println(f, "")
    println(f, "paired_heldout_species=$(length(paired_heldout)) paired_heldout_pairs=$n_ph_pairs")
    println(f, "paired_heldout_accuracy=$paired_heldout_accuracy  # the fair generalization number: species entirely outside paired_train_idx")
    println(f, "paired_heldout_chance_baseline=$paired_heldout_chance")
    println(f, "paired_trainonly_accuracy=$paired_trainonly_accuracy  # species entirely inside paired_train_idx (memorization check), n=$(length(paired_trainonly))")
    println(f, "")
    println(f, "code k correct correct_null chance n_train n_val n_unused n_paired_train n_paired_other")
    for r in results
        println(f, "$(r.code) $(r.k) $(r.correct) $(r.correct_null) $(r.chance) $(r.n_train) $(r.n_val) $(r.n_unused) $(r.n_paired_train) $(r.n_paired_other)")
    end
end

println("\n=== SUMMARY ===")
println("species tested                                     : $(length(results))")
println("total pairs                                         : $total_pairs")
println("overall accuracy                                     : $overall_accuracy")
println("null-model accuracy (sanity check only, see comments): $overall_null_accuracy")
println("paired-heldout accuracy (fair generalization number) : $paired_heldout_accuracy  ($n_ph_pairs pairs, $(length(paired_heldout)) species, chance=$paired_heldout_chance)")
println("paired-trainonly accuracy (memorization check)       : $paired_trainonly_accuracy  ($n_pt_pairs pairs, $(length(paired_trainonly)) species)")
println("chance-level baseline (mean 1/k across species)      : $mean_chance")
println("report written to $path_report")
