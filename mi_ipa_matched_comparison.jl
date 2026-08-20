using HDF5
using FASTX
using BioSequences
using Random
include("/home/cdiaz-faloh/MI_IPA.jl")

# =============================================================================
# Apples-to-apples MI-IPA comparison against the paired-RBM paralog-pairing
# test (paralog_pairing_pfam.jl), on the exact same PF00072/PF01339 v2
# species split.
#
# "Just the first step" = MI-IPA's Fig. 1A protocol (run_mi_ipa_with_training)
# with n_iterations=1: fit one PMI model from a fixed set of *known-correct*
# pairs (no iterative bootstrapping/self-training), then predict pairs for
# everyone. This is the direct MI-IPA analogue of the RBM: the RBM's training
# data is itself made only of already-correctly-paired sequences (each FASTA
# row concatenates a real protein's own two domains) -- so "N_start known
# pairs" for MI-IPA is the same kind of prior knowledge as "sequences in
# rbm_paired's training set" is for the RBM.
#
# N_start is NOT swept or drawn at random here: it is fixed to the *literal
# same* sequences that were in rbm_paired's own PAIRED training set for this
# species partition -- not merely a matching count -- and accuracy is then
# measured on the *literal same* held-out sequences the RBM's held-out
# numbers came from. That requires reproducing train_potts_pfam.jl's/
# paralog_pairing_pfam.jl's exact species grouping (species_code,
# placeholder-code exclusion, k>=2, exact-dup drop) and cross-referencing by
# FASTA header/id -- not by row position -- since MI_IPA.jl's read_alignment
# rejects a handful of sequences (ambiguous/lowercase/non-standard residues)
# that the RBM pipeline's BioSequences-based reader does not, so the two
# readers' row orders are not guaranteed to line up.
#
# "rbm_paired's own PAIRED training set" is paired_train_idx, not train_idx --
# see the matching comment in paralog_pairing_pfam.jl: at PAIRED_TRAIN_N=0/
# PAIRED_TRAIN_FRAC=1.0 (the default, "very large" case already run) these
# coincide, but PAIRED_TRAIN_N<the full train_idx size makes paired_train_idx
# a genuine subsample, and N_start/held-out here must track THAT, not the
# fixed RBM A/B train_idx/val_idx split (which never changes).
#
# ARGS: [PAIRED_TRAIN_FRAC=1.0] [PAIRED_TRAIN_N=0] [SEED=42]
# =============================================================================

const TRAIN_FRAC = 0.7
const VAL_FRAC   = 0.15
const DEDUP_IDENTITY_THRESHOLD = 0.97
const OUTPUT_DIR = "/home/cdiaz-faloh/results_pfam"
const FASTA_PATH = "/home/cdiaz-faloh/misc/PF00072_PF01339_v2_paired.fasta"
const SPLIT_SITE = 111
const PAIRED_TRAIN_FRAC = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const PAIRED_TRAIN_N    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
const SEED               = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 42

# -----------------------------------------------------------------------
# Step 1: reproduce paralog_pairing_pfam.jl's exact 225-species partition
# and its train/val/unused composition per species, from the RBM's own
# split cache -- verbatim species_code/is_placeholder_code/dedup logic.
# -----------------------------------------------------------------------
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

function species_code(id::AbstractString)
    id_part = split(id, "|")[end]
    id_part = split(id_part, "/")[1]
    parts = split(id_part, "_")
    return length(parts) >= 2 ? String(parts[end]) : String(id_part)
end
is_placeholder_code(code) = occursin(r"^[0-9]", code) || occursin("UNC", code) || occursin("UNK", code)

function build_species_groups_rbm(ids)
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
fasta_tag = replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => "")
split_cache_path = joinpath(OUTPUT_DIR, "split_cache_$(fasta_tag)_TRAIN=$(TRAIN_FRAC)_VAL=$(VAL_FRAC)_DEDUP=$(DEDUP_IDENTITY_THRESHOLD).hdf5")
isfile(split_cache_path) || error("No split cache at $split_cache_path")
# Keep as the raw Vector (HDF5-stored order) until after the paired_train_idx
# draw -- see the matching comment in paralog_pairing_pfam.jl.
train_idx_vec = h5read(split_cache_path, "train_idx")

# Replicate train_potts_pfam.jl's exact RNG draw: same SEED, drawn
# immediately after the split load, nothing else touching the RNG stream
# first. This IS "rbm_paired's own PAIRED training set".
Random.seed!(SEED)
n_paired_train = PAIRED_TRAIN_N > 0 ? min(PAIRED_TRAIN_N, length(train_idx_vec)) :
    floor(Int, PAIRED_TRAIN_FRAC * length(train_idx_vec))
paired_train_idx = (PAIRED_TRAIN_N == 0 && PAIRED_TRAIN_FRAC == 1.0) ? train_idx_vec :
    train_idx_vec[randperm(length(train_idx_vec))[1:n_paired_train]]
paired_train_idx_set = Set(paired_train_idx)
paired_membership(i) = i in paired_train_idx_set ? :paired_train : :paired_other
println("paired_train_idx: $(length(paired_train_idx)) / $(length(train_idx_vec)) train sequences " *
        (PAIRED_TRAIN_N > 0 ? "(PAIRED_TRAIN_N=$(PAIRED_TRAIN_N))" : "(PAIRED_TRAIN_FRAC=$(PAIRED_TRAIN_FRAC))"))

rbm_groups, _ = build_species_groups_rbm(ids)
known_ids   = String[]   # headers of species entirely inside paired_train_idx (the "N_start" set)
heldout_ids = String[]   # headers of species entirely outside paired_train_idx (the evaluation set)
n_species_train = 0
n_species_val   = 0
for (code, idxs) in rbm_groups
    kept = dedup_group(idxs, seqs_raw)
    length(kept) >= 2 || continue
    comp = [paired_membership(i) for i in kept]
    n_pt_i = count(==(:paired_train), comp)
    n_po_i = count(==(:paired_other), comp)
    if n_po_i == 0
        append!(known_ids, ids[kept]); global n_species_train += 1
    elseif n_pt_i == 0
        append!(heldout_ids, ids[kept]); global n_species_val += 1
    end
    # species that straddle paired_train_idx and paired_other (can happen
    # once PAIRED_TRAIN_N < the full train_idx size, unlike the RBM
    # A/B train_idx/val_idx split, which never straddles) are excluded from
    # both sets rather than silently assigned to one -- they're neither a
    # clean "known" nor a clean "held-out" test case.
end
println("Species partition (by paired_train_idx): $n_species_train all-paired-train species ($(length(known_ids)) seqs), " *
        "$n_species_val all-held-out species ($(length(heldout_ids)) seqs)")

known_ids_set   = Set(known_ids)
heldout_ids_set = Set(heldout_ids)

# -----------------------------------------------------------------------
# Step 2: MI-IPA's own alignment reading (may reject a few sequences
# ambiguous/lowercase/non-standard chars that the RBM's reader accepted --
# cross-reference by header, not row position).
# -----------------------------------------------------------------------
function pf_species_parser(header::AbstractString)
    id_part = split(header, "|")[end]
    id_part = split(id_part, "/")[1]
    parts = split(id_part, "_")
    code = length(parts) >= 2 ? String(parts[end]) : String(id_part)
    is_placeholder_code(code) && return missing
    return code
end

X, species, headers = read_alignment(FASTA_PATH; species_of_header = pf_species_parser)
groups, species_ids = build_species_groups(species; min_paralogs = 2)
println("MI-IPA alignment: $(length(headers)) sequences read, $(length(groups)) species (k>=2) after its own filtering")

header_to_row = Dict(h => i for (i, h) in enumerate(headers))

training_idx = Int[]
n_known_missing = 0
for h in known_ids
    if haskey(header_to_row, h)
        push!(training_idx, header_to_row[h])
    else
        global n_known_missing += 1
    end
end
heldout_idx = Set{Int}()
n_heldout_missing = 0
for h in heldout_ids
    if haskey(header_to_row, h)
        push!(heldout_idx, header_to_row[h])
    else
        global n_heldout_missing += 1
    end
end
n_known_missing > 0 && @warn "$n_known_missing known/training sequence(s) dropped by MI-IPA's own alignment reader"
n_heldout_missing > 0 && @warn "$n_heldout_missing held-out sequence(s) dropped by MI-IPA's own alignment reader"

N_start = length(training_idx)
println("N_start (known-correct training pairs, matched to rbm_paired's own training set) = $N_start")
println("held-out evaluation set = $(length(heldout_idx)) sequences")

# -----------------------------------------------------------------------
# Step 3: MI-IPA, first step only (n_iterations=1) -- fit PMI from the
# known pairs alone, predict for everyone, score held-out only.
# -----------------------------------------------------------------------
rng = MersenneTwister(SEED)
training_pairs = [(i, i) for i in training_idx]
training = build_concat(X, SPLIT_SITE, training_pairs)
PMI, Meff = compute_PMIs(training, SPLIT_SITE; pseudocount_weight = 0.15, theta = 0.15)
println("Meff of training set = $(round(Meff, digits=1)) (raw N_start=$N_start)")

Results = predict_pairs(rng, X, groups, PMI, SPLIT_SITE)

function tp_fraction(results, idx_filter)
    sub = filter(r -> r.hk in idx_filter, results)
    isempty(sub) && return (0.0, 0)
    tp = count(r -> r.hk == r.rr, sub)
    return (tp / length(sub), length(sub))
end

tp_heldout, n_heldout_scored = tp_fraction(Results, heldout_idx)
tp_known,   n_known_scored   = tp_fraction(Results, Set(training_idx))
tp_all,     n_all_scored     = tp_fraction(Results, Set(1:size(X, 1)))

# held-out-only chance (this specific 46-species/120-pair subpopulation, not
# the full 225-species population) -- same formula as the RBM plot's chance
# line (1 / mean paralogs per species) but restricted to the held-out rows
# that actually survived MI-IPA's own alignment reader.
heldout_species_k = Dict{Int,Int}()
for r in Results
    r.hk in heldout_idx || continue
    heldout_species_k[r.species] = get(heldout_species_k, r.species, 0) + 1
end
n_heldout_species_scored = length(heldout_species_k)
chance_heldout = n_heldout_species_scored / n_heldout_scored

println()
println("=== MI-IPA, first step only (n_iterations=1), N_start matched to rbm_paired's training set ===")
println("held-out accuracy (never in MI-IPA's known set) : $tp_heldout  ($n_heldout_scored pairs, $n_heldout_species_scored species)")
println("chance on this held-out subset                   : $chance_heldout")
println("known/training-set accuracy (re-predicted, not trivial): $tp_known  ($n_known_scored pairs)")
println("overall accuracy (whole population)               : $tp_all  ($n_all_scored pairs)")
