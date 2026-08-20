"""
Julia port of the MI-IPA algorithm (Bitbol, PLoS Comput Biol 14(11):e1006401, 2018),
originally distributed as Matlab code at https://github.com/anneflo/MI_IPA .

Iteratively pairs paralogs from two co-evolving protein families (e.g. HK/RR)
using mutual-information (pointwise mutual information, PMI) scores and the
Hungarian algorithm, bootstrapping the training set from confidently-predicted
pairs at each round.

Differences from the original Matlab code (generalizations, not algorithmic
changes):
  - Species are grouped with a Dict keyed on a parsed species string rather
    than assumed to be contiguous rows in the alignment, and species numbering
    is derived automatically instead of requiring a pre-built lookup table.
    This makes the code work on any similarly-formatted paired FASTA, not just
    the P2CS HK-RR export this algorithm shipped with.
  - The header -> species-string parser is a user-supplied function
    (`species_of_header`), since header formats vary by dataset. The default
    matches the P2CS convention used in the original dataset (species name is
    the 2nd '|'-delimited field).
  - Rows whose header does not yield a species string (e.g. stray reference or
    dummy records) are silently excluded, rather than relying on positional
    "first/last row" conventions.

Core algorithm (weighting, pseudocounts, PMI, Hungarian assignment + gap
scoring, iterative training-set growth) is a direct translation of:
  Compute_PMIs.m, Compute_pairing_scores.m, Predict_pairs.m, ScrambleSeqs.m,
  randomize_equal_rows.m / randomize_equal_cols.m, MI_IPA_main.m
"""

using Random
using DelimitedFiles
using FASTX
using Hungarian

# -----------------------------------------------------------------------
# Alphabet / encoding (mirrors readAlignment_and_NumberSpecies.m exactly:
# '-' => 1, then the 20 amino acids A..Y => 2..21)
# -----------------------------------------------------------------------

const Q = 21

const AA_TO_CODE = Dict{Char,Int}(
    '-' => 1,
    'A' => 2, 'C' => 3, 'D' => 4, 'E' => 5, 'F' => 6, 'G' => 7, 'H' => 8,
    'I' => 9, 'K' => 10, 'L' => 11, 'M' => 12, 'N' => 13, 'P' => 14,
    'Q' => 15, 'R' => 16, 'S' => 17, 'T' => 18, 'V' => 19, 'W' => 20, 'Y' => 21,
)
# Ambiguous (B,Z,J,X,U,O), lowercase (insert states) and '.' are not in
# AA_TO_CODE and cause the whole sequence to be rejected by encode_sequence,
# matching the original code's stated intent ("skip sequences containing
# these" / "skip in seq of interest").

"""Encode a sequence string to a Vector{Int} of codes in 1:21, or `nothing`
if it contains any character outside the standard 20-aa + gap alphabet."""
function encode_sequence(seq::AbstractString)
    L = length(seq)
    v = Vector{Int}(undef, L)
    @inbounds for (i, c) in enumerate(seq)
        code = get(AA_TO_CODE, c, 0)
        code == 0 && return nothing
        v[i] = code
    end
    return v
end

# -----------------------------------------------------------------------
# Reading + species grouping
# -----------------------------------------------------------------------

"""Default header -> species-string parser, matching the P2CS HK-RR dataset
convention: species name is the 2nd '|'-delimited field, e.g.
">Acear_0856_HK_Classic|Acetohalobium_arabaticum_DSM_5501|...".
Returns `missing` for headers that don't have at least 2 '|'-fields (e.g.
stray reference/dummy records), which then simply never join any species
group. Pass your own function for other header conventions."""
function default_species_parser(header::AbstractString)
    parts = split(header, '|')
    return length(parts) >= 2 ? String(parts[2]) : missing
end

"""Read a FASTA alignment, encode it (rejecting sequences with ambiguous /
non-standard characters), and parse a species string per row.

Returns `(X, species, headers)` where `X::Matrix{Int}` is N x L (1-indexed
amino acid codes, see AA_TO_CODE), `species::Vector{Union{String,Missing}}`,
and `headers::Vector{String}`."""
function read_alignment(fasta_path::AbstractString;
                         species_of_header::Function = default_species_parser)
    headers = String[]
    codes = Vector{Vector{Int}}()
    n_rejected = 0
    L = -1

    reader = FASTA.Reader(open(fasta_path))
    for record in reader
        h = FASTA.identifier(record)
        s = String(FASTA.sequence(record))
        enc = encode_sequence(s)
        if enc === nothing
            n_rejected += 1
            continue
        end
        if L == -1
            L = length(enc)
        elseif length(enc) != L
            error("Sequence '$h' has length $(length(enc)), expected $L (all rows " *
                  "of the alignment must have the same width).")
        end
        push!(headers, h)
        push!(codes, enc)
    end
    close(reader)

    n_rejected > 0 &&
        @warn "$n_rejected sequence(s) rejected (contained ambiguous, lowercase, or non-standard characters)"

    N = length(codes)
    X = Matrix{Int}(undef, N, L)
    @inbounds for i in 1:N
        X[i, :] = codes[i]
    end

    species = [species_of_header(h) for h in headers]

    return X, species, headers
end

"""Group row indices of X by species string, keeping only species with at
least `min_paralogs` members (mirrors SuppressSpeciesWithOnePair.m, which
special-cased exactly 1 pair; here it's a general threshold). Groups are
returned in order of first appearance, for deterministic downstream
ordering."""
function build_species_groups(species::AbstractVector; min_paralogs::Int = 2)
    idx_of = Dict{Any,Vector{Int}}()
    order = Any[]
    for (i, s) in enumerate(species)
        ismissing(s) && continue
        if !haskey(idx_of, s)
            idx_of[s] = Int[]
            push!(order, s)
        end
        push!(idx_of[s], i)
    end
    groups = Vector{Int}[]
    species_ids = eltype(order)[]
    for s in order
        if length(idx_of[s]) >= min_paralogs
            push!(groups, idx_of[s])
            push!(species_ids, s)
        end
    end
    return groups, species_ids
end

# -----------------------------------------------------------------------
# Reweighting (Meff) + PMI computation (Compute_PMIs.m)
# -----------------------------------------------------------------------

"""Sequence weights via the standard DCA-style reweighting scheme: weight_i =
1 / (1 + #{j != i : Hamming_fraction(i,j) < theta}). theta<=0 disables
reweighting (all weights = 1). Mirrors count_alignment's use of
pdist(...,'hamm') < theta.

Threaded over `i` with each thread scanning all `j != i` (rather than the
usual i<j half-loop), so per-i counts have no cross-thread write conflicts;
worthwhile trade given Julia's default is single-threaded and this is one of
the two hot loops (run with `julia -t auto` to benefit)."""
function compute_weights(X::AbstractMatrix{<:Integer}, theta::Real)
    M, L = size(X)
    W = ones(Float64, M)
    if theta > 0
        cnt = zeros(Int, M)
        Threads.@threads for i in 1:M
            ci = 0
            @inbounds for j in 1:M
                j == i && continue
                d = 0
                for k in 1:L
                    d += X[i, k] != X[j, k]
                end
                if d / L < theta
                    ci += 1
                end
            end
            cnt[i] = ci
        end
        W .= 1.0 ./ (1.0 .+ cnt)
    end
    return W, sum(W)
end

"""Compute PMI scores for a (training) concatenated alignment X (M x L),
restricted to the *inter-family* (A,B) block, i.e. site i in family A
(1:LengthA) paired with site j in family B (LengthA+1:L). This is the only
block `pairing_scores`/`predict_pairs` ever read (a pairing score only ever
sums inter-protein PMIs, see Eq. 3 of the paper / Compute_pairing_scores.m),
so the within-family blocks are skipped entirely -- same result, much less
compute and memory (roughly (L/LengthB) x less for typical family-length
splits) than materializing the full L x L PMI array.

Returns `(PMI, Meff)` where `PMI` is a LengthA x LengthB x q x q array;
`PMI[a, b, alpha, beta]` is the PMI between family-A site `a` and family-B
site `LengthA+b`. Only meaningful for alpha,beta in 2:q (the gap state,
index 1, is ignored, matching Get_PMIs' comment "the 1st aa type(=gap) is
ignored")."""
function compute_PMIs(X::AbstractMatrix{<:Integer}, LengthA::Int;
                       pseudocount_weight::Real = 0.15, theta::Real = 0.15, q::Int = Q)
    M, L = size(X)
    LengthB = L - LengthA
    W, Meff = compute_weights(X, theta)

    # single-site frequencies (still needed at every position, both families)
    Pi = zeros(Float64, L, q)
    @inbounds for m in 1:M
        w = W[m]
        for i in 1:L
            Pi[i, X[m, i]] += w
        end
    end
    Pi ./= Meff

    # inter-family pairwise frequencies only (i in A, j in B -- i can never
    # equal j, so no within-family/diagonal handling is needed here at all)
    Pij = zeros(Float64, LengthA, LengthB, q, q)
    @inbounds for m in 1:M
        w = W[m]
        for i in 1:LengthA
            ai = X[m, i]
            for jb in 1:LengthB
                aj = X[m, LengthA+jb]
                Pij[i, jb, ai, aj] += w
            end
        end
    end
    Pij ./= Meff

    # pseudocounts (with_pc.m); no diagonal special-case since i,j are
    # always in different families here
    Pij_pc = (1 - pseudocount_weight) .* Pij .+ pseudocount_weight / q^2
    Pi_pc = (1 - pseudocount_weight) .* Pi .+ pseudocount_weight / q

    # PMIs (Get_PMIs.m): only for non-gap states.
    # Loop order matters a lot here: arrays are column-major, so `i` (dim 1)
    # must be the innermost/fastest-varying loop for sequential memory
    # access -- `i,j,a,b` nesting (the "natural"-looking order) makes `b`
    # fastest, which strides badly and thrashes cache.
    PMI = zeros(Float64, LengthA, LengthB, q, q)
    @inbounds for b in 2:q, a in 2:q, jb in 1:LengthB
        Pi_pc_jb = Pi_pc[LengthA+jb, b]
        for i in 1:LengthA
            PMI[i, jb, a, b] = log(Pij_pc[i, jb, a, b] / (Pi_pc[i, a] * Pi_pc_jb))
        end
    end

    return PMI, Meff
end

"""Preallocated scratch space for `compute_PMIs!`, so repeated calls (e.g.
one per iteration of an MI-IPA run) don't each allocate ~3 fresh L x L x q x q
arrays (~330MB combined for the HK-RR dataset's L=176) and churn the GC.
`L`/`q` must match the alignment width/alphabet size used across all calls
sharing one buffer set."""
struct PMIBuffers
    LengthA::Int
    Pi::Matrix{Float64}
    Pij::Array{Float64,4}
    Pij_pc::Array{Float64,4}
    Pi_pc::Matrix{Float64}
    PMI::Array{Float64,4}
end
PMIBuffers(L::Int, LengthA::Int, q::Int = Q) = PMIBuffers(
    LengthA, zeros(L, q), zeros(LengthA, L - LengthA, q, q), zeros(LengthA, L - LengthA, q, q),
    zeros(L, q), zeros(LengthA, L - LengthA, q, q))

"""In-place version of `compute_PMIs`, writing into `buf` instead of
allocating fresh arrays (see `compute_PMIs` docstring re: restricting to the
inter-family block). Returns `(buf.PMI, Meff)` -- the returned PMI array is a
*view into `buf`* and will be overwritten by the next call, so it must be
fully consumed (e.g. by `predict_pairs`) before reusing `buf`."""
function compute_PMIs!(buf::PMIBuffers, X::AbstractMatrix{<:Integer};
                        pseudocount_weight::Real = 0.15, theta::Real = 0.15, q::Int = Q)
    M, L = size(X)
    LengthA = buf.LengthA
    LengthB = L - LengthA
    @assert size(buf.Pi, 1) == L "PMIBuffers built for a different alignment width L"
    W, Meff = compute_weights(X, theta)

    Pi, Pij, Pij_pc, Pi_pc, PMI = buf.Pi, buf.Pij, buf.Pij_pc, buf.Pi_pc, buf.PMI

    fill!(Pi, 0.0)
    @inbounds for m in 1:M
        w = W[m]
        for i in 1:L
            Pi[i, X[m, i]] += w
        end
    end
    Pi ./= Meff

    fill!(Pij, 0.0)
    @inbounds for m in 1:M
        w = W[m]
        for i in 1:LengthA
            ai = X[m, i]
            for jb in 1:LengthB
                aj = X[m, LengthA+jb]
                Pij[i, jb, ai, aj] += w
            end
        end
    end
    Pij ./= Meff

    Pij_pc .= (1 - pseudocount_weight) .* Pij .+ pseudocount_weight / q^2
    Pi_pc .= (1 - pseudocount_weight) .* Pi .+ pseudocount_weight / q

    # PMI[.,.,a,b] with a==1 or b==1 (gap state) must read as 0 (matches
    # Get_PMIs' "gap is ignored"), and since `buf.PMI` is reused across
    # calls it needs an explicit reset here -- a fresh `zeros(...)` array
    # would get this for free, but a reused buffer won't. Gaps are common
    # in these alignments (rows with '-'), so a stale nonzero here would be
    # a real correctness bug, not just cosmetic.
    fill!(PMI, 0.0)
    @inbounds for b in 2:q, a in 2:q, jb in 1:LengthB
        Pi_pc_jb = Pi_pc[LengthA+jb, b]
        for i in 1:LengthA
            PMI[i, jb, a, b] = log(Pij_pc[i, jb, a, b] / (Pi_pc[i, a] * Pi_pc_jb))
        end
    end

    return PMI, Meff
end

# -----------------------------------------------------------------------
# Pairing scores + Hungarian assignment with gap (Compute_pairing_scores.m,
# Predict_pairs.m, randomize_equal_rows.m/randomize_equal_cols.m)
# -----------------------------------------------------------------------

"""k x k matrix of summed PMI scores (to be *maximized*) between every pair
of candidate A/B sequences (rows `idxs` of X) within one species. `PMI` is
the LengthA x LengthB x q x q inter-family block returned by
`compute_PMIs`/`compute_PMIs!` (see their docstrings)."""
function pairing_scores(X::AbstractMatrix{<:Integer}, idxs::Vector{Int},
                         PMI::AbstractArray{Float64,4}, LengthA::Int)
    k = length(idxs)
    L = size(X, 2)
    LengthB = L - LengthA
    S = zeros(Float64, k, k)
    # `a` (dim 1 of PMI) innermost for sequential memory access, same
    # rationale as the loop reorder in compute_PMIs.
    @inbounds for j in 1:k
        rr = idxs[j]
        for i in 1:k
            hk = idxs[i]
            s = 0.0
            for jb in 1:LengthB
                aa2 = X[rr, LengthA+jb]
                for a in 1:LengthA
                    s += PMI[a, jb, X[hk, a], aa2]
                end
            end
            S[i, j] = s
        end
    end
    return S
end

"""Randomly permute `assignment` among rows (dims=1) or columns (dims=2) of
C that are exactly tied (identical score vectors), to remove the arbitrary
tie-breaking bias of the underlying deterministic solver. C is the k x k
*cost* matrix (i.e. -score) that produced `assignment`."""
function derandomize_ties!(rng::AbstractRNG, assignment::Vector{Int}, C::AbstractMatrix{Float64}, dims::Int)
    k = size(C, 1)
    groups = Dict{Vector{Float64},Vector{Int}}()
    if dims == 1
        for i in 1:k
            push!(get!(groups, C[i, :], Int[]), i)
        end
        for (_, rows) in groups
            if length(rows) > 1
                perm = rows[randperm(rng, length(rows))]
                assignment[rows] = assignment[perm]
            end
        end
    else
        for j in 1:k
            push!(get!(groups, C[:, j], Int[]), j)
        end
        for (_, cols) in groups
            if length(cols) > 1
                rows = findall(r -> assignment[r] in cols, 1:length(assignment))
                if length(rows) > 1
                    perm = rows[randperm(rng, length(rows))]
                    assignment[rows] = assignment[perm]
                end
            end
        end
    end
    return assignment
end

"""Solve the assignment problem for one species' k x k score matrix S
(to be maximized) and return `(assignment, gap)`, where `gap[j]` is the
increase in total optimal cost incurred by forbidding sequence j's chosen
partner (a per-row confidence/reliability score used to rank pairs for the
next training round). Mirrors the case analysis in Predict_pairs.m."""
function species_assignment_and_gap(rng::AbstractRNG, S::AbstractMatrix{Float64})
    k = size(S, 1)
    C = -S # Hungarian.jl minimizes; we want to maximize S

    if k == 1
        return [1], [abs(S[1, 1])]
    end

    if minimum(S) == maximum(S)
        return randperm(rng, k), zeros(Float64, k)
    end

    assignment, score = hungarian(C)
    derandomize_ties!(rng, assignment, C, 1)
    derandomize_ties!(rng, assignment, C, 2)

    bigval = 1e3 * maximum(abs, C) + 1.0
    gaps = Vector{Float64}(undef, k)
    Cmod = similar(C)
    @inbounds for j in 1:k
        copyto!(Cmod, C)
        Cmod[j, assignment[j]] = bigval
        _, score_mod = hungarian(Cmod)
        gaps[j] = score_mod - score
    end

    return assignment, gaps
end

"""One prediction record: `hk`/`rr` are row indices into the full alignment
X. Ground truth is `hk == rr` (each row of X is, by construction, a true
A/B pair before any scrambling)."""
const Pairing = @NamedTuple{species::Int, hk::Int, rr::Int, score::Float64, gap::Float64}

"""Predict pairings (with gap/confidence scores) for every species, using
the model `PMI`. `groups` is a Vector of row-index vectors, one per
species (see build_species_groups).

Threaded across species (independent work; species assignment is the
dominant cost via repeated Hungarian solves for the gap score). Each species
gets its own RNG stream, seeded from `rng`, both for thread-safety (a shared
MersenneTwister is not thread-safe) and to keep results reproducible
independent of the thread count."""
function predict_pairs(rng::AbstractRNG, X::AbstractMatrix{<:Integer}, groups::Vector{Vector{Int}},
                        PMI::AbstractArray{Float64,4}, LengthA::Int)
    nsp = length(groups)
    per_species = Vector{Vector{Pairing}}(undef, nsp)
    species_seeds = rand(rng, UInt64, nsp)
    Threads.@threads for sp in 1:nsp
        idxs = groups[sp]
        local_rng = MersenneTwister(species_seeds[sp])
        S = pairing_scores(X, idxs, PMI, LengthA)
        assignment, gaps = species_assignment_and_gap(local_rng, S)
        out = Vector{Pairing}(undef, length(idxs))
        for j in eachindex(idxs)
            out[j] = (species = sp, hk = idxs[j], rr = idxs[assignment[j]],
                       score = S[j, assignment[j]], gap = gaps[j])
        end
        per_species[sp] = out
    end
    return reduce(vcat, per_species)
end

# -----------------------------------------------------------------------
# Training-set construction (ScrambleSeqs.m + the training-set-growth logic
# from MI_IPA_main.m)
# -----------------------------------------------------------------------

"""Build an n x L concatenated-alignment matrix from a list of (hk_row,
rr_row) index pairs: column 1:LengthA taken from X[hk,:], the rest from
X[rr,:]."""
function build_concat(X::AbstractMatrix{<:Integer}, LengthA::Int, pairs::Vector{Tuple{Int,Int}})
    L = size(X, 2)
    Y = Matrix{Int}(undef, length(pairs), L)
    @inbounds for (row, (hk, rr)) in enumerate(pairs)
        Y[row, 1:LengthA] = @view X[hk, 1:LengthA]
        Y[row, LengthA+1:L] = @view X[rr, LengthA+1:L]
    end
    return Y
end

"""Random within-species pairing, used to bootstrap round 1 (ScrambleSeqs.m)."""
function scramble_pairs(rng::AbstractRNG, groups::Vector{Vector{Int}})
    pairs = Tuple{Int,Int}[]
    for idxs in groups
        k = length(idxs)
        p = randperm(rng, k)
        for i in 1:k
            push!(pairs, (idxs[i], idxs[p[i]]))
        end
    end
    return pairs
end

# -----------------------------------------------------------------------
# Main iterative driver (MI_IPA_main.m)
# -----------------------------------------------------------------------

"""
    run_mi_ipa(fasta_path; LengthA, Nincrement, kwargs...)

Run the full iterative MI-IPA on a paired-family FASTA alignment (fixed
width, family A occupying columns 1:LengthA, family B the rest).

Keyword args:
- `LengthA::Int`             length of protein/family A (required)
- `Nincrement::Int`          number of pairs added to the training set per round (required)
- `pseudocount_weight=0.15`  as in Compute_PMIs.m
- `theta=0.15`                Hamming-fraction threshold for Meff reweighting
- `min_paralogs=2`            species with fewer paralogs than this are dropped
- `species_of_header=default_species_parser`
- `seed=1`
- `verbose=true`

Returns `(Output, Results, X, groups, species_ids, headers)` where:
- `Output` is Nrounds x 6: [NSeqs_train, Meff, TP_all, FP_all, TP_train, FP_train]
  (columns match Output in MI_IPA_main.m)
- `Results` is the final round's Vector{Pairing} (species, hk row, rr row,
  score, gap) predicted on the *full* alignment
- `groups`/`species_ids` let you map `species` indices in `Results` back to
  original species strings
"""
function run_mi_ipa(fasta_path::AbstractString;
                     LengthA::Int,
                     Nincrement::Int,
                     pseudocount_weight::Real = 0.15,
                     theta::Real = 0.15,
                     min_paralogs::Int = 2,
                     species_of_header::Function = default_species_parser,
                     seed::Int = 1,
                     verbose::Bool = true)

    rng = MersenneTwister(seed)

    X, species_raw, headers = read_alignment(fasta_path; species_of_header = species_of_header)
    L = size(X, 2)
    verbose && println("Read $(size(X,1)) valid sequences, alignment width L=$L (A: 1:$LengthA, B: $(LengthA+1):$L)")

    groups, species_ids = build_species_groups(species_raw; min_paralogs = min_paralogs)
    Ntot = sum(length, groups)
    verbose && println("$(length(groups)) species with >= $min_paralogs paralogs, $Ntot sequences retained")

    Nrounds = ceil(Int, Ntot / Nincrement) + 1
    verbose && println("Running $Nrounds rounds (Nincrement=$Nincrement)")

    Output = zeros(Float64, Nrounds, 6)
    Results = Pairing[]
    NSeqs_new = 0

    for iter in 1:Nrounds
        verbose && println("round $iter / $Nrounds")

        if iter == 1
            pairs = scramble_pairs(rng, groups)
        else
            sorted = sort(Results, by = r -> -r.gap)
            NSeqs_new = min(NSeqs_new + Nincrement, length(sorted))
            top = @view sorted[1:NSeqs_new]
            Output[iter, 5] = count(r -> r.hk == r.rr, top)
            Output[iter, 6] = NSeqs_new - Output[iter, 5]
            pairs = [(r.hk, r.rr) for r in top]
        end

        training = build_concat(X, LengthA, pairs)
        PMI, Meff = compute_PMIs(training, LengthA; pseudocount_weight = pseudocount_weight, theta = theta)

        Results = predict_pairs(rng, X, groups, PMI, LengthA)

        Output[iter, 1] = NSeqs_new
        Output[iter, 2] = Meff
        Output[iter, 3] = count(r -> r.hk == r.rr, Results)
        Output[iter, 4] = length(Results) - Output[iter, 3]

        if verbose
            tp, fp = Output[iter, 3], Output[iter, 4]
            println("  NSeqs_train=$(Int(Output[iter,1])) Meff=$(round(Meff,digits=1)) TP=$(Int(tp)) FP=$(Int(fp)) (acc=$(round(tp/(tp+fp),digits=4)))")
        end
    end

    return Output, Results, X, groups, species_ids, headers
end

"""Write Output and Results in the same tab-delimited format as
MI_IPA_main.m's dlmwrite calls."""
function save_results(prefix::AbstractString, Output::Matrix{Float64}, Results::Vector{Pairing})
    writedlm(prefix * "_TP_data.txt", Output, '\t')
    M = [[r.species r.hk r.rr r.score r.gap] for r in Results]
    writedlm(prefix * "_Resf.txt", reduce(vcat, M), '\t')
end

# -----------------------------------------------------------------------
# Fig. 1A protocol (Bitbol 2018 biorxiv/PLoS CB, "MI-IPA starting from a
# training set of known partners"): distinct from run_mi_ipa above (which
# reproduces the "no training set" protocol of Fig. 2/MI_IPA_main.m).
#
# Per the Methods section ("Initialization of the CA" / step 4
# "Incrementation of the CA"):
#   - iteration 1: CA = the N_start known-correct training pairs only.
#   - iteration n>1: CA = the SAME N_start training pairs (always kept),
#     plus the (n-1)*Nincrement pairs with the highest confidence (gap)
#     score from iteration n-1's predictions, drawn only from OUTSIDE the
#     training set (training members are not "re-paired").
#   - At every iteration, pairs/scores are predicted for the full dataset
#     (all species, all members), and TP fraction is the fraction of the
#     full dataset correctly paired (hk row == rr row).
# -----------------------------------------------------------------------

"""Chance-level TP fraction for random within-species pairing: a uniformly
random permutation of m elements has expectation exactly 1 fixed point
regardless of m, so the expected number of correct pairs is just the number
of species, independent of species sizes (matches the paper's reported
~9% "random expectation" for the standard HK-RR dataset)."""
chance_tp_fraction(groups::Vector{Vector{Int}}) = length(groups) / sum(length, groups)

"""One replicate of the Fig. 1A protocol: MI-IPA starting from `N_start`
known-correct pairs (kept fixed in the CA throughout), growing the CA by
`Nincrement` highest-confidence predicted pairs per iteration, for
`n_iterations` iterations. Returns a Vector{Float64} of length
`n_iterations`, the TP fraction (over the full dataset) at each iteration.

`X`/`groups` can be precomputed once (via read_alignment/build_species_groups)
and passed in to avoid re-parsing the FASTA for every replicate/N_start."""
function run_mi_ipa_with_training(X::AbstractMatrix{<:Integer}, groups::Vector{Vector{Int}};
                                   LengthA::Int,
                                   N_start::Int,
                                   Nincrement::Int,
                                   n_iterations::Int,
                                   pseudocount_weight::Real = 0.15,
                                   theta::Real = 0.15,
                                   seed::Int = 1,
                                   buf::PMIBuffers = PMIBuffers(size(X, 2), LengthA))
    rng = MersenneTwister(seed)

    all_rows = reduce(vcat, groups)
    Ntot = length(all_rows)
    N_start = min(N_start, Ntot)

    training_idx = Set(all_rows[randperm(rng, Ntot)[1:N_start]])
    training_pairs = [(i, i) for i in training_idx]

    tp_fraction = Vector{Float64}(undef, n_iterations)
    Results = Pairing[]
    max_extra = Ntot - N_start

    for iter in 1:n_iterations
        if iter == 1
            pairs = training_pairs
        else
            n_extra = min((iter - 1) * Nincrement, max_extra)
            sorted = sort(Results, by = r -> -r.gap)
            extra = Tuple{Int,Int}[]
            sizehint!(extra, n_extra)
            for r in sorted
                r.hk in training_idx && continue
                push!(extra, (r.hk, r.rr))
                length(extra) == n_extra && break
            end
            pairs = vcat(training_pairs, extra)
        end

        training = build_concat(X, LengthA, pairs)
        PMI, Meff = compute_PMIs!(buf, training; pseudocount_weight = pseudocount_weight, theta = theta)
        Results = predict_pairs(rng, X, groups, PMI, LengthA)

        tp_fraction[iter] = count(r -> r.hk == r.rr, Results) / length(Results)
    end

    return tp_fraction
end
