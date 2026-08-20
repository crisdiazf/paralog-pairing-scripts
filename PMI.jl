using LinearAlgebra
using Serialization
using Random
using Hungarian
using LatentAlignedRBMs
using BioSequences
using DelimitedFiles

const N_STATES = 21

Random.seed!(42)  # For reproducibility

function compute_frequencies_from_CA(CA::Array{T,3}, pseudocount::Float64=0.01; 
                                     verbose::Bool=true) where T<:Real
    """
    Compute one-body and two-body frequencies from a concatenated alignment.
    Optimized version with sparse initialization.
    """
    q, L, M = size(CA)
    
    if verbose
        println("Computing frequencies from CA:")
        println("  Sequences (M): $M")
        println("  Positions (L): $L")
        println("  States (q): $q")
        println("  Pseudocount: $pseudocount")
        start_time = time()
    end
    
    # Convert to Float64
    verbose && print("  Converting to Float64... ")
    CA_float = Float64.(CA)
    verbose && println("done")
    
    # ----- One-body frequencies -----
    verbose && print("  Computing one-body frequencies... ")
    f1 = dropdims(sum(CA_float, dims=3), dims=3)
    f1 = permutedims(f1, [2, 1])
    f1 = (f1 .+ pseudocount) ./ (M + pseudocount * q)
    verbose && println("done")
    
    # ----- Two-body frequencies (lazy initialization) -----
    verbose && println("  Counting co-occurrences (building dictionary on the fly)...")
    
    f2 = Dict{Tuple{Int,Int,Int,Int},Float64}()
    
    # Count co-occurrences, only creating entries when needed
    for m in 1:M
        if verbose && m % max(1, M ÷ 10) == 0
            println("    Sequence $m / $M ($(round(100*m/M, digits=1))%)")
        end
        
        for i in 1:L
            a = argmax(CA[:, i, m])
            for j in 1:L
                if i != j
                    b = argmax(CA[:, j, m])
                    key = (i, j, a, b)
                    f2[key] = get(f2, key, 0.0) + 1.0
                end
            end
        end
    end
    
    if verbose
        println("    Observed $(length(f2)) unique (i,j,a,b) combinations")
        println("  Applying pseudocount and normalizing...")
    end
    
    # Apply pseudocount and normalize
    pseudocount_value = pseudocount / q^2
    
    for i in 1:L
        for j in 1:L
            if i != j
                # First, ensure all (a,b) have at least pseudocount
                # Then normalize
                total = 0.0
                for a in 1:q
                    for b in 1:q
                        key = (i, j, a, b)
                        val = get(f2, key, 0.0) / M + pseudocount_value
                        f2[key] = val
                        total += val
                    end
                end
                # Normalize to sum to 1
                if total > 0
                    for a in 1:q
                        for b in 1:q
                            key = (i, j, a, b)
                            f2[key] /= total
                        end
                    end
                end
            end
        end
        if verbose && i % 10 == 0
            println("    Processed position pair $i / $L")
        end
    end
    
    if verbose
        elapsed = time() - start_time
        println("✓ Frequency computation complete in $(round(elapsed, digits=2)) seconds")
    end
    
    return f1, f2
end


function compute_pmis_from_frequencies(f1::Matrix{Float64}, f2::Dict, 
                                        L_A::Int, L_B::Int; 
                                        verbose::Bool=true)
    """
    Compute Pointwise Mutual Information for inter-protein residue pairs.
    
    Arguments:
    - f1: (L, 21) one-body frequencies
    - f2: Dict[(i, j, a, b)] -> joint frequency
    - L_A: length of protein A domain
    - L_B: length of protein B domain
    - verbose: print progress
    
    Returns:
    - pmi: Dict[(i, j, a, b)] -> PMI value for i in A, j in B
    """
    
    L_total = size(f1, 1)
    
    if verbose
        println("Computing PMIs for inter-protein residue pairs:")
        println("  L_A: $L_A")
        println("  L_B: $L_B")
        println("  Total positions: $L_total")
        println("  Number of inter-protein pairs: $(L_A * L_B)")
    end
    
    pmi = Dict{Tuple{Int,Int,Int,Int},Float64}()
    
    # Small epsilon to avoid log(0) issues (though pseudocount should prevent zeros)
    eps = 1e-12
    
    count = 0
    total_pairs = L_A * L_B * 21 * 21
    
    verbose && println("  Computing PMIs...")
    
    for i in 1:L_A
        for j in 1:L_B
            j_global = L_A + j  # position in concatenated alignment
            
            for a in 1:21
                for b in 1:21
                    # Get joint probability
                    prob_ij = get(f2, (i, j_global, a, b), 0.0)
                    
                    # Get marginal probabilities
                    prob_i = f1[i, a]
                    prob_j = f1[j_global, b]
                    
                    # Compute PMI
                    if prob_ij > 0 && prob_i > 0 && prob_j > 0
                        pmi[(i, j, a, b)] = log(prob_ij / (prob_i * prob_j) + eps)
                    else
                        pmi[(i, j, a, b)] = 0.0
                    end
                    
                    count += 1
                    if verbose && count % 1_000_000 == 0
                        println("      Computed $count / $total_pairs PMI entries")
                    end
                end
            end
        end
        if verbose && i % 5 == 0
            println("    Processed A position $i / $L_A")
        end
    end
    
    if verbose
        println("  Done. Computed $(length(pmi)) PMI entries")
        println("✓ PMI computation complete")
    end
    
    return pmi
end


function score_candidate_pairs(seqsA_onehot::BitArray{3}, 
                                seqsB_onehot::BitArray{3},
                                indices::Vector{Int},
                                pmi::Dict,
                                L_A::Int;
                                verbose::Bool=false)
    """
    Score all candidate pairs for a given species.
    Works with BitArray{3} (boolean) one-hot encoding.
    """
    k = length(indices)
    scores = zeros(Float64, k, k)
    
    verbose && println("  Scoring $k x $k candidate pairs...")
    
    # Pre-extract sequences for this species
    A_seqs = [seqsA_onehot[:, :, idx] for idx in indices]
    B_seqs = [seqsB_onehot[:, :, idx] for idx in indices]
    
    # Get L_B from first B sequence
    L_B = size(B_seqs[1], 2)
    
    # Score each pair
    for i in 1:k
        seqA = A_seqs[i]
        
        for j in 1:k
            seqB = B_seqs[j]
            total_score = 0.0
            
            # Loop over all positions in A and B
            for posA in 1:L_A
                # Find which amino acid is at position posA in seqA
                aaA = findfirst(seqA[:, posA])
                if aaA === nothing
                    aaA = 21  # default to gap if not found
                end
                
                for posB in 1:L_B
                    aaB = findfirst(seqB[:, posB])
                    if aaB === nothing
                        aaB = 21
                    end
                    
                    # Look up PMI value
                    total_score += get(pmi, (posA, posB, aaA, aaB), 0.0)
                end
            end
            
            scores[i, j] = total_score
        end
        
        if verbose && i % 10 == 0
            println("    Scored row $i / $k")
        end
    end
    
    verbose && println("  Scoring complete")
    
    return scores
end

function load_split(filename::String)
    lines = readlines(filename)
    train_idx = parse.(Int, split(lines[1]))
    test_idx  = parse.(Int, split(lines[2]))
    return train_idx, test_idx
end

function sequences_with_keep()
    v = LatentAlignedRBMs.PF00072_PF01339_paired_20250416_load_sequences()

    seen = Dict{LongAA, Int}()
    keep = Int[]

    for (i, seq) in enumerate(v)
        if !haskey(seen, seq)
            seen[seq] = i
            push!(keep, i)
        end
    end

    v_unique = v[keep]

    seqs = LatentAlignedRBMs.onehot(v_unique)
    seqsA = seqs[:,1:111,:]
    seqsB = seqs[:,112:end,:]

    return seqsA, seqsB, keep
end

"""
Process a .faa file and keep only species with 2 ≤ paralogs ≤ q.
Returns:
- species_list: a list of species, each is a Vector of paired sequence indices
- species_names: optional vector of species names, aligned with species_list
"""
function build_species_list(species_unique::Vector{String}, q::Int)
    species_dict = Dict{String, Vector{Int}}()

    # Group indices by species
    for (i, sp) in enumerate(species_unique)
        push!(get!(species_dict, sp, Int[]), i)
    end

    species_list = Vector{Vector{Int}}()
    species_names = String[]

    # Filter and keep consistent ordering
    for (sp, inds) in species_dict
        if 2 ≤ length(inds) ≤ q
            push!(species_list, inds)
            push!(species_names, sp)
        end
    end

    return species_list, species_names
end

function extract_species_full(file_path::String)
    species = String[]

    open(file_path, "r") do io
        while !eof(io)
            line = readline(io)

            if startswith(line, ">")
                headers = filter(h -> startswith(h, ">"), split(line))
                first_header = headers[1]

                parts = split(first_header, '|')
                id_part = split(parts[3], "/")[1]
                sp = split(id_part, "_")[end]

                push!(species, sp)
            end
        end
    end

    return species
end

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

path_fasta="./misc/PF00072_PF01339_paired.faa"
species_full = extract_species_full(path_fasta)

# Load sequences and species information
seqsA, seqsB, keep = sequences_with_keep()
species_unique = species_full[keep]
q = 200
species_list, species_names = build_species_list(species_unique, q)

# Load full sequences
seqsA_full = open("./misc/seqs_0072_nodup.bin", "r") do f
    deserialize(f)
end

seqsB_full = open("./misc/seqs_1339_nodup.bin", "r") do f
    deserialize(f)
end

# Load train/test split
train_frac = parse(Int,ARGS[1])
real=parse(Int,ARGS[2])
trainset_name = "./misc/0072_1339_q=200_split_$(train_frac)_$(real).txt"
idx_train, idx_test = load_split(trainset_name)

# Get all indices that appear in species_list
all_species_indices = vcat(species_list...)
sort!(unique!(all_species_indices))

println("Total sequences in species_list: $(length(all_species_indices))")
println("Training sequences among these: $(sum(all_species_indices .∈ Ref(idx_train)))")

# Create a boolean mask for training indices (among those in species_list)
is_train = falses(length(all_species_indices))
for (i, idx) in enumerate(all_species_indices)
    is_train[i] = idx ∈ idx_train
end

# Extract sequences only for indices in species_list
seqsA_filtered = seqsA_full[:, :, all_species_indices]
seqsB_filtered = seqsB_full[:, :, all_species_indices]

# Extract sequences for training set only (for PMI computation or other purposes)
train_mask = [idx ∈ idx_train for idx in all_species_indices]
seqsA_train_only = seqsA_filtered[:, :, train_mask]
seqsB_train_only = seqsB_filtered[:, :, train_mask]
println("Training-only sequences: $(size(seqsA_train_only, 3)) pairs")

# Shuffle B sequences for NON-training sequences only
#idx_shuffle = randperm(size(seqsB_filtered, 3))
#seqsB_shuffled = copy(seqsB_filtered)#
#seqsB_shuffled[:, :, .!is_train] = seqsB_filtered[:, :, idx_shuffle[.!is_train]]

# Build concatenated sequences for ALL sequences in species_list
# Training sequences keep their true pairs, non-training sequences have shuffled B partners
seq = zeros(Int,
    size(seqsA_filtered, 1),
    size(seqsA_filtered, 2) + size(seqsB_filtered, 2),
    size(seqsA_filtered, 3)
)

seq[:, 1:size(seqsA_filtered, 2), :] .= seqsA_filtered
seq[:, size(seqsA_filtered, 2)+1:end, :] .= seqsB_filtered


seq_train_only = zeros(Int,
    size(seqsA_train_only, 1),
    size(seqsA_train_only, 2) + size(seqsB_train_only, 2),
    size(seqsA_train_only, 3)
)
seq_train_only[:, 1:size(seqsA_train_only, 2), :] .= seqsA_train_only
seq_train_only[:, size(seqsA_train_only, 2)+1:end, :] .= seqsB_train_only

# Compute frequencies using ONLY the sequences in species_list
f1, f2 = compute_frequencies_from_CA(seq_train_only, 0.01)

L_A = 111  # HisKA domain length
L_B = 112  # Response_reg domain length

# Compute PMIs
pmi = compute_pmis_from_frequencies(f1, f2, L_A, L_B; verbose=true)

# Evaluate on ALL species in species_list
tp = 0
total_pairs = 0

for (i, indices) in enumerate(species_list)
    k = length(indices)
    
    # Score all candidate pairs using the ORIGINAL sequences
    scores = score_candidate_pairs(seqsA, seqsB, indices, pmi, L_A; verbose=false)
    
    # Hungarian algorithm for matching
    max_score = maximum(scores)
    cost = max_score .- scores
    assignment = hungarian(cost)[1]
    
    # Count correct matches
    correct = sum(assignment[j] == j for j in 1:k)
    println("Species $i: $correct / $k correct pairs")
    
    global tp += correct
    global total_pairs += k
end

println("\nFinal accuracy: $(tp)/$total_pairs = $(round(100 * tp / total_pairs, digits=2))%")
writedlm("./resultados/pmi_tp_nstart=$(train_frac)_$(real).txt", tp/total_pairs)
