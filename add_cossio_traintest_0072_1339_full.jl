# ======================================================================
# Train paired RBM with partial freezing - Multiple realizations
# ======================================================================

using JLD2
using Serialization
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: RBM, Potts, xReLU, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, free_energy, moments_from_samples,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches
using StandardizedRestrictedBoltzmannMachines: standardize
using LatentAlignedRBMs
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm
using Optimisers: AbstractRule, Adam, setup, update!
using Random, Statistics, DelimitedFiles
using BioSequences
using Hungarian

# ======================================================================
# Function: species_level_split_by_sequences
# ======================================================================

function species_level_split_by_sequences(species_list::Vector{Vector{Int}}, nstart::Int; rng=Random.GLOBAL_RNG)
    n_species = length(species_list)
    
    # Calculate total sequences and cumulative sums
    species_sizes = [length(s) for s in species_list]
    total_sequences = sum(species_sizes)
    
    if nstart >= total_sequences
        @warn "nstart ($nstart) >= total sequences ($total_sequences), using all sequences for training"
        train_indices = vcat(species_list...)
        test_indices = Int[]
        return train_indices, test_indices, copy(species_list), Vector{Vector{Int}}()
    end
    
    # Shuffle species order
    shuffled_order = shuffle(rng, 1:n_species)
    
    # Greedy selection: add species until we reach or exceed nstart
    train_species_idx = Int[]
    current_count = 0
    
    for sp_idx in shuffled_order
        sp_size = species_sizes[sp_idx]
        
        # If adding this species would exceed nstart, check if it's closer than not adding
        if current_count + sp_size > nstart
            # Calculate how far we are from nstart with and without this species
            without_diff = abs(nstart - current_count)
            with_diff = abs(nstart - (current_count + sp_size))
            
            if with_diff < without_diff
                # Adding this species gets us closer to nstart
                push!(train_species_idx, sp_idx)
                current_count += sp_size
            end
            # If without is better, we stop here
            break
        else
            # Add species completely
            push!(train_species_idx, sp_idx)
            current_count += sp_size
        end
    end
    
    # Remaining species go to test
    test_species_idx = setdiff(1:n_species, train_species_idx)
    
    # Create indices and lists
    train_indices = vcat([species_list[i] for i in train_species_idx]...)
    test_indices = vcat([species_list[i] for i in test_species_idx]...)
    
    train_species_list = species_list[train_species_idx]
    test_species_list = species_list[test_species_idx]
    
    println("Target nstart: $nstart")
    println("Actual train sequences: $current_count")
    println("Train species: $(length(train_species_idx))")
    println("Test species: $(length(test_species_idx))")
    
    return train_indices, test_indices, train_species_list, test_species_list
end

# ======================================================================
# Function: pcd_freeze! (with freeze mask support)
# ======================================================================

function pcd_freeze!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1,
    steps::Int = 1,
    optim::AbstractRule = Adam(),
    wts::Union{AbstractVector, Nothing} = nothing,
    l2_fields::Real = 0,
    l1_weights::Real = 0,
    l2_weights::Real = 0,
    l2l1_weights::Real = 0,
    zerosum::Bool = true,
    rescale::Bool = true,
    callback = Returns(nothing),
    vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
    shuffle::Bool = true,
    freeze_mask = nothing
)
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)
    moments = moments_from_samples(rbm.visible, data; wts)
    wts_mean = isnothing(wts) ? 1 : mean(wts)

    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂  = ∂d - ∂m

        if freeze_mask !== nothing
            fm = freeze_mask
            haskey(fm, :visible) && (∂.visible .*= fm.visible)
            haskey(fm, :hidden)  && (∂.hidden  .*= fm.hidden)
            haskey(fm, :w)       && (∂.w       .*= fm.w)
        end

        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)
        callback(; rbm, optim, state, iter, vm, vd, wd)
    end

    return state, ps
end

# ======================================================================
# Load and build dataset
# ======================================================================

println("="^60)
println("LOADING DATASET")
println("="^60)

path_fasta = "./misc/PF00072_PF01339_paired.faa"

# Extract species from FASTA
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

# Load sequences and convert to one-hot
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
    seqsA = seqs[:, 1:111, :]
    seqsB = seqs[:, 112:end, :]
    return seqsA, seqsB, keep
end

# Build species list
function build_species_list(species_unique::Vector{String}, q::Int)
    species_dict = Dict{String, Vector{Int}}()
    for (i, sp) in enumerate(species_unique)
        push!(get!(species_dict, sp, Int[]), i)
    end
    species_list = Vector{Vector{Int}}()
    species_names = String[]
    for (sp, inds) in species_dict
        if 2 ≤ length(inds) ≤ q
            push!(species_list, inds)
            push!(species_names, sp)
        end
    end
    return species_list, species_names
end

# Load everything
species_full = extract_species_full(path_fasta)
seqsA_full, seqsB_full, keep = sequences_with_keep()
species_unique = species_full[keep]
q = 200
species_list, species_names = build_species_list(species_unique, q)

println("Total sequences: $(size(seqsA_full, 3))")
println("Species kept: $(length(species_list))")
println("Total pairs in species_list: $(sum(length.(species_list)))")

# ======================================================================
# Load pretrained RBMs
# ======================================================================

filename_teacher = "./misc/rbm_0072&relu=100&k=50&tt=50000"
filename_student = "./misc/rbm_1339&relu=100&k=50&tt=50000"

path_teacher = "$filename_teacher.hdf5"
path_student = "$filename_student.hdf5"

rbm0072 = load_rbm(path_teacher)
rbm1339 = load_rbm(path_student)

# ======================================================================
# Parse command line arguments
# ======================================================================
# ARGS[1] = nstart (number of training sequences)
# ARGS[2] = add_h (number of additional hidden units)
# ARGS[3] = realization (random seed index, e.g., 1, 2, 3, ...)

nstart = parse(Int, ARGS[1])
add_h = parse(Int, ARGS[2])
realization = parse(Int, ARGS[3])

# Set random seed for reproducibility
seed_value = 1234 + realization
my_rng = MersenneTwister(seed_value)

println("="^60)
println("TRAINING REALIZATION $realization")
println("  nstart = $nstart")
println("  add_h = $add_h")
println("  seed = $seed_value")
println("="^60)

# ======================================================================
# Split species into train/test based on nstart
# ======================================================================

train_idx, test_idx, train_species_list, test_species_list = 
    species_level_split_by_sequences(species_list, nstart; rng=my_rng)

println("\nTrain indices: $(length(train_idx)) sequences from $(length(train_species_list)) species")
println("Test indices: $(length(test_idx)) sequences from $(length(test_species_list)) species")

# ======================================================================
# Save the split indices for reproducibility
# ======================================================================

dir_splits = joinpath(@__DIR__, "splits")
isdir(dir_splits) || mkdir(dir_splits)

split_filename = joinpath(dir_splits, "split_nstart=$(nstart)_real=$(realization).txt")
open(split_filename, "w") do f
    println(f, join(train_idx, ","))
    println(f, join(test_idx, ","))
end
println("Split saved to: $split_filename")

# ======================================================================
# Extract training sequences
# ======================================================================

seqsA = seqsA_full[:, :, train_idx]
seqsB = seqsB_full[:, :, train_idx]

# Build training concatenated sequences
L_A = 111
L_B = size(seqsB, 2)

seq = zeros(Int,
    size(seqsA, 1),
    size(seqsA, 2) + size(seqsB, 2),
    size(seqsA, 3)
)

seq[:, 1:size(seqsA, 2), :] .= seqsA
seq[:, size(seqsA, 2)+1:end, :] .= seqsB

println("Training data shape: $(size(seq))")

# ======================================================================
# Define and initialize paired RBM
# ======================================================================

tt0 = 0

rbm_paired = RBM(Potts((21, L_A + L_B)), xReLU((200 + add_h,)), zeros(21, L_A + L_B, 200 + add_h))

initialize!(rbm_paired, seq)

# Copy pretrained parameters
rbm_paired.visible.par[:, :, 1:L_A] .= rbm0072.visible.par
rbm_paired.visible.par[:, :, L_A+1:end] .= rbm1339.visible.par

rbm_paired.hidden.par[:, 1:100] .= rbm0072.hidden.par
rbm_paired.hidden.par[:, 101:200] .= rbm1339.hidden.par

rbm_paired.w[:, 1:L_A, 1:100] .= rbm0072.w
rbm_paired.w[:, L_A+1:end, 101:200] .= rbm1339.w

# ======================================================================
# Freeze mask: freeze old parameters, train only new hidden units
# ======================================================================

fm = (
    visible = zeros(Float32, size(rbm_paired.visible.par)),
    hidden  = zeros(Float32, size(rbm_paired.hidden.par)),
    w       = zeros(Float32, size(rbm_paired.w))
)
fm.hidden[:, 201:end] .= 1f0
fm.w[:, :, 201:end] .= 1f0

# ======================================================================
# Training setup
# ======================================================================

k = 50       # MCMC steps per update
tt = 2000    # training iterations
reg = 0.01    # regularization

ttf = tt0 + tt

# Prepare output filenames with realization number
base_name = "rbm_paired_PF00072_PF01339_nstart=$(nstart)_nadd=$(add_h)_real=$(realization)_k=$(k)_tt=$(ttf)_reg=$(reg)"
dir_results = joinpath(@__DIR__, "resultados")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

# ======================================================================
# Callback function
# ======================================================================

function callback(; rbm, iter, vm, vd, wd, kwargs...)
    if iszero(iter % 5)
        lpl = mean(log_pseudolikelihood(rbm, vd))
        println("iter=$iter, lpl=$lpl")
        open(path_lpl, "a") do f
            println(f, "$iter $lpl")
        end
    end
end

# ======================================================================
# Train the RBM
# ======================================================================

println("\nStarting training...")
@time pcd_freeze!(
    rbm_paired,
    seq;
    optim = Adam(1f-4, (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = nstart,
    iters = tt,
    vm = bitrand(size(rbm_paired.visible)..., nstart),
    l2l1_weights = reg,
    freeze_mask = fm,
    callback
)

# ======================================================================
# Save trained RBM
# ======================================================================

save_rbm(path_rbm, rbm_paired; overwrite = true)
println("\nTraining complete!")
println("RBM saved to: $path_rbm")
println("LPL log saved to: $path_lpl")
println("Split saved to: $split_filename")
