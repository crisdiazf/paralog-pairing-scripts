# ======================================================================
# Train paired RBM with partial freezing for HKRR dataset
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
using FASTX
using BioSequences

# ======================================================================
# Load pretrained RBMs for HKRR (A and B chains)
# ======================================================================

# Assuming you have trained RBMs for chain A (positions 1-64) and chain B (positions 65-112)
# Update these paths to match your actual saved RBM files
filename_chainA = "./resultados/rbm_hk_k=50_tt=10000"  # Adjust as needed
filename_chainB = "./resultados/rbm_rr_k=50_tt=10000"  # Adjust as needed

path_chainA = "$filename_chainA.hdf5"
path_chainB = "$filename_chainB.hdf5"

rbm_A = load_rbm(path_chainA)
rbm_B = load_rbm(path_chainB)

# ======================================================================
# Load sequences and prepare one-hot representations
# ======================================================================
fraction = parse(Int, ARGS[1])  # number of sequences to use for training
real = parse(Int, ARGS[3])      # realization number for reproducibility

# Function to load sequences with gaps
function load_sequences_with_gaps(fasta_file::String)
    reader = open(FASTX.FASTA.Reader, fasta_file)
    sequences = LongAA[]
    headers = String[]
    
    for record in reader
        seq_str = String(FASTX.sequence(record))
        header_str = String(FASTX.identifier(record))
        push!(sequences, LongAA(seq_str))
        push!(headers, header_str)
    end
    
    close(reader)
    return headers, sequences
end

# Load full dataset
header, seq_full = load_sequences_with_gaps("./misc/Standard_HKRR_dataset.fasta")
seqs_full = LatentAlignedRBMs.onehot(seq_full)

# Split into chains
seqsA_full = seqs_full[:, 1:64, :]   # Chain A (first 64 positions)
seqsB_full = seqs_full[:, 65:end, :]  # Chain B (remaining positions)

# Total number of sequences
total_seqs = size(seqsA_full, 3)

# Set random seed for reproducibility based on realization number
Random.seed!(real)

# Randomly select 'fraction' sequences for training
idx_train = sort(Random.randperm(total_seqs)[1:fraction])

# Save training indices for reproducibility
dir_results = joinpath(@__DIR__, "resultados")
isdir(dir_results) || mkdir(dir_results)
indices_file = joinpath(dir_results, "train_indices_HKRR_n=$(fraction)_real=$(real).txt")
open(indices_file, "w") do f
    println(f, "# Training indices for HKRR paired RBM")
    println(f, "# fraction=$fraction, realization=$real")
    println(f, "# Total sequences available: $total_seqs")
    println(f, join(idx_train, ","))
end
println("Training indices saved to: $indices_file")

# Select training sequences
seqsA = seqsA_full[:, :, idx_train]
seqsB = seqsB_full[:, :, idx_train]

# Build training concatenated sequences
seq = zeros(Int,
    size(seqsA, 1),
    size(seqsA, 2) + size(seqsB, 2),
    size(seqsA, 3)
)

seq[:, 1:size(seqsA, 2), :] .= seqsA
seq[:, size(seqsA, 2)+1:end, :] .= seqsB

println("Training with $(length(idx_train)) sequences out of $total_seqs total")

# ======================================================================
# Define and initialize paired RBM
# ======================================================================

add_h = parse(Int, ARGS[2])  # number of new hidden units
tt0 = 0

# Total visible units = 64 (chain A) + 48 (chain B) = 112
# Total hidden units = 100 (from chain A) + 100 (from chain B) + add_h
L_A = 64
L_B = 112
total_L = L_A + L_B
total_hidden = 200 + add_h

rbm_paired = RBM(Potts((21, total_L)), xReLU((total_hidden,)), zeros(21, total_L, total_hidden))

initialize!(rbm_paired, seq)

# Copy pretrained parameters
rbm_paired.visible.par[:, :, 1:L_A] .= rbm_A.visible.par
rbm_paired.visible.par[:, :, L_A+1:end] .= rbm_B.visible.par

rbm_paired.hidden.par[:, 1:100] .= rbm_A.hidden.par
rbm_paired.hidden.par[:, 101:200] .= rbm_B.hidden.par

rbm_paired.w[:, 1:L_A, 1:100] .= rbm_A.w
rbm_paired.w[:, L_A+1:end, 101:200] .= rbm_B.w

# ======================================================================
# Define pcd_freeze! function
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
    vm = sample_from_inputs(rbm.visible, falses(size(rbm.visible)..., batchsize)),
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
# Freeze mask: freeze old parameters, train only new hidden units
# ======================================================================

fm = (
    visible = zeros(Float32, size(rbm_paired.visible.par)),
    hidden  = zeros(Float32, size(rbm_paired.hidden.par)),
    w       = zeros(Float32, size(rbm_paired.w))
)
fm.hidden[:, 201:end] .= 1f0
fm.w[:, :, 201:end]   .= 1f0

# ======================================================================
# Training setup
# ======================================================================

k   = 50     # MCMC steps per update
tt  = parse(Int, ARGS[4])   # training iterations
reg = parse(Float32, ARGS[5])     # regularization

ttf = tt0 + tt

# Prepare output filenames
base_name = "rbm_paired_HKRR_n=$(fraction)_nadd=$(add_h)&k=$(k)&tt=$(ttf)&l2l1_weights=$(reg)_real=$(real)"

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
            println(f, lpl)
        end
    end
end

# ======================================================================
# Train the RBM
# ======================================================================

@time pcd_freeze!(
    rbm_paired,
    seq;
    optim = Adam(1f-4, (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = fraction,
    iters = tt,
    vm = bitrand(size(rbm_paired.visible)..., fraction),
    l2l1_weights = reg,
    freeze_mask = fm,
    callback
)

# ======================================================================
# Save trained RBM
# ======================================================================

save_rbm(path_rbm, rbm_paired; overwrite = true)
println("Training complete. RBM saved to: $path_rbm")
println("LPL log saved to: $path_lpl")
