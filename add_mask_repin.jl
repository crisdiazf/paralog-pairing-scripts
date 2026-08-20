# ======================================================================
# Train paired RBM with partial freezing - GPU version with standardization
# ======================================================================
using JLD2
using HDF5
using Serialization
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: RBM, Potts, PottsGumbel, xReLU, StandardizedRBM, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, free_energy, moments_from_samples,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches, unstandardize,
    standardize, cpu, gpu, load_rbm, save_rbm, pcd!
using LatentAlignedRBMs
using Optimisers: AbstractRule, Adam, setup, update!
using Random, Statistics, DelimitedFiles
using CUDA
using ProgressMeter: @showprogress

# ======================================================================
# Load pretrained RBMs
# ======================================================================
#fraction = parse(Int, ARGS[1])
fraction=ARGS[1]
add_h = parse(Int, ARGS[2])      # number of new hidden units
real = parse(Int, ARGS[3])
tt = parse(Int, ARGS[4])         # training iterations
reg = parse(Float32, ARGS[5])    # regularization
keeptrain = parse(Bool, ARGS[6])
lrate = parse(Int, ARGS[7])

println("all imported")

tt0 = 0
k = 50                           # MCMC steps per update

if keeptrain
    tt0 = 10000
    filename_og = "./resultados/rbm_paired_PF00072_PF01339_q=200_fraction=$(fraction)_nadd=$(add_h)&k=50&tt=2000&l2l1_weights=$(reg)_$(real).hdf5"
    rbm_og = load_rbm(filename_og)
end

ttf = tt0 + tt

filename_teacher = "./resultados/rbm_PF00072_k=100&tt=50000&l2l1_weights=0.1_learnrate04"
filename_student = "./resultados/rbm_PF01339_k=50&tt=50000&l2l1_weights=0.2_learnrate04_nh=100"

path_teacher = "$filename_teacher.hdf5"
path_student = "$filename_student.hdf5"

rbm0072 = load_rbm(path_teacher)
rbm1339 = load_rbm(path_student)

rbm0072 = unstandardize(rbm0072)
rbm1339 = unstandardize(rbm1339)

zerosum!(rbm0072)
zerosum!(rbm1339)

# ======================================================================
# Load paired sequences and prepare one-hot representations
# ======================================================================

println("before")

# Load full datasets
seqsA_full = open("./misc/seqs_0072_nodup.bin", "r") do f
    deserialize(f)
end

seqsB_full = open("./misc/seqs_1339_nodup.bin", "r") do f
    deserialize(f)
end

#seqsA = seqsA_full
#seqsB = seqsB_full


function load_split(filename)
    lines = readlines(filename)
    train_idx = parse.(Int, split(lines[1]))
    test_idx  = parse.(Int, split(lines[2]))
    return train_idx, test_idx
end

#trainset_name="./misc/0072_1339_q=200_split_$(fraction)_$(real).txt"

#idx_train,_=load_split(trainset_name)

seqsA=seqsA_full[:,:,:]
seqsB=seqsB_full[:,:,:]


#Build training concatenated sequences
seq = zeros(Int,
    size(seqsA, 1),
    size(seqsA, 2) + size(seqsB, 2),
    size(seqsA, 3)
)

seq[:, 1:size(seqsA, 2), :] .= seqsA
seq[:, size(seqsA, 2)+1:end, :] .= seqsB

# ======================================================================
# Define and initialize paired RBM
# ======================================================================

rbm_paired = RBM(PottsGumbel((21, 111+174)), xReLU((200 + add_h,)), zeros(21, 111+174, 200 + add_h))

if keeptrain
    rbm_paired = deepcopy(rbm_og)
end

if keeptrain != true
    # Copy pretrained parameters
    rbm_paired.visible.par[:, :, 1:111] .= rbm0072.visible.par
    rbm_paired.visible.par[:, :, 112:end] .= rbm1339.visible.par
    
    rbm_paired.hidden.par[:, 1:100] .= rbm0072.hidden.par
    rbm_paired.hidden.par[:, 101:200] .= rbm1339.hidden.par
    
    rbm_paired.w[:, 1:111, 1:100] .= rbm0072.w
    rbm_paired.w[:, 112:end, 101:200] .= rbm1339.w
    
    rbm_paired.w[:, 1:111, 101:200] .= 0f0
    rbm_paired.w[:, 112:end, 1:100] .= 0f0
    
    rbm_paired.hidden.par[2, 1:end] .= 1
end

#initialize!(rbm_paired, seq)
#zerosum!(rbm_paired)
#rescale_weights!(rbm_paired)

# Store frozen parameters for reference
rbm_frozen = deepcopy(rbm_paired)

# Standardize and move to GPU
rbm_paired = gpu(standardize(rbm_paired))
rbm_frozen_gpu = gpu(standardize(rbm_frozen))
seq_gpu = gpu(seq)

# ======================================================================
# Modified pcd! that masks gradients instead of resetting parameters
# ======================================================================

function pcd_with_freeze!(
    rbm::StandardizedRBM,
    data::AbstractArray;
    batchsize::Int = 256,
    iters::Int = 1,
    steps::Int = 1,
    optim::Adam = Adam(),
    l2l1_weights::Real = 0,
    ϵv::Float32 = 0.1f0,
    ϵh::Float32 = 0f0,
    damping::Float32 = 0.1f0,
    callback = Returns(nothing),
    vm = nothing,
    freeze_mask = nothing
)
    # Setup optimizer state
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)
    
    # Initialize persistent visible units
    if vm === nothing
        vm = sample_from_inputs(rbm.visible, zeros(size(rbm.visible)..., batchsize))
    end
    
    # Precompute moments from data (constant throughout training)
    moments = moments_from_samples(rbm.visible, data)
    
    for iter in 1:iters
        # Sample mini-batch
        idx = randperm(size(data, 3))[1:batchsize]
        vd = data[:, :, idx]
        
        # Run persistent CD
        vm .= sample_v_from_v(rbm, vm; steps)
        
        # Compute gradients
        ∂d = ∂free_energy(rbm, vd; moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m
        
        # Apply freeze mask to gradients (set gradients of frozen params to 0)
        if freeze_mask !== nothing
            fm = freeze_mask
            haskey(fm, :visible) && (∂.visible .*= fm.visible)
            haskey(fm, :hidden)  && (∂.hidden  .*= fm.hidden)
            haskey(fm, :w)       && (∂.w       .*= fm.w)
        end
        
        # Apply regularization
        ∂regularize!(∂, rbm; l2l1_weights = l2l1_weights)
        
        # Update parameters with Adam
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        
        # Enforce constraints after update
        #zerosum!(rbm)
        #rescale_weights!(rbm)
        
        # Re-apply frozen parameters (ensure they match initial values exactly)
        # This is necessary because floating point errors can cause drift
        rbm.visible.par .= rbm_frozen_gpu.visible.par
        rbm.hidden.par[1, 1:200] .= rbm_frozen_gpu.hidden.par[1, 1:200]
	rbm.hidden.par[3, 1:200] .= rbm_frozen_gpu.hidden.par[3, 1:200]
	rbm.hidden.par[4, 1:200] .= rbm_frozen_gpu.hidden.par[4, 1:200]

        rbm.w[:, :, 1:200] .= rbm_frozen_gpu.w[:, :, 1:200]
        # Zero out cross-interactions again
        rbm.w[:, 1:111, 101:200] .= 0f0
        rbm.w[:, 112:end, 1:100] .= 0f0
        
	rbm.hidden.par[2,:].=1
	
        # Callback for logging
        callback(; rbm = rbm, iter = iter, vm = vm, vd = vd)
    end
    
    return state, ps
end

# ======================================================================
# Enhanced callback function for logging
# ======================================================================

# Global variables for paths
const GLOBAL_DIR_RESULTS = Ref{String}()
const GLOBAL_BASE_NAME = Ref{String}()

# Initialize global references
GLOBAL_DIR_RESULTS[] = joinpath(@__DIR__, "resultados/gpu")
GLOBAL_BASE_NAME[] = "rbm_paired_PF00072_PF01339_q=200_fraction=$(fraction)_nadd=$(add_h)&k=$(k)&tt=$(ttf)&l2l1_weights=$(reg)_$(real)_learnrate0$(lrate)_gpu"

function callback_with_logging(; rbm, iter, vm, vd)
    if iszero(iter % 50)
        # Move to CPU for logging
        rbm_cpu = cpu(rbm)
        
        # Compute pseudolikelihood
        lpl = mean(log_pseudolikelihood(rbm_cpu, cpu(seq_gpu[:,:,1:2000])))
        println("iter=$iter, lpl=$lpl")
        
        # Save to log file
        path_lpl = joinpath(GLOBAL_DIR_RESULTS[], "lpl_" * GLOBAL_BASE_NAME[] * ".txt")
        open(path_lpl, "a") do f
            println(f, lpl)
        end
        
    end
    return nothing
end

# ======================================================================
# Training setup
# ======================================================================

# Prepare output filenames
base_name = GLOBAL_BASE_NAME[]
dir_results = GLOBAL_DIR_RESULTS[]
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

# ======================================================================
# Define freeze mask: train only new hidden units (indices 201-200+add_h)
# ======================================================================

# Create mask: 1 = train, 0 = freeze
freeze_mask = (
    visible = zeros(Float32, size(rbm_paired.visible.par)),
    hidden = zeros(Float32, size(rbm_paired.hidden.par)),
    w = zeros(Float32, size(rbm_paired.w))
)

# Train new hidden units
freeze_mask.hidden[:, 201:end] .= 1f0

# Train weights connected to new hidden units
freeze_mask.w[:, :, 201:end] .= 1f0

# Move mask to GPU
freeze_mask_gpu = (
    visible = gpu(freeze_mask.visible),
    hidden = gpu(freeze_mask.hidden),
    w = gpu(freeze_mask.w)
)

# ======================================================================
# Train the RBM
# ======================================================================

batch = min(256, size(seq, 3))

println("batch = $batch")
println("total samples = $(size(seq, 3))")
println("iterations = $tt")
println("MCMC steps = $k")
println("Training new hidden units: 201 to $(200+add_h)")

# Initialize persistent visible units
vm_init = gpu(bitrand(size(rbm_paired.visible)..., batch))

@time pcd_with_freeze!(
    rbm_paired,
    seq_gpu;
    optim = Adam(Float32(10.0^(-lrate)), (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = batch,
    iters = tt,
    vm = vm_init,
    l2l1_weights = reg,
    freeze_mask = freeze_mask_gpu,
    callback = callback_with_logging
)

# ======================================================================
# Save trained RBM
# ======================================================================

println("finished training")
rbm_paired = cpu(rbm_paired)
println("back to cpu")

save_rbm(path_rbm, rbm_paired; overwrite = true)
println("Training complete. RBM saved to: $path_rbm")
println("LPL log saved to: $path_lpl")
