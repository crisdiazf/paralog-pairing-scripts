# ======================================================================
# Train paired RBM with partial freezing - GPU version with standardization
# ======================================================================
using JLD2
using HDF5
using Serialization
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: RBM, Potts,PottsGumbel, xReLU, initialize!, log_pseudolikelihood,
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
fraction = parse(Int, ARGS[1])
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

function load_split(filename)
    lines = readlines(filename)
    train_idx = parse.(Int, split(lines[1]))
    test_idx = parse.(Int, split(lines[2]))
    return train_idx, test_idx
end

trainset_name = "./misc/0072_1339_q=200_split_$(fraction)_$(real).txt"

# Load full datasets
seqsA_full = open("./misc/seqs_0072_nodup.bin", "r") do f
    deserialize(f)
end

seqsB_full = open("./misc/seqs_1339_nodup.bin", "r") do f
    deserialize(f)
end

seqsA = seqsA_full
seqsB = seqsB_full

# Build training concatenated sequences
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

initialize!(rbm_paired, seq)
zerosum!(rbm_paired)
rescale_weights!(rbm_paired)

# Store frozen parameters
rbm_copy = deepcopy(rbm_paired)

# Standardize and move to GPU
rbm_paired = gpu(standardize(rbm_paired))
seq_gpu = gpu(seq)

# ======================================================================
# Enhanced callback function that handles parameter freezing and hidden unit normalization
# ======================================================================

function callback_with_freeze(; rbm, iter, vm, vd, kwargs...)
    # Move to CPU temporarily to modify parameters
    rbm_cpu = cpu(rbm)
    rbm_copy_cpu = cpu(rbm_copy)
    
    # Project frozen parameters back to their initial values
    rbm_cpu.visible.par[:, :, 1:111] .= rbm_copy_cpu.visible.par[:, :, 1:111]
    rbm_cpu.visible.par[:, :, 112:end] .= rbm_copy_cpu.visible.par[:, :, 112:end]
    
    rbm_cpu.hidden.par[:, 1:100] .= rbm_copy_cpu.hidden.par[:, 1:100]
    rbm_cpu.hidden.par[:, 101:200] .= rbm_copy_cpu.hidden.par[:, 101:200]
    
    rbm_cpu.w[:, 1:111, 1:100] .= rbm_copy_cpu.w[:, 1:111, 1:100]
    rbm_cpu.w[:, 112:end, 101:200] .= rbm_copy_cpu.w[:, 112:end, 101:200]
    
    rbm_cpu.w[:, 1:111, 101:200] .= 0f0
    rbm_cpu.w[:, 112:end, 1:100] .= 0f0
    
    # Keep hidden unit parameters at 1 (xReLU normalization)
   # rbm_cpu.hidden.par[2, 1:end] .= 1
    
    # Move back to GPU
    rbm = gpu(standardize(rbm_cpu))
    
    # Compute and log pseudolikelihood every 50 iterations
    if iszero(iter % 50)
        # Compute LPL on CPU for logging
        vd_cpu = cpu(vd)
        rbm_cpu_for_lpl = cpu(rbm)
        lpl = mean(log_pseudolikelihood(rbm_cpu_for_lpl, vd_cpu))
        
        println("iter=$iter, lpl=$lpl")
        
        open(path_lpl, "a") do f
            println(f, lpl)
        end
        
        # Optionally save checkpoint
        if iter % 500 == 0
            path_rbm_t = joinpath(dir_results, base_name * "_$(iter).hdf5")
            save_rbm(path_rbm_t, rbm_cpu_for_lpl; overwrite = true)
        end
    end
    
    return nothing
end

# ======================================================================
# Training setup
# ======================================================================

# Prepare output filenames
base_name = "rbm_paired_PF00072_PF01339_q=200_fraction=$(fraction)_nadd=$(add_h)&k=$(k)&tt=$(ttf)&l2_weights=$(reg)_$(real)_learnrate0$(lrate)_gpu"
dir_results = joinpath(@__DIR__, "resultados/splitnew")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

# ======================================================================
# Train the RBM using standard pcd! with enhanced callback
# ======================================================================

batch = min(256, size(seq, 3))

println("batch = $batch")
println("total samples = $(size(seq, 3))")
println("iterations = $tt")
println("MCMC steps = $k")

# Initialize persistent visible units on GPU
vm_init = gpu(bitrand(size(rbm_paired.visible)..., batch))

@time pcd!(
    rbm_paired,
    seq_gpu;
    optim = Adam(Float32(10.0^(-lrate)), (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = batch,
    iters = tt,
    vm = vm_init,
    l2_weights = reg,
    ϵv = 1f-1,
    ϵh = 0f0,
    damping = 1f-1,
    callback = callback_with_freeze
)

# ======================================================================
# Save trained RBM (move to CPU first)
# ======================================================================

println("finished training")
rbm_paired = cpu(rbm_paired)
println("back to cpu")

save_rbm(path_rbm, rbm_paired; overwrite = true)
println("Training complete. RBM saved to: $path_rbm")
println("LPL log saved to: $path_lpl")
