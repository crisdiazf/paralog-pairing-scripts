# ======================================================================
# Train paired RBM - NULL MODEL (no transfer learning, from scratch)
# ======================================================================
using JLD2
using HDF5
using Serialization
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: RBM, Potts, PottsGumbel, xReLU, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, pcd!, standardize, cpu, gpu, save_rbm
using LatentAlignedRBMs
using Optimisers: Adam
using Random, Statistics, DelimitedFiles
using CUDA
using ProgressMeter: @showprogress

# ======================================================================
# Parse command line arguments
# ======================================================================
fraction = parse(Int, ARGS[1])      # fraction of data to use (unused but kept for compatibility)
add_h = parse(Int, ARGS[2])         # number of hidden units
real = parse(Int, ARGS[3])          # random seed / realization number
tt = parse(Int, ARGS[4])            # training iterations
reg = parse(Float32, ARGS[5])       # regularization
keeptrain = parse(Bool, ARGS[6])    # unused in null model (kept for compatibility)
lrate = parse(Int, ARGS[7])         # learning rate exponent

println("=== NULL MODEL: Training RBM from scratch with $add_h hidden units ===")
println("fraction = $fraction, add_h = $add_h, real = $real, tt = $tt, reg = $reg, lrate = $lrate")

# ======================================================================
# Load paired sequences
# ======================================================================
println("Loading sequences...")

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

trainset_name="./misc/0072_1339_q=200_split_$(fraction)_$(real).txt"

idx_train,_=load_split(trainset_name)

seqsA=seqsA_full[:,:,idx_train]
seqsB=seqsB_full[:,:,idx_train]


#Build training concatenated sequences
seq = zeros(Int,
    size(seqsA, 1),
    size(seqsA, 2) + size(seqsB, 2),
    size(seqsA, 3)
)


seq[:, 1:size(seqsA, 2), :] .= seqsA
seq[:, size(seqsA, 2)+1:end, :] .= seqsB

println("Data loaded. Total sequences: $(size(seq, 3))")
println("Visible units: $(size(seq, 2)) (111 + 174)")
println("Hidden units: $add_h")

# ======================================================================
# Initialize RBM from scratch
# ======================================================================
println("Initializing RBM from scratch...")

k = 50  # MCMC steps per update

# Create RBM with Potts visible (21 states) and xReLU hidden
rbm = RBM(PottsGumbel((21, 111+174)), xReLU((add_h,)), zeros(21, 111+174, add_h))

# Initialize parameters based on data statistics
initialize!(rbm, seq)
println("Initialization complete")

# Standardize and move to GPU
println("Standardizing and moving to GPU...")
rbm = gpu(standardize(rbm))
seq_gpu = gpu(seq)
println("RBM on GPU: ", typeof(rbm))
println("Data on GPU: ", typeof(seq_gpu))

# ======================================================================
# Setup output paths
# ======================================================================
base_name = "rbm_paired_null_PF00072_PF01339_q=200_fraction=$(fraction)_nh=$(add_h)&k=$(k)&tt=$(tt)&l2l1_weights=$(reg)_real=$(real)_learnrate0$(lrate)_gpu"
dir_results = joinpath(@__DIR__, "resultados/null_models_notall")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

println("Output files:")
println("  RBM: $path_rbm")
println("  LPL log: $path_lpl")

# ======================================================================
# Callback function for logging
# ======================================================================
function callback(; rbm, iter, vm, vd, kwargs...)
    if iszero(iter % 50)
        # Move to CPU for logging
        rbm_cpu = cpu(rbm)
        vd_cpu = cpu(vd)
        
        # Compute pseudolikelihood
        lpl = mean(log_pseudolikelihood(rbm_cpu, vd_cpu))
        
        println("iter=$iter, lpl=$lpl")
        
        # Save to log file
        open(path_lpl, "a") do f
            println(f, lpl)
        end
        
    end
    return nothing
end

# ======================================================================
# Train the RBM using standard pcd!
# ======================================================================
batch = min(256, size(seq, 3))

println("\n=== Training Configuration ===")
println("  Batch size: $batch")
println("  Total samples: $(size(seq, 3))")
println("  Iterations: $tt")
println("  MCMC steps per iteration: $k")
println("  Learning rate: 1e-$lrate")
println("  Regularization (l2l1_weights): $reg")
println("  Device: GPU (CUDA)")
println("================================\n")

# Initialize persistent visible units on GPU
vm_init = gpu(bitrand(size(rbm.visible)..., batch))

@time pcd!(
    rbm,
    seq_gpu;
    optim = Adam(Float32(10.0^(-lrate)), (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = batch,
    iters = tt,
    vm = vm_init,
    l2l1_weights = reg,
    ϵv = 1f-1,
    ϵh = 0f0,
    damping = 1f-1,
    callback = callback
)

# ======================================================================
# Save trained RBM
# ======================================================================
println("\n=== Training Complete ===")
println("Moving RBM to CPU for saving...")
rbm = cpu(rbm)
println("Saving RBM...")
save_rbm(path_rbm, rbm; overwrite = true)
println("RBM saved to: $path_rbm")
println("LPL log saved to: $path_lpl")
println("========================")
