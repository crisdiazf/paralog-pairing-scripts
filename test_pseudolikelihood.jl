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
    
   # rbm_paired.hidden.par[2, 1:end] .= 1
end

#initialize!(rbm_paired, seq)
#zerosum!(rbm_paired)
#rescale_weights!(rbm_paired)



for i in 1:100
id=rand(1:23000, 256)
 lpl = mean(log_pseudolikelihood(rbm_paired, seq[:,:,id]))
println("lpl=$lpl")
end


