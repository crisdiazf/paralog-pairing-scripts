
# ======================================================================
# Train paired RBM with partial freezing
# ======================================================================
using JLD2
using Serialization
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: RBM, Potts, xReLU, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, free_energy, moments_from_samples,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches
using StandardizedRestrictedBoltzmannMachines: StandardizedRBM
using LatentAlignedRBMs
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm
using Optimisers: AbstractRule, Adam, setup, update!
using Random, Statistics, DelimitedFiles

# ======================================================================
# Load pretrained RBMs
# ======================================================================
fraction=parse(Int,ARGS[1])
add_h = parse(Int,ARGS[2])  # number of new hidden units
real=parse(Int,ARGS[3])
tt  = parse(Int, ARGS[4])   # training iterations
reg = parse(Float32,ARGS[5])     # regularization
keeptrain=parse(Bool,ARGS[6])
lrate=parse(Int,ARGS[7])

tt0=0
k   = 50     # MCMC steps per update

if keeptrain
tt0=10000
filename_og="./resultados/rbm_paired_PF00072_PF01339_q=200_fraction=$(fraction)_nadd=$(add_h)&k=50&tt=2000&l2l1_weights=$(reg)_$(real).hdf5"
rbm_og=load_rbm(filename_og)
end

ttf=tt0+tt

filename_previous="./resultados/rbm_paired_PF00072_PF01339_q=200_fraction=$(fraction)_nadd=$(add_h)&k=50&tt=2000&l2l1_weights=$(reg)_$(real).hdf5"

filename_teacher="./misc/rbm_0072&relu=100&k=50&tt=50000"
filename_student="./misc/rbm_1339&relu=100&k=50&tt=50000"

path_teacher="$filename_teacher.hdf5"
path_student="$filename_student.hdf5"

rbm0072=load_rbm(path_teacher)
rbm1339=load_rbm(path_student)

#rbm0072=StandardizedRestrictedBoltzmannMachines.unstandardize(rbm0072)
#rbm1339=StandardizedRestrictedBoltzmannMachines.unstandardize(rbm1339)

zerosum!(rbm0072)
zerosum!(rbm1339)

# ======================================================================
# Load paired sequences and prepare one-hot representations
# ======================================================================

function load_split(filename)
    lines = readlines(filename)
    train_idx = parse.(Int, split(lines[1]))
    test_idx  = parse.(Int, split(lines[2]))
    return train_idx, test_idx
end

trainset_name="./misc/0072_1339_q=200_split_$(fraction)_$(real).txt"

# Load full datasets
seqsA_full = open("./misc/seqs_0072_nodup.bin", "r") do f
    deserialize(f)
end

seqsB_full = open("./misc/seqs_1339_nodup.bin", "r") do f
    deserialize(f)
end

idx_train,_=load_split(trainset_name)

seqsA=seqsA_full[:,:,idx_train]
seqsB=seqsB_full[:,:,idx_train]

#seqsA=seqsA_full
#seqsB=seqsB_full


#Build training concatenated sequences
seq = zeros(Int,
    size(seqsA, 1),
    size(seqsA, 2) + size(seqsB, 2),
    size(seqsA, 3)
)

seq[:, 1:size(seqsA, 2), :] .= seqsA
seq[:, size(seqsA, 2)+1:end, :] .= seqsB

#seq=hcat(seqsA_full, seqsB_full)

# ======================================================================
# Define and initialize paired RBM
# ======================================================================

#path="./rbm_paired_PF00072_PF00486&nadd=30&k=50&tt=$(tt0)&l2l1_weights=0.1.hdf5"
rbm_paired = RBM(Potts((21, 111+174)), xReLU((200 + add_h,)), zeros(21, 111+174, 200 + add_h))
if keeptrain 
rbm_paired=deepcopy(rbm_og)
end
#rbm_paired =load_rbm(path)

if keeptrain!=true
initialize!(rbm_paired, seq)

# Copy pretrained parameters
rbm_paired.visible.par[:, :, 1:111]  .= rbm0072.visible.par
rbm_paired.visible.par[:, :, 112:end] .= rbm1339.visible.par

rbm_paired.hidden.par[:, 1:100] .= rbm0072.hidden.par
rbm_paired.hidden.par[:, 101:200] .= rbm1339.hidden.par

rbm_paired.w[:, 1:111, 1:100] .= rbm0072.w
rbm_paired.w[:, 112:end, 101:200] .= rbm1339.w


rbm_paired.w[:, 1:111, 101:200] .= 0f0
rbm_paired.w[:, 112:end, 1:100] .= 0f0


rbm_paired=StandardizedRestrictedBoltzmannMachines.StandardizedRBM(rbm_paired)
end


zerosum!(rbm_paired)
#rescale_weights!(rbm_paired)
rbm_copy=deepcopy(rbm_paired)


# ======================================================================
# Define pcd_freeze! function
# ======================================================================

function pcd_freeze!(
    rbm::StandardizedRBM,
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
    damping::Real = 1//100, ϵv::Real = 0, ϵh::Real = 0,
    rescale_hidden::Bool = true,
    zerosum::Bool = true,
    rescale::Bool = true,
    callback = Returns(nothing),
    vm = sample_from_inputs(rbm.visible, zeros(size(rbm.visible)..., batchsize)),
    shuffle::Bool = true,
    freeze_mask = nothing
)
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)
    moments = moments_from_samples(rbm.visible, data; wts)
    wts_mean = isnothing(wts) ? 1 : mean(wts)

    zerosum && zerosum!(rbm)
    #rescale && rescale_weights!(rbm)

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

	#rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)


	rbm.visible.par[:, :, 1:111]  .= rbm_copy.visible.par[:, :, 1:111]
        rbm.visible.par[:, :, 112:end] .= rbm_copy.visible.par[:, :, 112:end]

        rbm.hidden.par[:, 1:100] .= rbm_copy.hidden.par[:, 1:100]
        rbm.hidden.par[:, 101:200] .= rbm_copy.hidden.par[:, 101:200]

        rbm.w[:, 1:111, 1:100] .= rbm_copy.w[:, 1:111, 1:100]
        rbm.w[:, 112:end, 101:200] .= rbm_copy.w[:, 112:end, 101:200]

        rbm.w[:, 1:111, 101:200] .= 0f0
        rbm.w[:, 112:end, 1:100] .= 0f0

	#rbm.hidden.par[2,201:end].=0.5

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

# Prepare output filenames
base_name = "rbm_paired_PF00072_PF01339_q=200_fraction=$(fraction)_nadd=$(add_h)&k=$(k)&tt=$(ttf)&l2l1_weights=$(reg)_$(real)_learnrate0$(lrate)"
dir_results = joinpath(@__DIR__, "resultados/nomask/stand")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

# ======================================================================
# Callback function
# ======================================================================

function callback(; rbm, iter, vm, vd, wd, kwargs...)
    if iszero(iter % 50)
	path_rbm_t = joinpath(dir_results, base_name * "_$(iter).hdf5")
	#save_rbm(path_rbm_t,rbm; overwrite = true)
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
batch=256

if size(seq,3) < 256
batch=size(seq,3)
end

println(batch)
println(size(seq,3))

@time pcd_freeze!(
    rbm_paired,
    seq;
    optim = Adam(Float32(10.0^(-lrate)), (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = batch,
    iters = tt,
    vm = bitrand(size(rbm_paired.visible)..., batch),
    l2l1_weights = reg,
    ϵv=1f-1, ϵh=0f0, damping=1f-1, rescale_hidden=false,
    freeze_mask = nothing,
    callback
)

# ======================================================================
# Save trained RBM
# ======================================================================

save_rbm(path_rbm, rbm_paired; overwrite = true)
println("Training complete. RBM saved to: $path_rbm")
println("LPL log saved to: $path_lpl")

#rbm=deepcopy(rbm_paired)
#nrows, ncols = 10,10
#nsteps = 200
#fantasy_F = zeros(nrows*ncols, nsteps)
#fantasy_x = bitrand(21, 111+174, nrows*ncols)
#fantasy_F[:,1] .= free_energy(rbm, fantasy_x)
#@time for t in 2:nsteps
#    println("Step $t")
#    fantasy_x .= sample_v_from_v(rbm, fantasy_x,steps=50)
#    fantasy_F[:,t] .= free_energy(rbm, fantasy_x)
#end

#path_fantasy = joinpath(dir_results, "fantasy_nsteps=$(nsteps)_" * base_name * ".jld2")
#@save path_fantasy fantasy_x fantasy_F nsteps

