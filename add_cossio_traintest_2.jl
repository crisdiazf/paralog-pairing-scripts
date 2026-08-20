# ======================================================================
# Train paired RBM with partial freezing
# ======================================================================

using JLD2
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: RBM, Potts, xReLU, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, free_energy, moments_from_samples,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches
using StandardizedRestrictedBoltzmannMachines: standardize
using LatentAlignedRBMs
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm
using Optimisers: AbstractRule, Adam, setup, update!
using Random, Statistics, DelimitedFiles

# ======================================================================
# Load pretrained RBMs
# ======================================================================

rbm0072 = jldopen("rbm_std_PF00072.jld2", "r") do f
    f["rbm"]
end

rbm0486 = jldopen("rbm_std_PF00486.jld2", "r") do f
    f["rbm"]
end

# ======================================================================
# Load paired sequences and prepare one-hot representations
# ======================================================================

v0 = LatentAlignedRBMs.PF00072_PF00486_paired_20250623_load_sequences()
v  = LatentAlignedRBMs.PF00072_PF00486_paired_20250623_load_sequences_split()

seq  = LatentAlignedRBMs.onehot(v0)
seqA = LatentAlignedRBMs.onehot(v.PF00072)
seqB = LatentAlignedRBMs.onehot(v.PF00486)

# ======================================================================
# Define and initialize paired RBM
# ======================================================================

add_h = 30  # number of new hidden units
tt0=60000
path="./rbm_paired_PF00072_PF00486&nadd=30&k=50&tt=$(tt0)&l2l1_weights=0.1.hdf5"
#rbm_paired = RBM(Potts((21, 188)), xReLU((600 + add_h,)), zeros(21, 188, 600 + add_h))
rbm_paired =load_rbm(path)

initialize!(rbm_paired, seq)

# Copy pretrained parameters
#rbm_paired.visible.par[:, :, 1:111]  .= rbm0072.visible.par
#rbm_paired.visible.par[:, :, 112:end] .= rbm0486.visible.par

#rbm_paired.hidden.par[:, 1:300] .= rbm0072.hidden.par
#rbm_paired.hidden.par[:, 301:600] .= rbm0486.hidden.par

#rbm_paired.w[:, 1:111, 1:300] .= rbm0072.w
#rbm_paired.w[:, 112:end, 301:600] .= rbm0486.w

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
# Freeze mask: freeze old parameters, train only new hidden units
# ======================================================================

fm = (
    visible = zeros(Float32, size(rbm_paired.visible.par)),
    hidden  = zeros(Float32, size(rbm_paired.hidden.par)),
    w       = zeros(Float32, size(rbm_paired.w))
)
fm.hidden[:, 601:end] .= 1f0
fm.w[:, :, 601:end]   .= 1f0

# ======================================================================
# Training setup
# ======================================================================

k   = 50     # MCMC steps per update
tt  = 30000   # training iterations
reg = 0.1     # regularization

ttf=tt0+tt

# Prepare output filenames
base_name = "rbm_paired_PF00072_PF00486&nadd=$(add_h)&k=$(k)&tt=$(ttf)&l2l1_weights=$(reg)"
dir_results = joinpath(@__DIR__, "resultados")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

# ======================================================================
# Callback function
# ======================================================================

function callback(; rbm, iter, vm, vd, wd, kwargs...)
    if iszero(iter % 100)
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

pcd_freeze!(
    rbm_paired,
    seq;
    optim = Adam(1f-4, (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = 256,
    iters = tt,
    vm = bitrand(size(rbm_paired.visible)..., 256),
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

