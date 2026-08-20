import LatentAlignedRBMs
using RestrictedBoltzmannMachines: RBM, Potts, xReLU, log_pseudolikelihood, initialize!, pcd!, 
    sample_from_inputs, sample_v_from_v, free_energy, mean_h_from_v, Spin, sample_h_from_v, standardize
using Random
using Optimisers: Adam
using Statistics: mean, cov
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm 
using BioSequences
using DelimitedFiles

using Optimisers: AbstractRule
using Optimisers: Adam
using Optimisers: setup
using Optimisers: update!
using RestrictedBoltzmannMachines: ∂free_energy_h
using RestrictedBoltzmannMachines: ∂free_energy_v
using RestrictedBoltzmannMachines: ∂regularize!
using RestrictedBoltzmannMachines: infinite_minibatches
using RestrictedBoltzmannMachines: initialize_w!
using RestrictedBoltzmannMachines: initialize!
using RestrictedBoltzmannMachines: moments_from_samples
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: rescale_hidden_activations!
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: sample_h_from_h
using RestrictedBoltzmannMachines: sample_h_from_v
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: standardize_hidden_from_v!
using RestrictedBoltzmannMachines: standardize_visible_from_data!
using RestrictedBoltzmannMachines: StandardizedRBM
using RestrictedBoltzmannMachines: zerosum!
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: ∂RBM

v = LatentAlignedRBMs.PF00072_PF01339_paired_20250416_load_sequences()
seen = Dict{LongAA, Int}()
keep = Int[]

for (i, seq) in enumerate(v)
    if !haskey(seen, seq)
        seen[seq] = i
        push!(keep, i)
    end
end

all_indices = collect(1:length(v))
duplicate_indices = setdiff(all_indices, keep)
v_unique = v[keep]

seqs=v_unique

println("after")
seqs=LatentAlignedRBMs.onehot(seqs)
seqsA=seqs[:,1:111,:]
seqsB=seqs[:,112:end,:]

path="./sims/rbm_0072&relu=100&k=50&tt=50000.hdf5"
rbmA=load_rbm(path)

teacher_rbm=deepcopy(rbmA)
teacher_samples_v=Float64.(seqsB)
teacher_samples_h = sample_h_from_v(teacher_rbm, seqsA)

function alnpcdL2!(
    rbm::RBM,
    data_v::AbstractArray,
    data_h::AbstractArray,
    λ::Real = 0.5;

    batchsize::Int = 1,
    iters::Int = 1,
    steps::Int = 1,
    optim::AbstractRule = Adam(),

    moments_v = moments_from_samples(rbm.visible, data_v),
    moments_h = moments_from_samples(rbm.hidden, data_h),

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    zerosum::Bool = true,

    callback = Returns(nothing),

    vm = sample_from_inputs(rbm.visible, zeros(size(rbm.visible)..., batchsize)),
    hm = sample_from_inputs(rbm.hidden, zeros(size(rbm.hidden)..., batchsize)),

    shuffle::Bool = true,

    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps) # initialize optimiser state
)
    @assert size(data_v) == (size(rbm.visible)..., size(data_v)[end])
    @assert size(data_h) == (size(rbm.hidden)..., size(data_h)[end])

    zerosum && zerosum!(rbm)

    for (iter, (vd,), (hd,)) in zip(1:iters, infinite_minibatches(data_v; batchsize, shuffle), infinite_minibatches(data_h; batchsize, shuffle))
             
        ∂d_v = ∂free_energy_v(rbm, vd; moments = moments_v)
        ∂d_h = ∂free_energy_h(rbm, hd; moments = moments_h)
        ∂m_v = ∂free_energy_v(rbm, vm)
        ∂m_h = ∂free_energy_h(rbm, hm)

        ∂_v = ∂d_v - ∂m_v
        ∂_h = ∂d_h - ∂m_h

        # Create ∂_h_squared with the correct field order
        ∂_h_squared = ∂RBM(
            ∂_h.visible .^ 2,  # first: visible
            ∂_h.hidden .^ 2,   # second: hidden
            ∂_h.w .^ 2         # third: w
        )

        # Then compute your expression
        ∂ = (1 - λ) * ∂_v + λ * ∂_h_squared

        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        zerosum && zerosum!(rbm)

        callback(; rbm, optim, state, iter, vm, hm, vd, hd)
    end
    return state, ps
end

function train_student(rbm)
    function callback(; rbm, optim, state, iter, vm, hm, vd, hd)
            lpl = mean(log_pseudolikelihood(rbm, seqsB))
            println("iter=$iter, lpl=$lpl")  
            #writedlm("resultados/lpl_1339_L2_&xrelu=100&k=50&tt=50000.txt", lpl)
    end
        
	alnpcdL2!(
		rbm,
		teacher_samples_v,
		teacher_samples_h,
		0.5; optim=Adam(1f-4, (0f0, 999f-3), 1f-6), steps=50, batchsize=256,l2l1_weights=0.1,
        iters = 5,
        callback = callback
	)
	return rbm
end


rbm = RBM(Potts((21, 174)), xReLU((100,)), zeros(21, 174, 100));
#rbm=standardize(rbm)
student_rbm = train_student(rbm);


direccion1=joinpath(@__DIR__,"resultados/rbm_1339_L2_xrelu=100&k=50&tt=50000.hdf5")
save_rbm(direccion1, rbm; overwrite=true)


