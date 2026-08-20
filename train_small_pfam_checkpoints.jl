using CUDA
using JLD2
using HDF5
using Serialization
using RestrictedBoltzmannMachines: RBM, Potts, PottsGumbel, xReLU, initialize!,
    log_pseudolikelihood, sample_from_inputs, sample_v_from_v, free_energy,
    moments_from_samples, standardize, zerosum!, rescale_weights!, ∂free_energy,
    ∂regularize!, infinite_minibatches
using RestrictedBoltzmannMachines: load_rbm, save_rbm, unstandardize,
    StandardizedRBM, pcd!, sample_h_from_v, gpu, cpu
using LatentAlignedRBMs
using Optimisers: AbstractRule, Adam, setup, update!
using Random, Statistics, DelimitedFiles
using FASTX
using BioSequences
using LinearAlgebra

# =============================================================================
# CONFIG
# =============================================================================
path = "./PF00072_PF00512_paired.fasta"

path_rbm_A = "./rbm_goodcopy_PFAM00072_pairedttoPFAM00512_k=10&tt=500&l2l1_weights=0.0_learnrate03.hdf5"
path_rbm_B = "./rbm_PFAM00512_pairedttoPFAM00072_k=10&tt=4000&l2l1_weights=0.0_learnrate03.hdf5"

dir_results = "./resultados/julio/"
isdir(dir_results) || mkdir(dir_results)

domain_A = "PFAM00072"
domain_B = "PFAM00512"
domain_id = "$(domain_A)-$(domain_B)"

H_ADD = 10
k     = 100
tt    = 50000
reg   = 0.0

LOG_EVERY = 1000

base_name = "rbm_paired_$(domain_id)_hadd=$(H_ADD)_k=$(k)&tt=$(tt)&l2l1_weights=$(reg)_learnrate05"

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")
path_jld = joinpath(dir_results, base_name * "_training_info.jld2")

Random.seed!(42)

# =============================================================================
# LOAD DATA
# =============================================================================
function load_sequences_and_identifiers(path)
    """Load FASTA file, returning identifiers and sequences as separate arrays."""
    reader = open(FASTA.Reader, path)
    identifiers = String[]
    sequences = LongAA[]

    for record in reader
        push!(identifiers, FASTA.identifier(record))
        push!(sequences, LongAA(FASTA.sequence(record)))
    end

    close(reader)
    return identifiers, sequences
end

function load_sequences(path)
    """Load just the sequences from a FASTA file."""
    _, sequences = load_sequences_and_identifiers(path)
    return sequences
end

seqs = load_sequences(path)
seqs = LatentAlignedRBMs.onehot(seqs)

# =============================================================================
# LOAD PRETRAINED RBMs
# =============================================================================
rbm_A = load_rbm(path_rbm_A)
rbm_B = load_rbm(path_rbm_B)

n_vis_A = size(rbm_A.w, 2)
n_hid_A = size(rbm_A.w, 3)

n_vis_B = size(rbm_B.w, 2)
n_hid_B = size(rbm_B.w, 3)

n_vis_total = n_vis_A + n_vis_B
n_hid_total = n_hid_A + n_hid_B + H_ADD

@assert size(seqs, 2) == n_vis_total "sequence length does not match n_vis_A + n_vis_B"

# =============================================================================
# BUILD PAIRED RBM
# =============================================================================
rbm_paired = RBM(
    PottsGumbel((21, n_vis_total)),
    xReLU((n_hid_total,)),
    zeros(21, n_vis_total, n_hid_total),
)

initialize!(rbm_paired, seqs)
rbm_paired = standardize(rbm_paired)
rbm_paired=gpu(rbm_paired)

println("passed to gpu")


# =============================================================================
# PARAMETER PROJECTION
# =============================================================================
function project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)
    rbm_paired = unstandardize(rbm_paired)
    rbm_A = unstandardize(rbm_A)
    rbm_B = unstandardize(rbm_B)

    n_vis_A = size(rbm_A.w, 2)
    n_hid_A = size(rbm_A.w, 3)

    n_vis_B = size(rbm_B.w, 2)
    n_hid_B = size(rbm_B.w, 3)

    n_vis_total = n_vis_A + n_vis_B
    n_hid_total = n_hid_A + n_hid_B + h_add

    vis_A_range = 1:n_vis_A
    vis_B_range = (n_vis_A + 1):n_vis_total

    hid_A_range = 1:n_hid_A
    hid_B_range = (n_hid_A + 1):(n_hid_A + n_hid_B)
    hid_add_range = (n_hid_A + n_hid_B + 1):n_hid_total

    rbm_paired.visible.par[:, :, vis_A_range] .= rbm_A.visible.par
    rbm_paired.visible.par[:, :, vis_B_range] .= rbm_B.visible.par

    rbm_paired.hidden.par[:, hid_A_range] .= rbm_A.hidden.par
    rbm_paired.hidden.par[:, hid_B_range] .= rbm_B.hidden.par

    rbm_paired.w[:, vis_A_range, hid_A_range] .= rbm_A.w
    rbm_paired.w[:, vis_A_range, hid_B_range] .= 0

    rbm_paired.w[:, vis_B_range, hid_A_range] .= 0
    rbm_paired.w[:, vis_B_range, hid_B_range] .= rbm_B.w

    # Blocks involving hid_add_range are intentionally left trainable.
    rbm_paired = standardize(rbm_paired)

    return rbm_paired
end

println("before projecting")

rbm_paired = project_to_frozen_par!(cpu(rbm_paired), rbm_A, rbm_B, H_ADD)

println("after projecting")

# =============================================================================
# LOGGING
# =============================================================================
open(path_lpl, "w") do io
    println(io, "iter\tlog_pseudolikelihood")
end

lpl_trace = Float64[]
iter_trace = Int[]

# =============================================================================
# CALLBACK
# =============================================================================
function callback(; rbm, iter, vd, kwargs...)
    rbm = project_to_frozen_par!(cpu(rbm), rbm_A, rbm_B, H_ADD)
	#println("callback")
    if iszero(iter % LOG_EVERY)
        lpl = mean(log_pseudolikelihood(cpu(rbm), cpu(vd)))

        push!(iter_trace, iter)
        push!(lpl_trace, lpl)

        println("iter=$iter   log_pseudolikelihood=$lpl")

        open(path_lpl, "a") do io
            println(io, "$(iter)\t$(lpl)")
        end

        path_checkpoint = joinpath(
            dir_results,
            base_name * "_iter=$(iter).hdf5",
        )

        save_rbm(path_checkpoint, cpu(rbm))
    end

    return nothing
end

# =============================================================================
# TRAIN
# =============================================================================
initial_lpl = mean(log_pseudolikelihood(rbm_paired, seqs))
println("Initial log pseudolikelihood = $initial_lpl")

println("start training")

@time pcd!(
   gpu(rbm_paired),
    gpu(seqs);
    optim = Adam(1f-5, (0f0, 999f-3), 1f-6),
    steps = k,
    batchsize = 256,
    iters = tt,
    l2l1_weights = reg,
    ϵv = 1f-1,
    ϵh = 0f0,
    damping = 1f-1,
    callback,
)

rbm_paired = project_to_frozen_par!(cpu(rbm_paired), rbm_A, rbm_B, H_ADD)
final_lpl = mean(log_pseudolikelihood(rbm_paired, seqs))

rbm_paired = cpu(rbm_paired)


save_rbm(path_rbm, rbm_paired)

@save path_jld base_name domain_A domain_B H_ADD k tt reg LOG_EVERY path path_rbm_A path_rbm_B path_rbm path_lpl initial_lpl final_lpl iter_trace lpl_trace

println("Final log pseudolikelihood = $final_lpl")
println("Saved final RBM to: $path_rbm")
println("Saved log pseudolikelihood trace to: $path_lpl")
println("Saved training info to: $path_jld")
