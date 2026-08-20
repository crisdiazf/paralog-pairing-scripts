import CUDA
import LatentAlignedRBMs
using FASTX
using HDF5
using RestrictedBoltzmannMachines: RBM,PottsGumbel, Potts, xReLU, log_pseudolikelihood, initialize!, pcd!, 
    sample_from_inputs, sample_v_from_v, free_energy, standardize, cpu, gpu, save_rbm
using Random
using Optimisers: Adam
using Statistics: mean, cov
#using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm 
using DelimitedFiles
using ProgressMeter: @showprogress
using BioSequences
using Serialization
using LinearAlgebra: norm


k=100
tt=parse(Int, ARGS[1])
reg=0.1
nh=100

t0=parse(Int,ARGS[2])

if t0!=0

	path=ARGS[3]
	tt=t0+tt
	rbm_og=load_rbm(path)
end


function load_sequences_and_identifiers(path)
    """Load FASTA file, returning identifiers and sequences as separate a0072ays."""
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

path="./.pfam/alignment_files/PF00072_filtered.fasta"

 _, sequences = load_sequences_and_identifiers(path)
seqsA=LatentAlignedRBMs.onehot(sequences)



function callback(; iter, _...)
    if iszero(iter % 500)
        Δt = @elapsed (lpl = mean(log_pseudolikelihood(cpu(rbm), cpu(seqsA))))
        @info "iter=$iter, lpl=$lpl, Δt=$Δt"

	open(path_lpl, "a") do f
            println(f, lpl)
        end

    end
end
     

rbm = RBM(PottsGumbel((21, 111)), xReLU((nh,)), zeros(21, 111, nh))

if t0!=0
	rbm=deepcopy(rbm_og)
end

initialize!(rbm, seqsA)
rbm = gpu(standardize(rbm))

base_name = "rbm_PF00072_k=$(k)&tt=$(tt)&l2l1_weights=$(reg)_nh=$(nh)_learnrate04"
dir_results = joinpath(@__DIR__, "resultados","PFAM")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")


@time pcd!(
        rbm, gpu(seqsA);
        optim=Adam(1f-4, (0f0, 999f-3), 1f-6), steps=k, batchsize=256, iters=tt,
        vm = gpu(bitrand(size(rbm.visible)..., 256)), l2l1_weights=reg,
        ϵv=1f-1, ϵh=0f0, damping=1f-1, # parameters controlling the hidden unit statistics normalization
        callback
    )

println("finished training")

rbm=cpu(rbm)

println("back to cpu")


save_rbm(path_rbm, rbm; overwrite = true)
