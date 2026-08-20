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


k=parse(Int,ARGS[1])
tt=parse(Int, ARGS[2])
reg=parse(Float32,ARGS[3])
nh=parse(Int, ARGS[4])


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

path="./.pfam/alignment_files/PF00512_filtered.fasta"

 _, sequences = load_sequences_and_identifiers(path)
seqsB=LatentAlignedRBMs.onehot(sequences)


function callback(; iter, _...)
    if iszero(iter % 500)
        Δt = @elapsed (lpl = mean(log_pseudolikelihood(cpu(rbm), cpu(seqsB))))
        @info "iter=$iter, lpl=$lpl, Δt=$Δt"

	open(path_lpl, "a") do f
            println(f, lpl)
        end

    end
end
     

rbm = RBM(PottsGumbel((21, 66)), xReLU((nh,)), zeros(21, 66, nh))

initialize!(rbm, seqsB)
rbm = gpu(standardize(rbm))

base_name = "rbm_PF00512_k=$(k)&tt=$(tt)&l2l1_weights=$(reg)_learnrate04_nh=$(nh)"
dir_results = joinpath(@__DIR__, "resultados","PFAM")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")


@time pcd!(
        rbm, gpu(seqsB);
        optim=Adam(1f-4, (0f0, 999f-3), 1f-6), steps=k, batchsize=256, iters=tt,
        vm = gpu(bitrand(size(rbm.visible)..., 256)), l2l1_weights=reg,
        ϵv=1f-1, ϵh=0f0, damping=1f-1, # parameters controlling the hidden unit statistics normalization
        callback
    )

println("finished training")

rbm=cpu(rbm)

println("back to cpu")


save_rbm(path_rbm, rbm; overwrite = true)
