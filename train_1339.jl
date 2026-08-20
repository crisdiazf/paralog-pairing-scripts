import LatentAlignedRBMs
using RestrictedBoltzmannMachines: RBM, Potts, xReLU, log_pseudolikelihood, initialize!, pcd!, 
    sample_from_inputs, sample_v_from_v, free_energy
using StandardizedRestrictedBoltzmannMachines: standardize
using Random
using Optimisers: Adam
using Statistics: mean, cov
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm 
using DelimitedFiles
using ProgressMeter: @showprogress
using BioSequences
using Serialization
using LinearAlgebra: norm


k=100
tt=parse(Int, ARGS[1])
reg=0.1
t0=parse(Int, ARGS[2])

if t0!=0
	path=ARGS[3]
	rbm_og=load_rbm(path)
	tt=t0+tt
end



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

seqs=LatentAlignedRBMs.onehot(seqs)
seqsA=seqs[:,1:111,:]
seqsB=seqs[:,112:end,:]



function callback(; rbm, iter, vm, vd, wd, kwargs...)
    if iszero(iter % 50)
        lpl = mean(log_pseudolikelihood(rbm, vd))
        println("iter=$iter, lpl=$lpl")
        open(path_lpl, "a") do f
            println(f, lpl)
        end
    end
     if iszero(iter % 1000)
        base_name = "rbm_PF01339_k=$(k)&tt=$(iter)&l2l1_weights=$(reg)_learnrate04"
        dir_results = joinpath(@__DIR__, "resultados")
        isdir(dir_results) || mkdir(dir_results)
        path_rbm = joinpath(dir_results, base_name * ".hdf5")
        save_rbm(path_rbm, rbm; overwrite = true)
    end



end
     

rbm = RBM(Potts((21, 174)), xReLU((100,)), zeros(21, 174, 100))

if t0!=0
	rbm=deepcopy(rbm_og)
end

initialize!(rbm, seqsB)
#rbm = standardize(rbm)



base_name = "rbm_PF01339_k=$(k)&tt=$(tt)&l2l1_weights=$(reg)_learnrate04"
dir_results = joinpath(@__DIR__, "resultados")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")


@time pcd!(
           rbm,
           seqsB;
           optim = Adam(1f-4, (0f0, 999f-3), 1f-6),
           steps = k,
           batchsize = 256,
           iters = tt,
           vm = bitrand(size(rbm.visible)..., 256),
           l2l1_weights = reg,
           callback
       )



save_rbm(path_rbm, rbm; overwrite = true)
