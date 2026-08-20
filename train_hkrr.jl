using RestrictedBoltzmannMachines: pcd!, RBM, Potts, xReLU, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, free_energy, moments_from_samples,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches
using Random
using Optimisers: Adam
using RestrictedBoltzmannMachinesHDF5: save_rbm
using LatentAlignedRBMs
using FASTX
using BioSequences
using Statistics
using DelimitedFiles



L=parse(Int,ARGS[1])
k=parse(Int, ARGS[2])

batchsize = 256
reg=parse(Float32, ARGS[3])
tt=parse(Int, ARGS[4])



# Function to load sequences as LongAA (keeping gaps)
function load_sequences_with_gaps(fasta_file::String)
    reader = open(FASTX.FASTA.Reader, fasta_file)
    sequences = LongAA[]
    headers = String[]
    
    for record in reader
        # Use FASTX functions to access record components
        seq_str = String(FASTX.sequence(record))
        header_str = String(FASTX.identifier(record))
        # If you want the full header with description:
        # header_str = String(FASTX.description(record))
        
        push!(sequences, LongAA(seq_str))
        push!(headers, header_str)
    end
    
    close(reader)
    return headers, sequences
end



header,seq=load_sequences_with_gaps("./misc/Standard_HKRR_dataset.fasta")

seqs=LatentAlignedRBMs.onehot(seq)
seqsA=seqs[:,1:64,:]
seqsB=seqs[:,65:end,:]

if L == 112
    nh = 100
    train_x = seqsB
    name = "rr"
elseif L == 64
    nh = 100
    train_x = seqsA
    name = "hk"
end


rbm=RBM(Potts((21, L)), xReLU((nh,)), zeros(21, L, nh))
initialize!(rbm, train_x)


function callback(; rbm, iter, vm, vd, wd, kwargs...)
    if iszero(iter % 50)
        lpl = mean(log_pseudolikelihood(rbm, vd))
        println("iter=$iter, lpl=$lpl")
        open(path_lpl, "a") do f
            println(f, lpl)
        end
    end
end


base_name = "rbm_$(name)_k=$(k)&tt=$(tt)&l2l1_weights=$(reg)_learnrate04"
dir_results = joinpath(@__DIR__, "resultados")
isdir(dir_results) || mkdir(dir_results)

path_lpl = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm = joinpath(dir_results, base_name * ".hdf5")

@time pcd!(
           rbm,
           train_x;
           optim = Adam(1f-4, (0f0, 999f-3), 1f-6),
           steps = k,
           batchsize = 256,
           iters = tt,
           vm = bitrand(size(rbm.visible)..., 256),
           l2l1_weights = reg,
           callback
       )



#writedlm("./resultados/lpll_rbm_$(name)_k=$(k)_tt=$(iters)_reg=$(reg).txt", lpll)
save_rbm(path_rbm, rbm, overwrite=true)

