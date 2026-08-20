
using RestrictedBoltzmannMachines: pcd!, RBM, Potts, xReLU, initialize!, log_pseudolikelihood,
    sample_from_inputs, sample_v_from_v, free_energy, moments_from_samples,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches
using RestrictedBoltzmannMachinesHDF5: save_rbm
using LatentAlignedRBMs
using FASTX
using BioSequences
using Statistics
using DelimitedFiles

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

L=parse(Int,ARGS[1])

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

k=parse(Int, ARGS[2])

batchsize = 256
iters = 30000
lpll=[]
reg=parse(Float32, ARGS[3])
@time pcd!(
    rbm, train_x; iters,steps=k, batchsize,l2l1_weights = reg,
    callback = function(; iter, _...)
        if iszero(iter % 50)
            lpl = mean(log_pseudolikelihood(rbm, train_x))
            println("$iter $lpl")
	    push!(lpll, lpl)
        end
    end
)

writedlm("./resultados/lpll_rbm_$(name)_k=$(k)_reg=$(reg)_tt=$(iters).txt", lpll)
save_rbm("./resultados/rbm_$(name)_k=$(k)_reg=$(reg)_tt=$(iters).hdf5", rbm, overwrite=true)
