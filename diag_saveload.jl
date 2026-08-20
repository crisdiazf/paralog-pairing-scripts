using HDF5
using RestrictedBoltzmannMachines: load_rbm, log_pseudolikelihood, sample_h_from_v, mean_h_from_v, gpu, cpu
using Statistics: mean, cor, std
using Random
using LatentAlignedRBMs
using FASTX
using BioSequences

Random.seed!(42)

const FASTA_PATH = "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE = 111
const TRAIN_FRAC = 0.7
const VAL_FRAC   = 0.15

function load_sequences(path)
    reader = open(FASTA.Reader, path)
    sequences = LongAA[]
    for record in reader
        push!(sequences, LongAA(FASTA.sequence(record)))
    end
    close(reader)
    return sequences
end

seqs_raw = load_sequences(FASTA_PATH)
seqs_onehot = LatentAlignedRBMs.onehot(seqs_raw)
n_samples = size(seqs_onehot, 3)
n_train = floor(Int, TRAIN_FRAC * n_samples)
n_val   = floor(Int, VAL_FRAC * n_samples)
perm = randperm(n_samples)
train_idx = perm[1:n_train]
val_idx   = perm[n_train+1:n_train+n_val]

# small batch, same convention as training's minibatch check
XA_train_batch = seqs_onehot[:, 1:SPLIT_SITE, train_idx[1:2000]]
XA_val_batch   = seqs_onehot[:, 1:SPLIT_SITE, val_idx[1:2000]]

path_rbm_A = "./results_pfam/rbm_A_N_HIDDEN_A=150_N_HIDDEN_B=100_H_ADD=20_K=50_N_ITERS=10000_REG=0_BS=256_LR=0.0001_DATA=PF00072_PF00512_l111.hdf5"

println("Loading rbm_A from disk...")
rbm_A = load_rbm(path_rbm_A)
println("Loaded. Type: ", typeof(rbm_A))
println("q,n_vis,n_hid from w: ", size(rbm_A.w))

lpl_train = mean(log_pseudolikelihood(rbm_A, XA_train_batch))
lpl_val   = mean(log_pseudolikelihood(rbm_A, XA_val_batch))
println("Reloaded-model lpl on train batch: ", lpl_train)
println("Reloaded-model lpl on val   batch: ", lpl_val)
println()
println("(Training's own final logged values were train=-1.184, val=-1.2577 -- 'iter=10000' checkpoint)")

# Also check mean-field <h> spread -- a collapsed/broken model would show ~0 variance across units
hmf = mean_h_from_v(rbm_A, XA_train_batch)
println("\nmean_h_from_v on train batch: shape=", size(hmf), " mean=", mean(hmf), " std=", std(hmf))
