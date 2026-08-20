using HDF5
using RestrictedBoltzmannMachines: load_rbm, sample_v_from_v, sample_h_from_v, mean_h_from_v,
    free_energy, sample_from_inputs, Falses
using Statistics: mean, cor
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

println("Loading sequences..."); flush(stdout)
seqs_raw = load_sequences(FASTA_PATH)
seqs_onehot = LatentAlignedRBMs.onehot(seqs_raw)
n_samples = size(seqs_onehot, 3)
n_train = floor(Int, TRAIN_FRAC * n_samples)
n_val   = floor(Int, VAL_FRAC * n_samples)
perm = randperm(n_samples)
train_idx = perm[1:n_train]
val_idx   = perm[n_train+1:n_train+n_val]
println("Done loading. n_samples=$n_samples"); flush(stdout)

const N_TEST = 150
XA_data  = seqs_onehot[:, 1:SPLIT_SITE, train_idx[1:1000]]

path_rbm_A = "./results_pfam/rbm_A_N_HIDDEN_A=150_N_HIDDEN_B=100_H_ADD=20_K=50_N_ITERS=10000_REG=0_BS=256_LR=0.0001_DATA=PF00072_PF00512_l111.hdf5"
rbm_A = load_rbm(path_rbm_A)
println("Loaded rbm_A: q,n_vis,n_hid = ", size(rbm_A.w)); flush(stdout)

vh_scatter_r(hd, xd, hm, xm) = begin
    q, nv, _ = size(xd)
    hv_d = hd * reshape(Float64.(xd), q*nv, :)' / size(xd, 3)
    hv_m = hm * reshape(Float64.(xm), q*nv, :)' / size(xm, 3)
    cor(vec(hv_d), vec(hv_m))
end
v_scatter_r(xd, xm) = cor(vec(mean(reshape(Float64.(xd), :, size(xd,3)); dims=2)), vec(mean(reshape(Float64.(xm), :, size(xm,3)); dims=2)))
h_scatter_r(hd, hm) = cor(vec(mean(hd; dims=2)), vec(mean(hm; dims=2)))

println("Computing reference data-side <h>..."); flush(stdout)
hd_ref = Array(Float64.(sample_h_from_v(rbm_A, XA_data)))
println("Done."); flush(stdout)

function report(label, x_final)
    hm = Array(Float64.(sample_h_from_v(rbm_A, x_final)))
    r_h  = h_scatter_r(hd_ref, hm)
    r_v  = v_scatter_r(XA_data, x_final)
    r_hv = vh_scatter_r(hd_ref, XA_data, hm, x_final)
    println("[$label]  <h> r=$(round(r_h,digits=3))   <v> r=$(round(r_v,digits=3))   <hv> r=$(round(r_hv,digits=3))")
    flush(stdout)
end

# Extended cumulative-sweep checkpoints, continuing well past where the first
# diagnostic stopped (1000), to see whether the decline seen there (0.90 ->
# 0.81 by 1000 sweeps) keeps going down toward the ~0.07 seen in the real
# 1,000,000-sweep pipeline run, or plateaus at some higher floor (which would
# mean drift alone can't explain it).
const DELTAS = (1000, 2000, 3000, 5000, 10000, 10000)  # cumulative: 1000,3000,6000,11000,21000,31000

println("\n=== Cold start (bias-only marginal), extended budget ==="); flush(stdout)
x_cold = sample_from_inputs(rbm_A.visible, Falses(size(rbm_A.visible)..., N_TEST))
cum = 0
for d in DELTAS
    global x_cold = sample_v_from_v(rbm_A, x_cold; steps=d)
    global cum += d
    F = mean(Array(free_energy(rbm_A, x_cold)))
    println("  cumulative sweeps=$cum  mean F=$(round(F,digits=2))"); flush(stdout)
    report("cold, $cum sweeps", x_cold)
end

println("\nDONE"); flush(stdout)
