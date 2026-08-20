using CUDA
using HDF5
using RestrictedBoltzmannMachines: load_rbm, sample_v_from_v, sample_h_from_v, mean_h_from_v,
    free_energy, sample_from_inputs, Falses, gpu, cpu
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
println("Done loading. n_samples=$n_samples"); flush(stdout)

# Small footprint on purpose -- GPU is shared with the live v6 training job.
const N_TEST = 100
XA_data_cpu = seqs_onehot[:, 1:SPLIT_SITE, train_idx[1:500]]
XA_data = gpu(XA_data_cpu)

path_rbm_A = "./results_pfam/rbm_A_N_HIDDEN_A=150_N_HIDDEN_B=100_H_ADD=20_K=50_N_ITERS=10000_REG=0_BS=256_LR=0.0001_DATA=PF00072_PF00512_l111.hdf5"
rbm_A = gpu(load_rbm(path_rbm_A))
println("Loaded rbm_A onto GPU: q,n_vis,n_hid = ", size(rbm_A.w)); flush(stdout)

vh_scatter_r(hd, xd, hm, xm) = begin
    q, nv, _ = size(xd)
    hv_d = hd * reshape(Float64.(xd), q*nv, :)' / size(xd, 3)
    hv_m = hm * reshape(Float64.(xm), q*nv, :)' / size(xm, 3)
    cor(vec(hv_d), vec(hv_m))
end
v_scatter_r(xd, xm) = cor(vec(mean(reshape(Float64.(xd), :, size(xd,3)); dims=2)), vec(mean(reshape(Float64.(xm), :, size(xm,3)); dims=2)))
h_scatter_r(hd, hm) = cor(vec(mean(hd; dims=2)), vec(mean(hm; dims=2)))

println("Computing reference data-side <h> (GPU)..."); flush(stdout)
hd_ref = Array(Float64.(sample_h_from_v(rbm_A, XA_data)))
XA_data_c = Array(XA_data_cpu)
println("Done."); flush(stdout)

function report(label, x_final_gpu)
    hm = Array(Float64.(sample_h_from_v(rbm_A, x_final_gpu)))
    xm_c = Array(x_final_gpu)
    r_h  = h_scatter_r(hd_ref, hm)
    r_v  = v_scatter_r(XA_data_c, xm_c)
    r_hv = vh_scatter_r(hd_ref, XA_data_c, hm, xm_c)
    println("[$label]  <h> r=$(round(r_h,digits=3))   <v> r=$(round(r_v,digits=3))   <hv> r=$(round(r_hv,digits=3))")
    flush(stdout)
end

const DELTAS = (100, 300, 600, 3000, 6000, 10000)  # cumulative: 100,400,1000,4000,10000,20000

println("\n=== GPU, cold start (bias-only marginal) ==="); flush(stdout)
x_cold = sample_from_inputs(rbm_A.visible, Falses(size(rbm_A.visible)..., N_TEST))
cum = 0
for d in DELTAS
    global x_cold = sample_v_from_v(rbm_A, x_cold; steps=d)
    global cum += d
    F = mean(Array(free_energy(rbm_A, x_cold)))
    println("  cumulative sweeps=$cum  mean F=$(round(F,digits=2))"); flush(stdout)
    report("GPU cold, $cum sweeps", x_cold)
end

println("\nDONE"); flush(stdout)
