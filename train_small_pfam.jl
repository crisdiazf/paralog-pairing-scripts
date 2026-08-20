using HDF5
using CUDA
using RestrictedBoltzmannMachines: RBM, Potts, PottsGumbel,xReLU, initialize!, log_pseudolikelihood,
    sample_v_from_v, free_energy, moments_from_samples, standardize,
    zerosum!, rescale_weights!, ∂free_energy, ∂regularize!, infinite_minibatches
using RestrictedBoltzmannMachines: load_rbm, save_rbm, unstandardize, StandardizedRBM,
    pcd!, sample_h_from_v, cpu, gpu
using LatentAlignedRBMs
using Optimisers: Adam
using Random, Statistics, DelimitedFiles
using FASTX
using BioSequences
using LinearAlgebra
using Plots
gr()
ENV["GKSwstype"] = "100"

# =============================================================================
# LOAD DATA — only the PF00072 part (first 111 positions)
# =============================================================================
function load_sequences(path)
    reader = open(FASTA.Reader, path)
    sequences = LongAA[]
    for record in reader
        push!(sequences, LongAA(FASTA.sequence(record)))
    end
    close(reader)
    return sequences
end

path     = "./PF00072_PF00512_paired.fasta"
l        = 66
seqs_raw = load_sequences(path)
seqs     = LatentAlignedRBMs.onehot(seqs_raw)[:, 112:end, :]

# =============================================================================
# MODEL SETUP
# =============================================================================
rbm = RBM(Potts((21, l)), xReLU((50,)), zeros(21,l, 50))
initialize!(rbm, seqs)
rbm = standardize(rbm)

#rbm=gpu(rbm)

# =============================================================================
# TRAINING
# =============================================================================
k   = 30
tt  = 6000
reg = 0.0

domain_id   = "PFAM00512"
dir_results = "./resultados_pfam_cristina/"
isdir(dir_results) || mkdir(dir_results)

base_name = "rbm_$(domain_id)_pairedtoPFAM00072_k=$(k)_tt=$(tt)_l2l1=$(reg)"
path_lpl  = joinpath(dir_results, "lpl_" * base_name * ".txt")
path_rbm  = joinpath(dir_results, base_name * ".hdf5")

function callback(; rbm, iter, vm, vd, kwargs...)
    if iszero(iter % 50)
        lpl = mean(log_pseudolikelihood(cpu(rbm), cpu(vd)))
        println("iter=$iter, lpl=$lpl")
        open(path_lpl, "a") do f
            println(f, lpl)
        end
    end
end

println("=== Training ===")
@time pcd!(
    rbm, seqs;
    optim        = Adam(1f-4, (0f0, 999f-3), 1f-6),
    steps        = k,
    batchsize    = 256,
    iters        = tt,
    vm           = bitrand(size(rbm.visible)..., 256),
    l2l1_weights = reg,
    ϵv=1f-1, ϵh=0f0, damping=1f-1, rescale_hidden=false,
    callback
)


#rbm=cpu(rbm)
save_rbm(path_rbm, rbm; overwrite = true)
println("RBM saved to: $path_rbm")

# =============================================================================
# SAMPLING + FREE ENERGY
# =============================================================================
nrows, ncols = 50, 50
nsteps       = 500
n_chains     = nrows * ncols

fantasy_x = bitrand(21, l, n_chains)
fantasy_F = zeros(n_chains, nsteps)
fantasy_F[:, 1] .= free_energy(rbm, fantasy_x)

println("=== Sampling ===")
@time for t in 2:nsteps
    println("Step $t")
    fantasy_x .= sample_v_from_v(rbm, fantasy_x, steps=1)
    fantasy_F[:, t] .= free_energy(rbm, fantasy_x)
end

path_samples = joinpath(dir_results, "samples_" * base_name * ".hdf5")
h5open(path_samples, "w") do f
    f["fantasy_x"] = Array{Float32}(fantasy_x)
    f["fantasy_F"] = fantasy_F
end
println("Samples saved to: $path_samples")

# =============================================================================
# METRICS
# =============================================================================
original_data  = seqs
sampled_data   = fantasy_x
L              = size(original_data, 2)
n_samples_orig = size(original_data, 3)
n_samples_samp = size(fantasy_x, 3)

# 1. Visible means
mdata    = moments_from_samples(rbm.visible, original_data)[:]
msamples = moments_from_samples(rbm.visible, fantasy_x)[:]
ρ_v      = cor(mdata, msamples)

# 2. Hidden activations
h_data  = mean(sample_h_from_v(rbm, original_data), dims=2)
h_model = mean(sample_h_from_v(rbm, fantasy_x),     dims=2)
ρ_h     = cor(vec(h_data), vec(h_model))

# 3. Hidden-visible correlations
hv_data  = (reshape(original_data, 21 * L, n_samples_orig) * sample_h_from_v(rbm, original_data)') / n_samples_orig
hv_model = (reshape(fantasy_x,     21 * L, n_samples_samp) * sample_h_from_v(rbm, fantasy_x)')    / n_samples_samp
ρ_vh     = cor(vec(hv_data), vec(hv_model))

# 4. Pairwise covariance (second moment)
orig_flat = reshape(original_data, 21 * L, n_samples_orig)
samp_flat = reshape(sampled_data,  21 * L, n_samples_samp)
corr_orig = cov(orig_flat, dims=2)
corr_samp = cov(samp_flat, dims=2)
mask      = triu(trues(size(corr_orig)), 1)
orig_vec  = corr_orig[mask]
samp_vec  = corr_samp[mask]
ρ_c       = cor(orig_vec, samp_vec)

println("\n" * "="^50)
println("SUMMARY — $domain_id")
println("="^50)
println("1. Visible means          ρ = $(round(ρ_v,  digits=4))")
println("2. Hidden activations     ρ = $(round(ρ_h,  digits=4))")
println("3. Hidden-visible corr.   ρ = $(round(ρ_vh, digits=4))")
println("4. Pairwise covariance    ρ = $(round(ρ_c,  digits=4))")
println("="^50)

# =============================================================================
# PLOTS (saved to disk, not displayed)
# =============================================================================
output_dir = joinpath(dir_results, "plots_" * base_name)
isdir(output_dir) || mkdir(output_dir)

function compact_scatter(xdata, ydata, xlabel, ylabel, title, color)
    ρ = cor(vec(xdata), vec(ydata))
    p = Plots.scatter(vec(xdata), vec(ydata),
        xlabel=xlabel, ylabel=ylabel, title=title,
        alpha=0.5, markersize=4, label="",
        color=color, framestyle=:box, grid=false,
        aspect_ratio=:equal,
        tickfontsize=11, guidefontsize=13, titlefontsize=14)
    Plots.plot!(p, identity, linestyle=:dash, color=:red, linewidth=1.5, label="")
    Plots.annotate!(p, Plots.xlims(p)[2] * 0.3, Plots.ylims(p)[2] * 0.9,
        Plots.text("ρ = $(round(ρ, digits=4))", :black, 13))
    return p
end

p_h  = compact_scatter(h_data,       h_model,       "<h> data",  "<h> model",  "Hidden Activations",   :blue)
p_v  = compact_scatter(mdata,        msamples,      "<v> data",  "<v> model",  "Visible Means",        :green)
p_vh = compact_scatter(vec(hv_data), vec(hv_model), "<vh> data", "<vh> model", "Hidden-Visible Corr.", :purple)

p_c = Plots.histogram2d(orig_vec, samp_vec,
    bins=(200, 200),
    xlabel="<vv> data", ylabel="<vv> model",
    title="Pairwise Covariance",
    colorbar=false, color=:viridis,
    framestyle=:box, grid=false,
    aspect_ratio=:equal,
    tickfontsize=11, guidefontsize=13, titlefontsize=14)
Plots.plot!(p_c, identity, linestyle=:dash, color=:red, linewidth=1.5, label="")
Plots.annotate!(p_c, Plots.xlims(p_c)[2] * 0.3, Plots.ylims(p_c)[2] * 0.9,
    Plots.text("ρ = $(round(ρ_c, digits=4))", :black, 13))

p_combined = Plots.plot(p_h, p_v, p_vh, p_c,
    layout=(2, 2),
    size=(1400, 1400),
    plot_title="RBM Performance Summary — $(domain_id)",
    plot_titlefontsize=16)

savefig(p_combined, joinpath(output_dir, "$(domain_id)_summary.png"))
println("Plot saved to: $output_dir")
