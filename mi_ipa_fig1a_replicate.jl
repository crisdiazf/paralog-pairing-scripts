"""
Run one replicate of the Fig. 1A protocol (Bitbol 2018) and write the
per-iteration TP fraction to a file. One process per (N_start, replicate)
job, so the outer sweep can be parallelized across OS processes with
`xargs -P` (Threads.@threads inside MI_IPA.jl scaled poorly at high thread
counts for this workload -- process-level parallelism across independent
replicates is far more efficient here).

Usage: julia -t 1 --project=lal mi_ipa_fig1a_replicate.jl <N_start> <replicate> <n_iterations> <out_file>
"""

include("MI_IPA.jl")

N_start = parse(Int, ARGS[1])
replicate = parse(Int, ARGS[2])
n_iterations = parse(Int, ARGS[3])
out_file = ARGS[4]

const FASTA_PATH = "Standard_HKRR_dataset.fasta"
const LENGTH_A = 64
const NINCREMENT = 6

X, species, headers = read_alignment(FASTA_PATH)
groups, species_ids = build_species_groups(species)

seed = N_start * 100_003 + replicate # distinct, reproducible seed per (N_start, replicate)

tp = run_mi_ipa_with_training(X, groups; LengthA = LENGTH_A, N_start = N_start,
                               Nincrement = NINCREMENT, n_iterations = n_iterations, seed = seed)

open(out_file, "w") do io
    println(io, join(tp, '\t'))
end
