include("MI_IPA.jl")
using DelimitedFiles, Statistics

const RESULTS_DIR = "mi_ipa_fig1a_results"
const N_STARTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

X, species, headers = read_alignment("Standard_HKRR_dataset.fasta")
groups, species_ids = build_species_groups(species)
chance = chance_tp_fraction(groups)
println("chance level: $chance")

open("mi_ipa_fig1a_summary.tsv", "w") do io
    println(io, "N_start\tn_replicates\t", join(1:500, '\t'))
    for N in N_STARTS
        files = filter(f -> occursin(Regex("^N$(N)_r\\d+\\.txt\$"), f), readdir(RESULTS_DIR))
        n_rep = length(files)
        if n_rep == 0
            println("N_start=$N: NO REPLICATES, skipping")
            continue
        end
        mat = zeros(Float64, n_rep, 500)
        for (i, f) in enumerate(files)
            row = parse.(Float64, split(strip(readline(joinpath(RESULTS_DIR, f))), '\t'))
            @assert length(row) == 500 "unexpected length in $f: $(length(row))"
            mat[i, :] = row
        end
        avg = vec(mean(mat, dims = 1))
        println("N_start=$N: $n_rep replicates, iter1=$(round(avg[1],digits=3)) iter500=$(round(avg[end],digits=3))")
        println(io, N, '\t', n_rep, '\t', join(avg, '\t'))
    end
end
println("chance\t-\t", chance)
