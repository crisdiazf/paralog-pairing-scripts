using Statistics

const RESULTS_DIR = "/home/cdiaz-faloh/faa_sweep_results"
const N_STARTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

for N in N_STARTS
    files = filter(f -> occursin(Regex("^N$(N)_r\\d+\\.txt\$"), f), readdir(RESULTS_DIR))
    n_rep = length(files)
    mat = zeros(Float64, n_rep, 5)
    for (i, f) in enumerate(files)
        row = parse.(Float64, split(strip(readline(joinpath(RESULTS_DIR, f))), '\t'))
        mat[i, :] = row
    end
    avg = vec(mean(mat, dims = 1))
    sd = vec(std(mat, dims = 1))
    println("N_start=$N n_rep=$n_rep  mean=", round.(avg, digits = 4), "  std=", round.(sd, digits = 4))
end
