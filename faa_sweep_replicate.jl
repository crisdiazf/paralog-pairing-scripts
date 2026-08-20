# One replicate of the Fig-1A-style protocol for PF00072-PF01339 (misc/.faa
# version). Usage: julia -t 1 --project=lal faa_sweep_replicate.jl <N_start> <replicate> <n_iterations> <out_file>
include("/home/cdiaz-faloh/MI_IPA.jl")

function pf_species_parser(header::AbstractString)
    id_part = split(header, "|")[end]
    id_part = split(id_part, "/")[1]
    parts = split(id_part, "_")
    code = length(parts) >= 2 ? String(parts[end]) : String(id_part)
    if occursin(r"^[0-9]", code) || occursin("UNC", code) || occursin("UNK", code)
        return missing
    end
    return code
end

N_start = parse(Int, ARGS[1])
replicate = parse(Int, ARGS[2])
n_iterations = parse(Int, ARGS[3])
out_file = ARGS[4]

const FASTA_PATH = "/home/cdiaz-faloh/misc/PF00072_PF01339_paired.faa"
const LENGTH_A = 111
const NINCREMENT = 6

X, species, headers = read_alignment(FASTA_PATH; species_of_header=pf_species_parser)
groups, species_ids = build_species_groups(species)

seed = N_start * 100_003 + replicate

tp = run_mi_ipa_with_training(X, groups; LengthA = LENGTH_A, N_start = N_start,
                               Nincrement = NINCREMENT, n_iterations = n_iterations, seed = seed)

open(out_file, "w") do io
    println(io, join(tp, '\t'))
end
