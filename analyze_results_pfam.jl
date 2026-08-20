using Plots
using Statistics: mean
gr()
ENV["GKSwstype"] = "100"

# =============================================================================
# CONFIG
# =============================================================================
# Protein analogue of TwoFamilyIsing/analyze_results.jl: reads the same
# "key=value" checkpoint .txt logs written by train_potts_pfam.jl's loggers
# (identical format, since train_potts_pfam.jl's checkpoint functions were
# copied verbatim from train_potts.jl) and reproduces the same diagnostic
# figures + written report. No dataset/model loading here — purely a log
# parser+plotter, so it needs no changes to the parsing/loading logic below,
# only the file-naming/config section and (since this environment doesn't have
# CairoMakie, only Plots — see train_small_pfam.jl) the plotting backend.
#
# Same ARGS convention as train_potts_pfam.jl / pair_results_pfam.jl:
# N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN_A=30] [N_HIDDEN_B=30] [FASTA_PATH=./PF00072_PF00512_paired.fasta] [SPLIT_SITE=111] [N_ITERS_B=N_ITERS] [PAIRED_TRAIN_FRAC=1.0] [USE_REWEIGHTING=true] [REG_B=0] [REG_PAIRED=0] [REG_A=0] [SEED=42] [CD_STEPS=100] [BATCH_SIZE=256] [LR=1e-4] [CLIP_NORM=1.0] [PAIRED_TRAIN_N=0]
# This script never opens FASTA_PATH itself (it's a pure log parser/plotter),
# but needs the same string train_potts_pfam.jl was run with to reconstruct
# an identical DATASET_TAG and thus find the right checkpoint files.
const OUTPUT_DIR   = "./results_pfam"
# CD_STEPS/BATCH_SIZE/LR/CLIP_NORM: ARGS-configurable (appended after SEED,
# positions 1-14 untouched) so this script can reconstruct the checkpoint
# filenames of a run trained with any of these overridden -- see the matching
# comment in train_potts_pfam.jl.
const CD_STEPS     = length(ARGS) >= 15 ? parse(Int, ARGS[15]) : 100
const CLIP_NORM    = length(ARGS) >= 18 ? parse(Float64, ARGS[18]) : 1.0   # must match train_potts_pfam.jl's CLIP_NORM to reconstruct the same filenames
const PAIRED_TRAIN_N = length(ARGS) >= 19 ? parse(Int, ARGS[19]) : 0   # must match train_potts_pfam.jl's PAIRED_TRAIN_N to reconstruct the same filenames
const LOG_EVERY    = 100
const N_ITERS      = parse(Int, ARGS[1])
const PAIRED_ITERS = parse(Int, ARGS[2])
const H_ADD        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
const N_HIDDEN_A   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const N_HIDDEN_B   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 30
const REG          = 0
const BATCH_SIZE   = length(ARGS) >= 16 ? parse(Int, ARGS[16]) : 256
const LR           = length(ARGS) >= 17 ? parse(Float32, ARGS[17]) : 1f-4
const FASTA_PATH   = length(ARGS) >= 6 ? ARGS[6] : "./PF00072_PF00512_paired.fasta"
const SPLIT_SITE   = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 111
const N_ITERS_B    = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : N_ITERS
const PAIRED_TRAIN_FRAC = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0
const USE_REWEIGHTING = length(ARGS) >= 10 ? parse(Bool, ARGS[10]) : true
# RBM B-specific l2l1_weights override -- must match train_potts_pfam.jl's
# constant of the same name to reconstruct the right log-file paths.
const REG_B        = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : REG
# Paired-model-specific l2l1_weights override -- must match train_potts_pfam.jl's
# constant of the same name (see the comment there) to reconstruct the right
# log-file paths.
const REG_PAIRED   = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : REG
# RBM A-specific l2l1_weights override -- must match train_potts_pfam.jl's
# constant of the same name to reconstruct the right log-file paths.
const REG_A        = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : REG
const SEED         = length(ARGS) >= 14 ? parse(Int, ARGS[14]) : 42
const DATASET_TAG  = "$(replace(splitext(basename(FASTA_PATH))[1], r"_paired$" => ""))_l$(SPLIT_SITE)"

# Only gains a distinguishing "_NITERSB=..." tag when N_ITERS_B actually
# differs from N_ITERS, so the common (symmetric) case's directory names are
# unchanged from before this feature existed. Same for "_PAIREDFRAC=..." vs
# PAIRED_TRAIN_FRAC.
n_iters_b_tag() = N_ITERS_B == N_ITERS ? "" : "_NITERSB=$(N_ITERS_B)"
paired_frac_tag() = PAIRED_TRAIN_N > 0 ? "_PAIREDN=$(PAIRED_TRAIN_N)" :
    (PAIRED_TRAIN_FRAC == 1.0 ? "" : "_PAIREDFRAC=$(PAIRED_TRAIN_FRAC)")
# Canonicalizes an integer-valued Float64 (e.g. from `parse(Float64, ARGS[n])`)
# to the same string an untyped Int 0 literal would produce ("0", not "0.0")
# -- see the matching comment in train_potts_pfam.jl. Must be defined before
# reg_b_tag()/PARAMTAG below, since PARAMTAG is a top-level const evaluated
# immediately (unlike function bodies, whose free variables resolve at call
# time) -- placing regstr() after PARAMTAG throws UndefVarError at load time.
regstr(r) = isinteger(r) ? string(Int(r)) : string(r)
reg_b_tag() = REG_B == REG ? "" : "_REGB=$(regstr(REG_B))"
reg_paired_tag() = REG_PAIRED == REG ? "" : "_REGPAIRED=$(regstr(REG_PAIRED))"
reg_a_tag() = REG_A == REG ? "" : "_REGA=$(regstr(REG_A))"
seed_tag() = SEED == 42 ? "" : "_SEED=$(SEED)"
const PARAMTAG = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_H_ADD=$(H_ADD)_N_ITERS=$(N_ITERS)_PAIRED_ITERS=$(PAIRED_ITERS)_REG=$(REG)_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)$(n_iters_b_tag())$(paired_frac_tag())$(reg_a_tag())$(reg_b_tag())$(reg_paired_tag())"
const REPORT_DIR = joinpath(OUTPUT_DIR, "analysis_$(PARAMTAG)")
isdir(REPORT_DIR) || mkpath(REPORT_DIR)

# Every figure filename carries the full param tag too (not just the
# containing directory) so files don't get confused once copied elsewhere or
# viewed in a flat gallery alongside other runs.
figpath(name) = joinpath(REPORT_DIR, "$(name)_$(PARAMTAG).png")

# Only one architecture exists on the protein side so far (no train_binary_pfam.jl
# analogue), but keep the ARCHS list structure so a future "binary_" variant
# just slots in — every downstream loop/table already guards on length(ARCHS).
const ARCHS = [("", "Potts+nsReLU")]

# H_ADD is deliberately absent from single_suffix_A/B() — see train_potts_pfam.jl's
# comment above its own definition: RBM A/B don't depend on H_ADD, so their
# saved files are shared across an H_ADD sweep, and only paired_suffix()
# (built independently below) varies by it. single_suffix_A/B() are separate
# (not one shared single_suffix()) so N_ITERS_B can differ from N_ITERS
# without affecting RBM A's filename — see the matching comment in
# train_potts_pfam.jl.
single_suffix_A() = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG_A))_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)"
single_suffix_B() = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS_B)_REG=$(regstr(REG_B))_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)"
paired_suffix()  = "N_HIDDEN_A=$(N_HIDDEN_A)_N_HIDDEN_B=$(N_HIDDEN_B)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(REG)_BS=$(BATCH_SIZE)_LR=$(LR)_CLIP=$(CLIP_NORM)_SPLIT=bal_RW=$(USE_REWEIGHTING ? 1 : 0)_DATA=$(DATASET_TAG)$(n_iters_b_tag())$(paired_frac_tag())$(reg_a_tag())$(reg_b_tag())$(reg_paired_tag())$(seed_tag())_PAIRED_ITERS=$(PAIRED_ITERS)"
path(arch, name, suffix) = joinpath(OUTPUT_DIR, "$(arch)$(name)_$(suffix).txt")

# =============================================================================
# LINE PARSERS
# =============================================================================
# All checkpoint files were written by our own `println`-based loggers in
# train_potts_pfam.jl (copied verbatim from train_potts.jl), so the format is
# fixed and known — this reconstructs the numbers directly from those exact
# "key=value" text lines rather than re-deriving anything from the raw hdf5
# models.
extract_iter(line) = parse(Int, match(r"\biter=(\d+)", line).captures[1])

function extract_scalar(line, key)
    m = match(Regex("\\b" * key * "=([^\\s]+)"), line)
    isnothing(m) && return NaN
    # normalize Float32 "f-exponent" scientific notation (e.g. "6.76f-5") to
    # standard "e-exponent" notation, which Base.parse actually accepts.
    return parse(Float64, replace(m.captures[1], r"f(-?\d+)$" => s"e\1"))
end

function extract_array(line, key)
    m = match(Regex("\\b" * key * "=(?:\\w+)?\\[([^\\]]*)\\]"), line)
    (isnothing(m) || isempty(strip(m.captures[1]))) && return Float64[]
    # small (e.g. regularized-toward-zero) Float32 values print in "f-exponent"
    # scientific notation (e.g. "6.76f-5"), which Base.parse doesn't accept —
    # normalize to standard "e-exponent" notation first.
    vals = replace.(strip.(split(m.captures[1], ",")), r"f(-?\d+)$" => s"e\1")
    return parse.(Float64, vals)
end

function extract_namedtuple(line, key)
    m = match(Regex("\\b" * key * "=\\(([^)]*)\\)"), line)
    isnothing(m) && return Dict{String,Float64}()
    out = Dict{String,Float64}()
    for pair in split(m.captures[1], ",")
        kv = split(pair, "=")
        length(kv) == 2 || continue
        v = replace(strip(kv[2]), r"f(-?\d+)$" => s"e\1")  # Float32 literal -> parseable
        out[strip(kv[1])] = parse(Float64, v)
    end
    return out
end

# =============================================================================
# FILE-TYPE LOADERS
# =============================================================================
function load_lpl(p)
    isfile(p) || return (iters=Int[], values=Float64[])
    vals = parse.(Float64, readlines(p))
    return (iters=collect(LOG_EVERY .* (1:length(vals))), values=vals)
end

function load_correlation(p)  # vh/vv/ab_check common fields
    isfile(p) || return (iters=Int[], r=Float64[], slope=Float64[], intercept=Float64[],
                          mean_abs_diff=Float64[], max_abs_diff=Float64[])
    lines = readlines(p)
    getf(k) = [extract_scalar(l, k) for l in lines]
    return (iters=extract_iter.(lines), r=getf("r"), slope=getf("slope"), intercept=getf("intercept"),
            mean_abs_diff=getf("mean_abs_diff"), max_abs_diff=getf("max_abs_diff"))
end

function load_vh(p)
    base = load_correlation(p)
    isfile(p) || return merge(base, (mean_h_data=Vector{Float64}[], mean_h_model=Vector{Float64}[]))
    lines = readlines(p)
    return merge(base, (mean_h_data=[extract_array(l, "mean_h_data") for l in lines],
                         mean_h_model=[extract_array(l, "mean_h_model") for l in lines]))
end

function load_norm_check(p)  # wn/gn_check
    isfile(p) || return (iters=Int[], mean_norm=Float64[], max_norm=Float64[], min_norm=Float64[], norms=Vector{Float64}[])
    lines = readlines(p)
    getf(k) = [extract_scalar(l, k) for l in lines]
    return (iters=extract_iter.(lines), mean_norm=getf("mean_norm"), max_norm=getf("max_norm"),
            min_norm=getf("min_norm"), norms=[extract_array(l, "norms") for l in lines])
end

function load_firing(p)
    isfile(p) || return (iters=Int[], data_rate=Vector{Float64}[], model_rate=Vector{Float64}[])
    lines = readlines(p)
    return (iters=extract_iter.(lines), data_rate=[extract_array(l, "data_rate") for l in lines],
            model_rate=[extract_array(l, "model_rate") for l in lines])
end

function load_freeze(p)
    isfile(p) || return (iters=Int[], status=String[], max_before=Float64[], max_after=Float64[], final_status="n/a")
    all_lines = readlines(p)
    # train_potts_pfam.jl appends one "FINAL status=... drift_after_reset=..."
    # line after the training loop returns (see its per-iteration hard freeze
    # assert) — same key=value shape but no "iter=" field, so it's parsed
    # separately from the regular per-checkpoint series below.
    final_idx = findlast(l -> startswith(l, "FINAL"), all_lines)
    final_status = isnothing(final_idx) ? "n/a" : String(match(r"\bstatus=(\S+)", all_lines[final_idx]).captures[1])
    lines = filter(l -> startswith(l, "iter="), all_lines)
    status = [String(match(r"\bstatus=(\S+)", l).captures[1]) for l in lines]
    before = [extract_namedtuple(l, "drift_before_reset") for l in lines]
    after  = [extract_namedtuple(l, "drift_after_reset") for l in lines]
    return (iters=extract_iter.(lines), status=status,
            max_before=[isempty(d) ? NaN : maximum(values(d)) for d in before],
            max_after=[isempty(d) ? NaN : maximum(values(d)) for d in after],
            final_status=final_status)
end

function load_architecture(arch)
    ssA, ssB, ps = single_suffix_A(), single_suffix_B(), paired_suffix()
    return (
        lpl_A         = load_lpl(path(arch, "lpl_A", ssA)),
        lpl_B         = load_lpl(path(arch, "lpl_B", ssB)),
        lpl_paired    = load_lpl(path(arch, "lpl_paired", ps)),
        lplval_A      = load_lpl(path(arch, "lplval_A", ssA)),
        lplval_B      = load_lpl(path(arch, "lplval_B", ssB)),
        lplval_paired = load_lpl(path(arch, "lplval_paired", ps)),
        vh_A          = load_vh(path(arch, "vh_check_A", ssA)),
        vh_B          = load_vh(path(arch, "vh_check_B", ssB)),
        vh_paired     = load_vh(path(arch, "vh_check_paired", ps)),
        h_A           = load_correlation(path(arch, "h_check_A", ssA)),
        h_B           = load_correlation(path(arch, "h_check_B", ssB)),
        h_paired      = load_correlation(path(arch, "h_check_paired", ps)),
        v_A           = load_correlation(path(arch, "v_check_A", ssA)),
        v_B           = load_correlation(path(arch, "v_check_B", ssB)),
        v_paired      = load_correlation(path(arch, "v_check_paired", ps)),
        ab_paired     = load_correlation(path(arch, "ab_check_paired", ps)),
        ab_val_paired = load_correlation(path(arch, "ab_val_check_paired", ps)),
        firing_paired = load_firing(path(arch, "firing_check_paired", ps)),
        wn_A          = load_norm_check(path(arch, "wn_check_A", ssA)),
        wn_B          = load_norm_check(path(arch, "wn_check_B", ssB)),
        wn_paired     = load_norm_check(path(arch, "wn_check_paired", ps)),
        gn_A          = load_norm_check(path(arch, "gn_check_A", ssA)),
        gn_B          = load_norm_check(path(arch, "gn_check_B", ssB)),
        gn_paired     = load_norm_check(path(arch, "gn_check_paired", ps)),
        freeze_paired = load_freeze(path(arch, "freeze_check_paired", ps)),
    )
end

data = Dict(label => load_architecture(arch) for (arch, label) in ARCHS)

has(x) = !isempty(x.iters)

# tail-average over the last `frac` fraction of checkpoints (min 1 point) —
# used to summarize "where a metric ended up" without being noisy from a
# single last data point.
function tailmean(v::AbstractVector{<:Real}; frac=0.2)
    isempty(v) && return NaN
    n = max(1, round(Int, frac * length(v)))
    return mean(v[end-n+1:end])
end

println("Loaded architectures: ", join(last.(ARCHS), ", "))
for (_, label) in ARCHS
    d = data[label]
    println("  [$label] lpl_A n=$(length(d.lpl_A.iters))  ab_paired n=$(length(d.ab_paired.iters))  ab_val_paired n=$(length(d.ab_val_paired.iters))")
end

# =============================================================================
# PLOTS
# =============================================================================
# This environment (julia --project=lal) has Plots (used by train_small_pfam.jl)
# but not CairoMakie, so panels are built with Plots.jl instead of Makie —
# same content/layout as analyze_results.jl, different plotting API.
colors = Dict("Potts+nsReLU" => :dodgerblue, "BinaryRBM" => :orangered)
const CELL = 380

function lines_and_points!(plt, x, y; color, linestyle=:solid, label="")
    isempty(x) && return plt
    Plots.plot!(plt, x, y; color, linestyle, label, marker=:circle, markersize=3, markerstrokewidth=0)
end

function fig_lpl()
    # RBM A/B: raw log-pseudolikelihood is a fine, sensitive training-progress
    # metric — every hidden unit is free, so it reflects the whole model.
    # Paired: it is not — 250/270 hidden units are frozen and already fit, so
    # total pseudolikelihood is dominated by them and barely moves regardless
    # of what the H_ADD free units learn (see train_potts_pfam.jl's ab_check
    # comments). ab_check/ab_check_val (train vs held-out r of the A-B
    # connected correlation) is the metric that's actually restricted to what
    # those free units can move, so the Paired panel plots that instead,
    # mirroring the same train-vs-held-out-curve layout as the lpl panels.
    panels = Plots.Plot[]
    for (col, (model, tlabel)) in enumerate([(:A, "RBM A"), (:B, "RBM B")])
        plt = Plots.plot(; title=tlabel, xlabel="iteration", ylabel=(col == 1 ? "log-pseudolikelihood" : ""),
                          legend=false, framestyle=:box, grid=false, legendfontsize=7)
        for (_, label) in ARCHS
            d = data[label]
            train = getfield(d, Symbol("lpl_$model"))
            val   = getfield(d, Symbol("lplval_$model"))
            has(train) && lines_and_points!(plt, train.iters, train.values; color=colors[label], linestyle=:solid, label="$label train")
            has(val)   && lines_and_points!(plt, val.iters, val.values; color=colors[label], linestyle=:dash, label="$label val")
        end
        push!(panels, plt)
    end
    plt_paired = Plots.plot(; title="Paired", xlabel="iteration", ylabel="r (A-B connected corr.)",
                             legend=:bottomright, framestyle=:box, grid=false, legendfontsize=7)
    Plots.hline!(plt_paired, [1.0]; color=:gray, linestyle=:dot, label="")
    for (_, label) in ARCHS
        d = data[label]
        train = d.ab_paired
        val   = d.ab_val_paired
        has(train) && lines_and_points!(plt_paired, train.iters, train.r; color=colors[label], linestyle=:solid, label="$label train")
        has(val)   && lines_and_points!(plt_paired, val.iters, val.r; color=colors[label], linestyle=:dash, label="$label val")
    end
    push!(panels, plt_paired)
    fig = Plots.plot(panels...; layout=(1, 3), size=(3 * CELL, CELL))
    savefig(fig, figpath("fig1_training_progress"))
    return fig
end

function fig_alignment(checkname, getter, title_prefix)
    panels = Plots.Plot[]
    for (col, (model, tlabel)) in enumerate([(:A, "RBM A"), (:B, "RBM B"), (:paired, "Paired")])
        axr = Plots.plot(; title=tlabel, ylabel=(col == 1 ? "r (correlation)" : ""),
                          legend=(col == 3 ? :bottomright : false), framestyle=:box, grid=false, legendfontsize=7)
        Plots.hline!(axr, [1.0]; color=:gray, linestyle=:dot, label="")
        for (_, label) in ARCHS
            d = getter(data[label], model)
            has(d) && lines_and_points!(axr, d.iters, d.r; color=colors[label], label=label)
        end
        push!(panels, axr)
    end
    for (col, (model, tlabel)) in enumerate([(:A, "RBM A"), (:B, "RBM B"), (:paired, "Paired")])
        axs = Plots.plot(; xlabel="iteration", ylabel=(col == 1 ? "slope" : ""), legend=false, framestyle=:box, grid=false)
        Plots.hline!(axs, [1.0]; color=:gray, linestyle=:dot, label="")
        for (_, label) in ARCHS
            d = getter(data[label], model)
            has(d) && lines_and_points!(axs, d.iters, d.slope; color=colors[label], label=label)
        end
        push!(panels, axs)
    end
    fig = Plots.plot(panels...; layout=(2, 3), size=(3 * CELL, 2 * CELL),
                      plot_title="$title_prefix — r/slope -> 1 means model tracks data along y=x", plot_titlefontsize=12)
    savefig(fig, figpath("fig_$(checkname)"))
    return fig
end

function fig_ab_headline()
    # title is long relative to a single-column CELL-wide panel — GR clips it
    # against the canvas edge unless the panel is widened and given explicit
    # top margin.
    axr = Plots.plot(; title="A-B cross-family correlation alignment (paired RBM)", titlefontsize=10,
                      ylabel="r", legend=:bottomright, framestyle=:box, grid=false, top_margin=5Plots.mm)
    Plots.hline!(axr, [1.0]; color=:gray, linestyle=:dot, label="")
    axs = Plots.plot(; ylabel="slope", xlabel="iteration", legend=false, framestyle=:box, grid=false)
    Plots.hline!(axs, [1.0]; color=:gray, linestyle=:dot, label="")
    for (_, label) in ARCHS
        d = data[label].ab_paired
        if has(d)
            lines_and_points!(axr, d.iters, d.r; color=colors[label], label=label)
            lines_and_points!(axs, d.iters, d.slope; color=colors[label], label=label)
        end
    end
    fig = Plots.plot(axr, axs; layout=(2, 1), size=(1.7 * CELL, 2 * CELL))
    savefig(fig, figpath("fig_headline_ab_check"))
    return fig
end

function fig_firing()
    panels = Plots.Plot[]
    for (_, label) in ARCHS
        d = data[label].firing_paired
        plt = Plots.plot(; title=label, xlabel="h_data - P(h=1 | data)", ylabel="h_model - P(h=1 | model)",
                          xlims=(0, 1), ylims=(0, 1), legend=false, framestyle=:box, grid=false, aspect_ratio=:equal)
        Plots.plot!(plt, [0, 1], [0, 1]; color=:gray, linestyle=:dot)
        has(d) && Plots.scatter!(plt, last(d.data_rate), last(d.model_rate); color=colors[label], markersize=4, markerstrokewidth=0)
        push!(panels, plt)
    end
    fig = Plots.plot(panels...; layout=(1, length(panels)), size=(length(panels) * CELL, CELL),
                      plot_title="Added-unit firing rate at last checkpoint (near y=x & away from 0/1 = healthy)", plot_titlefontsize=11)
    savefig(fig, figpath("fig_firing_rate"))
    return fig
end

function fig_norms()
    axw = Plots.plot(; title="Weight norm (added block)", ylabel="mean ||w||", legend=:bottomright, framestyle=:box, grid=false)
    axg = Plots.plot(; title="Gradient norm (added block)", ylabel="mean ||dw||", xlabel="iteration", legend=false, framestyle=:box, grid=false)
    for (_, label) in ARCHS
        wn = data[label].wn_paired
        gn = data[label].gn_paired
        has(wn) && lines_and_points!(axw, wn.iters, wn.mean_norm; color=colors[label], label=label)
        has(gn) && lines_and_points!(axg, gn.iters, gn.mean_norm; color=colors[label], label=label)
    end
    fig = Plots.plot(axw, axg; layout=(2, 1), size=(CELL, 2 * CELL))
    savefig(fig, figpath("fig_norms_added_block"))
    return fig
end

fig_lpl()
fig_alignment("vh_check", (d, m) -> getfield(d, Symbol("vh_$m")), "VH alignment (hidden units)")
fig_alignment("h_check", (d, m) -> getfield(d, Symbol("h_$m")), "H alignment (single hidden-unit means)")
fig_alignment("v_check", (d, m) -> getfield(d, Symbol("v_$m")), "V alignment (single visible-unit means)")
fig_ab_headline()
fig_firing()
fig_norms()

# =============================================================================
# WRITTEN REPORT
# =============================================================================
io = IOBuffer()
p(x...) = println(io, x...)

p("# Training analysis (protein pair $(DATASET_TAG)) — N_ITERS=$(N_ITERS), PAIRED_ITERS=$(PAIRED_ITERS)")
p()
p("Figures saved to `$(REPORT_DIR)/`.")
p()

for (_, label) in ARCHS
    d = data[label]
    p("## $label")
    p()

    # --- 1. Did the individual RBMs train well? ---
    p("### 1. Individual RBM quality (A=sites 1:$(SPLIT_SITE), B=sites $(SPLIT_SITE+1):end)")
    for m in (:A, :B)
        lpl = getfield(d, Symbol("lpl_$m"))
        val = getfield(d, Symbol("lplval_$m"))
        vh  = getfield(d, Symbol("vh_$m"))
        h   = getfield(d, Symbol("h_$m"))
        v   = getfield(d, Symbol("v_$m"))
        if !has(lpl)
            p("- RBM $m: no data yet.")
            continue
        end
        gap = has(val) ? tailmean(lpl.values) - tailmean(val.values) : NaN
        p("- RBM $m: final lpl=$(round(tailmean(lpl.values); digits=4)), lpl_val=$(has(val) ? round(tailmean(val.values); digits=4) : "n/a") (train-val gap=$(round(gap; digits=4)))")
        p("  vh_check r=$(round(tailmean(vh.r); digits=3)) slope=$(round(tailmean(vh.slope); digits=3)) — joint ⟨v h⟩ correlation, data vs model")
        has(h) && p("  h_check  r=$(round(tailmean(h.r); digits=3)) slope=$(round(tailmean(h.slope); digits=3)) — single hidden-unit ⟨h⟩ means, data vs model")
        has(v) && p("  v_check  r=$(round(tailmean(v.r); digits=3)) slope=$(round(tailmean(v.slope); digits=3)) — single visible-unit ⟨v⟩ means, data vs model")
    end
    p()

    # --- 2. Are the added hidden units learning anything meaningful? ---
    p("### 2. Added hidden units (paired RBM)")
    freeze = d.freeze_paired
    vh = d.vh_paired
    wn = d.wn_paired
    gn = d.gn_paired
    fr = d.firing_paired
    if has(freeze)
        nviol = count(==( "VIOLATION"), freeze.status)
        p("- Freeze check: $(length(freeze.status) - nviol)/$(length(freeze.status)) logged checkpoints OK" * (nviol > 0 ? " — **$nviol VIOLATIONS, freezing is broken!**" : " — private blocks stayed correctly frozen throughout."))
        p("  (every training iteration is hard-asserted, not just logged checkpoints — training would have errored out immediately on any violation)")
        p("  Final post-training check: **$(freeze.final_status)**")
    end
    if has(vh)
        p("- vh_check: r=$(round(tailmean(vh.r); digits=3)), slope=$(round(tailmean(vh.slope); digits=3)), mean|Δ|=$(round(tailmean(vh.mean_abs_diff); digits=4))")
    end
    if has(wn) && has(gn)
        growth = wn.mean_norm[end] / wn.mean_norm[1]
        p("- Weight norm: $(round(wn.mean_norm[1]; digits=3)) → $(round(wn.mean_norm[end]; digits=3)) (×$(round(growth; digits=2)) growth)")
        p("- Gradient norm (final): mean=$(round(tailmean(gn.mean_norm); digits=3)) — " *
           (tailmean(gn.mean_norm) > 0.05 ? "still a live training signal." : "signal has largely died out (converged, or stuck)."))
    end
    if has(fr)
        dr, mr = last(fr.data_rate), last(fr.model_rate)
        ndead = count(x -> x < 0.05 || x > 0.95, dr)
        p("- Firing rate (last checkpoint): $(ndead)/$(length(dr)) added units pinned near 0 or 1 under data " *
           (ndead == 0 ? "(no dead units detected)." : "(possible dead units)."))
    end
    p()

    # --- 3. Are they learning interfamily correlations specifically? ---
    p("### 3. Interfamily (A-B) correlation — the headline question")
    ab = d.ab_paired
    ab_val = d.ab_val_paired
    if has(ab)
        p("- ab_check (train): r=$(round(tailmean(ab.r); digits=3)), slope=$(round(tailmean(ab.slope); digits=3)), mean|Δ|=$(round(tailmean(ab.mean_abs_diff); digits=4))")
        if has(ab_val)
            gap = tailmean(ab.r) - tailmean(ab_val.r)
            p("- ab_check_val (held-out): r=$(round(tailmean(ab_val.r); digits=3)), slope=$(round(tailmean(ab_val.slope); digits=3)), mean|Δ|=$(round(tailmean(ab_val.mean_abs_diff); digits=4)) (train-val r gap=$(round(gap; digits=3)))")
        else
            p("- No ab_check_val data yet.")
        end
        verdict_r = has(ab_val) ? tailmean(ab_val.r) : tailmean(ab.r)
        verdict = verdict_r > 0.8 ? "learning real cross-family structure that generalizes" :
                  verdict_r > 0.4 ? "partial/early progress — needs more training or more H_ADD units to confirm" :
                  "not yet capturing cross-family correlations that generalize"
        p("  → **Verdict: $verdict** (held-out r=$(round(verdict_r; digits=2)))")
    else
        p("- No ab_check data yet.")
    end
    p()
end

if length(ARCHS) == 2
    p("## Potts vs Binary — head-to-head")
    p()
    p("| metric | $(ARCHS[1][2]) | $(ARCHS[2][2]) |")
    p("|---|---|---|")
    function row(name, getter)
        vals = [has(getter(data[label])) ? round(tailmean(getter(data[label]).r); digits=3) : NaN for (_, label) in ARCHS]
        p("| $name (r) | $(vals[1]) | $(vals[2]) |")
    end
    row("vh_check (RBM A)", d -> d.vh_A)
    row("h_check (RBM A)", d -> d.h_A)
    row("ab_check (paired) — **headline**", d -> d.ab_paired)
    p()
end
p("Caveat: this is a **training-dynamics** report from logged summary statistics only —")
p("it does not yet verify the paired model against freshly *sampled* data (Gibbs), which is")
p("what pair_results_pfam.jl does.")

report = String(take!(io))
open(joinpath(REPORT_DIR, "report.md"), "w") do f
    write(f, report)
end
println(report)
println("\nReport + figures written to $(REPORT_DIR)/")
