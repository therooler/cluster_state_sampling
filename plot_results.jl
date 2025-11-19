#!/usr/bin/env julia
# Plot mean and std versus L^2 for different bond dimensions D, using CSVs in ./data
# Usage:
#   julia --project=. plot_results.jl [data_dir] [output_path]
# Example:
#   julia --project=. plot_results.jl data data/plots/summary.png

using Printf
using Plots
using Measures

# Extract first floating point number from a string (handles values like "0.9 + 0.0im")
function parse_first_float(s::AbstractString)
    m = match(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", s)
    m === nothing && error("Cannot parse float from '" * s * "'")
    return parse(Float64, m.captures[1])
end

# Discover CSVs named like L{L}_D{D}.csv or L{L}_D{D}_NS{n}[_SEED{s}].csv and parse mean/std
function load_results(data_dir::AbstractString; ns_filter::Union{Nothing,Int}=nothing, seed_filter::Union{Nothing,Int}=nothing)
    files = readdir(data_dir; join=true)
    rows = NamedTuple[]
    for f in files
        isfile(f) || continue
        name = split(basename(f), '.')[1]
    # Accept: L{L}_D{D}.csv, L{L}_D{D}_NS{n}.csv, and with optional _SEED{seed}
    m = match(r"^L(\d+)_D(\d+)(?:_NS(\d+))?(?:_SEED(\d+))?$", name)
        m === nothing && continue
        L = parse(Int, m.captures[1])
        D = parse(Int, m.captures[2])
        ns = m.captures[3] === nothing ? missing : parse(Int, m.captures[3])
        seed = m.captures[4] === nothing ? missing : parse(Int, m.captures[4])
        if ns_filter !== nothing
            if ismissing(ns) || ns != ns_filter
                continue
            end
        end
        if seed_filter !== nothing
            if ismissing(seed) || seed != seed_filter
                continue
            end
        end
        # Read simple two-line CSV
        lines = readlines(f)
        if length(lines) < 2
            @warn "Skipping file with insufficient lines" f
            continue
        end
        # Expect header: metric,mean,std; data row with 3 fields
        data = split(lines[2], ',')
        if length(data) < 3
            @warn "Skipping file with malformed row" f
            continue
        end
        mu = parse_first_float(strip(data[2]))
        sigma = parse_first_float(strip(data[3]))
        push!(rows, (L=L, L2=L*L, D=D, mean=mu, std=sigma))
    end
    return rows
end

function plot_results(data_dir::AbstractString="data", out_path::AbstractString="data/plots/summary.png"; ns_filter::Union{Nothing,Int}=nothing, seed_filter::Union{Nothing,Int}=nothing)
    rows = load_results(data_dir; ns_filter=ns_filter, seed_filter=seed_filter)
    isempty(rows) && error("No result CSVs found in $(data_dir). Expected files named L{L}_D{D}.csv")

    # Group by bond dimension D
    byD = Dict{Int, Vector{NamedTuple}}()
    for r in rows
        get!(byD, r.D, NamedTuple[])
        push!(byD[r.D], r)
    end
    # Sort each group by L2 for nice lines
    for (k, v) in byD
        byD[k] = sort(v, by = x -> x.L2)
    end

    # Prepare three subplots: mean vs L2, std vs L2, and mean vs D (lines for different L)
    plt = plot(layout=(1,3), size=(1650, 450), left_margin=8mm, right_margin=8mm, bottom_margin=10mm, top_margin=6mm)

    # Left: mean
    for (D, group) in sort(collect(byD); by=first)
        xs = [r.L2 for r in group]
        ys = [r.mean for r in group]
        plot!(plt[1], xs, abs.(ys.-1); label=@sprintf("D=%d", D), marker=:circle)
    end
    plot!(plt[1], yscale=:log10)
    xlabel!(plt[1], "n")
    ylabel!(plt[1], "abs(1-mean)")
    title_left = "Mean vs L^2"
    if ns_filter !== nothing && seed_filter !== nothing
        title_left *= " (NS=$(ns_filter), SEED=$(seed_filter))"
    elseif ns_filter !== nothing
        title_left *= " (NS=$(ns_filter))"
    elseif seed_filter !== nothing
        title_left *= " (SEED=$(seed_filter))"
    end
    title!(plt[1], title_left)

    # Right: std
    for (D, group) in sort(collect(byD); by=first)
        xs = [r.L2 for r in group]
        ys = [r.std for r in group]
        plot!(plt[2], xs, abs.(ys); label=@sprintf("D=%d", D), marker=:circle)
    end
    xlabel!(plt[2], "n")
    ylabel!(plt[2], "std")
    title_right = "Std vs L^2"
    if ns_filter !== nothing && seed_filter !== nothing
        title_right *= " (NS=$(ns_filter), SEED=$(seed_filter))"
    elseif ns_filter !== nothing
        title_right *= " (NS=$(ns_filter))"
    elseif seed_filter !== nothing
        title_right *= " (SEED=$(seed_filter))"
    end
    plot!(plt[2], yscale=:log10)
    title!(plt[2], title_right)

    # Third: mean vs D, with different lines for each system size L
    # Group by L
    byL = Dict{Int, Vector{NamedTuple}}()
    for r in rows
        get!(byL, r.L, NamedTuple[])
        push!(byL[r.L], r)
    end
    # Sort each group's entries by D for clean lines
    for (k, v) in byL
        byL[k] = sort(v, by = x -> x.D)
    end

    for (L, group) in sort(collect(byL); by=first)
        xs = [r.D for r in group]
        ys = [r.mean for r in group]
        plot!(plt[3], xs, abs.(ys.-1); label=@sprintf("L=%d", L), marker=:circle)
    end
    # set linear y-limits for the third subplot (mean vs D)
    plot!(plt[3])
    xlabel!(plt[3], "D")
    ylabel!(plt[3], "mean")
    plot!(plt[3], yscale=:log10)
    title_third = "Mean vs D"
    if ns_filter !== nothing && seed_filter !== nothing
        title_third *= " (NS=$(ns_filter), SEED=$(seed_filter))"
    elseif ns_filter !== nothing
        title_third *= " (NS=$(ns_filter))"
    elseif seed_filter !== nothing
        title_third *= " (SEED=$(seed_filter))"
    end
    title!(plt[3], title_third)

    # Make sure output dir exists
    mkpath(dirname(out_path))
    savefig(plt, out_path)
    println("Saved plot to: " * out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = length(ARGS) >= 1 ? ARGS[1] : "data"
    out_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(data_dir, "plots", "summary.png")
    ns_filter = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
    seed_filter = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
    plot_results(data_dir, out_path; ns_filter=ns_filter, seed_filter=seed_filter)
end
