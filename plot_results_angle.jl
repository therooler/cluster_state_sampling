#!/usr/bin/env julia
# Plot per-angle sweep results
# Usage:
#   julia --project=. plot_results_angle.jl [data_dir] [out_prefix]
# Example:
#   julia --project=. plot_results_angle.jl data data/plots/angles

using Printf
using Plots
using Measures
using Statistics

# Extract first floating point number from a string (handles values like "0.9 + 0.0im")
function parse_first_float(s::AbstractString)
    m = match(r"([+-]?(?:\d+\.?\d*|\.?\d+)(?:[eE][+-]?\d+)?)", s)
    m === nothing && error("Cannot parse float from '" * s * "'")
    return parse(Float64, m.captures[1])
end

# Load CSVs that contain per-angle rows with columns including angle, mean, std.
# Filenames should include L{L}_D{D} so we can group by system size and bond dim.
function load_angle_results(data_dir::AbstractString="data")
    files = readdir(data_dir; join=true)
    rows = Vector{NamedTuple}()
    for f in files
        isfile(f) || continue
        name = split(basename(f), '.')[1]
        # Accept filenames like L{L}_D{D}[_NS{n}]_SEED{seed}.csv
        m = match(r"^L(\d+)_D(\d+)(?:_NS(\d+))?(?:_SEED(\d+))?$", name)
        m === nothing && continue
        L = parse(Int, m.captures[1])
        D = parse(Int, m.captures[2])
        seed = m.captures[4] === nothing ? missing : parse(Int, m.captures[4])

        lines = readlines(f)
        if length(lines) < 2
            @warn "Skipping file with insufficient lines" f
            continue
        end
        header = split(lines[1], ',')
        lowerh = lowercase.(strip.(header))
        idx_angle = findfirst(x->x=="angle", lowerh)
        idx_mean = findfirst(x->x=="mean", lowerh)
        idx_std = findfirst(x->x=="std", lowerh)
        if idx_angle === nothing || idx_mean === nothing || idx_std === nothing
            @warn "Skipping file with unexpected header (need angle,mean,std)" f
            continue
        end

        for ln in lines[2:end]
            data = split(ln, ',')
            # protect against short/malformed lines
            length(data) < max(idx_angle, idx_mean, idx_std) && continue
            a = parse_first_float(strip(data[idx_angle]))
            mu = parse_first_float(strip(data[idx_mean]))
            sigma = parse_first_float(strip(data[idx_std]))
            push!(rows, (L=L, D=D, angle=a, mean=mu, std=sigma, seed=seed))
        end
    end
    return rows
end

# Round a Float64 to Float32 precision and return as Float64 (clipped precision)
clip_to_f32(x::Real) = Float64(Float32(x))

function plot_angle_results(data_dir::AbstractString="data", out_prefix::AbstractString=joinpath(data_dir, "plots", "angles"))
    rows = load_angle_results(data_dir)
    isempty(rows) && error("No per-angle CSV results found in $(data_dir). Expected files named like L{L}_D{D}*.csv with header containing angle,mean,std")

    seed_list = [100, 200, 300, 400, 500, 600, 700, 800,900,1000]
    if seed_list !== nothing
        # Filter rows to only include requested seeds
        rows = filter(r -> !(ismissing(r[:seed])) && in(r[:seed], seed_list), rows)
        isempty(rows) && error("No rows found for requested seeds: $(seed_list)")
    end

    # compute how many unique seeds are present after filtering (for title)
    seeds_used = unique(filter(x -> !ismissing(x), [r.seed for r in rows]))
    n_seeds = length(seeds_used)

    # collect unique L and D
    Ls = sort(unique(r.L for r in rows))
    Ds = sort(unique(r.D for r in rows))

    # create a mapping for fast lookup: map (L,D) -> vector of rows
    byLD = Dict{Tuple{Int,Int}, Vector{NamedTuple}}()
    for r in rows
        get!(byLD, (r.L, r.D), NamedTuple[])
        push!(byLD[(r.L, r.D)], r)
    end
    # sort each vector by angle
    for (k,v) in byLD
        byLD[k] = sort(v, by = x -> x.angle)
    end

    # If multiple seeds were requested, aggregate across seeds for each (L,D,angle)
    # and replace byLD entries with aggregated points (angle, mean, std)
    if seed_list !== nothing
        agg_byLD = Dict{Tuple{Int,Int}, Vector{NamedTuple}}()
        for ((L,D), vec) in byLD
            # group by angle
            angle_groups = Dict{Float64, Vector{NamedTuple}}()
            for r in vec
                get!(angle_groups, r.angle, NamedTuple[])
                push!(angle_groups[r.angle], r)
            end
            agg_points = NamedTuple[]
            for (ang, group) in sort(collect(angle_groups); by=first)
                mus = [g.mean for g in group]
                stds = [g.std for g in group]
                mean_mu = Statistics.mean(mus)
                mean_std = Statistics.mean(stds)
                err_mu = length(mus) > 1 ? Statistics.std(mus) : 0.0
                err_std = length(stds) > 1 ? Statistics.std(stds) : 0.0
                push!(agg_points, (angle=ang, mean=mean_mu, std=mean_std, err_mean=err_mu, err_std=err_std))
            end
            agg_byLD[(L,D)] = sort(agg_points, by = x -> x.angle)
        end
        byLD = agg_byLD
    end

    # layout: make a grid with one subplot per L
    nL = length(Ls)
    ncols = ceil(Int, sqrt(nL))
    nrows = ceil(Int, nL / ncols)

    # Minimum plotted value to avoid zeros on log scale
    min_val = 1e-8

    # Mean plot
    plt_mean = plot(layout=(nrows, ncols), size=(600*ncols, 500*nrows), left_margin=8mm, right_margin=8mm, bottom_margin=10mm, top_margin=6mm)
    for (i,L) in enumerate(Ls)
        for D in Ds
            key = (L,D)
            haskey(byLD, key) || continue
            group = byLD[key]
            xs = [r.angle/2 for r in group]
            # clip means to float32 numerical precision
            ys = [clip_to_f32(r.mean) for r in group]
            # compute error bars (std of mus) if available
            if :err_mean in propertynames(group[1])
                yerrs = [clip_to_f32(getfield(r, :err_mean)) for r in group]
            else
                yerrs = fill(0.0, length(ys))
            end
            # existing transformation preserved (distance from 1 then abs)
            ys_t = abs.(ys .- 1)
            # ensure a minimum positive value so zeros show on a log scale
            ys_t = max.(ys_t, min_val)
            # sanitize y-errors
            yerrs = [isnan(x) ? 0.0 : x for x in yerrs]
            yerrs = max.(yerrs, 0.0)
            plot!(plt_mean[i], xs, ys_t; yerror=yerrs, label=@sprintf("D=%d", D), marker=:circle)
        end
        # set y-limits using the chosen minimum and log scale for visibility
        plot!(ylim=(min_val, 1.0), yscale=:log10)
        xlabel!(plt_mean[i], "angle")
        ylabel!(plt_mean[i], "mean")
        title!(plt_mean[i], @sprintf("L=%d - %d seeds", L, length(seed_list)))
    end

    # Std plot
    plt_std = plot(layout=(nrows, ncols), size=(600*ncols, 500*nrows), left_margin=8mm, right_margin=8mm, bottom_margin=10mm, top_margin=6mm)
    for (i,L) in enumerate(Ls)
        for D in Ds
            key = (L,D)
            haskey(byLD, key) || continue
            group = byLD[key]
            xs = [r.angle/2 for r in group]
            # clip stds to float32 numerical precision
            ys = [clip_to_f32(r.std) for r in group]
            # compute error bars for std plot if available (std of stds)
            if :err_std in propertynames(group[1])
                yerrs = [clip_to_f32(getfield(r, :err_std)) for r in group]
            else
                yerrs = fill(0.0, length(ys))
            end
            # optionally ensure a small positive floor to avoid plotting issues
            ys = max.(ys, min_val)
            yerrs = [isnan(x) ? 0.0 : x for x in yerrs]
            yerrs = max.(yerrs, 0.0)
            plot!(plt_std[i], xs, ys; yerror=yerrs, label=@sprintf("D=%d", D), marker=:circle)
        end
        xlabel!(plt_std[i], "angle (rad)")
        ylabel!(plt_std[i], "std")
        plot!(ylim=[0,20])
        title!(plt_std[i], @sprintf("L=%d - %d seeds", L, length(seed_list)))
    end

    # add a title mentioning how many seeds were used
    # title_str = @sprintf("L=(s) seed(s)", L, n_seeds)

    # ensure output dir
    mkpath(dirname(out_prefix))
    # attach the overall title to each figure
    # plot!(plt_mean, title=title_str)
    # plot!(plt_std, title=title_str)

    mean_out = out_prefix * "_mean.png"
    std_out = out_prefix * "_std.png"
    savefig(plt_mean, mean_out)
    savefig(plt_std, std_out)
    println("Saved mean plot to: " * mean_out)
    println("Saved std plot to: " * std_out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    out_prefix = joinpath("figures", "angles")
    plot_angle_results("data", out_prefix)
end
