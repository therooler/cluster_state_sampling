#!/usr/bin/env julia
"""Plot certified p_over_q vs R arranged in a grid with rows = L and columns = D.

Usage:
  julia --project=. plotting/plot_certify_byL.jl <data_root> [out_dir]

Expects data under <data_root>/L{L}/D{D}/SEED{seed}/R{R}/stats_<angle>.csv
"""

using Printf, Statistics, Plots
using ColorSchemes
using Measures
using LaTeXStrings

function parse_angle_from_filename(fname::AbstractString)
    name = basename(fname)
    m = match(r"stats[_-]([0-9p\.]+)\.csv$", name)
    m === nothing && return nothing
    s = replace(m.captures[1], 'p' => '.')
    try
        return parse(Float64, s)
    catch
        return nothing
    end
end

function read_mean_certified(filepath::AbstractString)
    vals = Float64[]
    open(filepath, "r") do io
        firstline = try readline(io) catch; return NaN end
        try
            push!(vals, parse(Float64, strip(firstline)))
        catch
            # header, ignore
        end
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            m = match(r"([+-]?(?:\d+\.?\d*|\.?\d+)(?:[eE][+-]?\d+)?)", s)
            m === nothing && continue
            push!(vals, parse(Float64, m.captures[1]))
        end
    end
    return isempty(vals) ? NaN : mean(vals)
end

function collect_nested(data_root::AbstractString)
    nested = Dict{Int, Dict{Int, Dict{Float64, Vector{Float64}}}}()
    for (root, dirs, files) in walkdir(data_root)
        for f in files
            startswith(f, "stats_") || continue
            ang = parse_angle_from_filename(f)
            ang === nothing && continue
            comps = splitpath(root)
            L = nothing; D = nothing; R = nothing
            for c in comps
                mL = match(r"^L(\d+)$", c); mL !== nothing && (L = parse(Int, mL.captures[1]))
                mD = match(r"^D(\d+)$", c); mD !== nothing && (D = parse(Int, mD.captures[1]))
                mR = match(r"^[Rr](\d+)$", c); mR !== nothing && (R = parse(Float64, mR.captures[1]))
            end
            (L === nothing || D === nothing || R === nothing) && continue
            m = read_mean_certified(joinpath(root, f))
            isnan(m) && continue
            get!(nested, L, Dict{Int, Dict{Float64, Vector{Float64}}}())
            get!(nested[L], D, Dict{Float64, Vector{Float64}}())
            get!(nested[L][D], R, Float64[])
            push!(nested[L][D][R], m)
        end
    end
    # average per R across seeds
    for (L, dmap) in nested
        for (D, rmap) in dmap
            for (R, vals) in collect(rmap)
                rmap[R] = [mean(vals)]
            end
        end
    end
    return nested
end

function plot_grid_by_LD(data_root::AbstractString = "data", out_dir::AbstractString = "figures/certify_grid")
    nested = collect_nested(data_root)
    isempty(nested) && error("No data found under ", data_root)
    mkpath(out_dir)

    Ls = sort(collect(keys(nested)))
    Ds_set = Set{Int}()
    for L in Ls
        for D in keys(nested[L])
            push!(Ds_set, D)
        end
    end
    Ds = sort(collect(Ds_set))
    nrows = max(1, length(Ls)); ncols = max(1, length(Ds))

    # global angle slices and palette
    n_slices = 20
    global_angles = collect(range(0, stop = pi/2, length = n_slices))
    pal = reverse([get(ColorSchemes.viridis, t) for t in range(0, stop = 1, length = n_slices)])

    plt = plot(layout = (nrows, ncols), size = (600*ncols, 500*nrows),
               left_margin = 25mm, right_margin = 10mm, top_margin = 5mm, bottom_margin = 5mm,
               tickfont = font(15), xtickfont = font(20), guidefont=font(15))

    for (iL, L) in enumerate(Ls)
        for (jD, D) in enumerate(Ds)
            idx = (iL - 1) * ncols + jD
            ax = plt[idx]
            xlabel!(ax, "R", fontsize=font(20))
            ylabel!(ax, "p/q", fontsize=font(20))
            title!(ax, @sprintf("L=%d  D=%d", L, D))

            if !haskey(nested, L) || !haskey(nested[L], D)
                continue
            end

            Rs = sort(collect(keys(nested[L][D])))

            # find angles present by scanning files under L/D
            sample_dir = joinpath(data_root, @sprintf("L%d", L), @sprintf("D%d", D))
            angles = Float64[]
            for (root, dirs, files) in walkdir(sample_dir)
                for f in files
                    startswith(f, "stats_") || continue
                    a = parse_angle_from_filename(basename(f))
                    a !== nothing && push!(angles, a)
                end
            end
            angles = sort(unique(angles))
            isempty(angles) && continue

            for ang in angles
                xs = Float64[]; ys = Float64[]
                for R in Rs
                    found = nothing
                    for (root2, dirs2, files2) in walkdir(joinpath(data_root, @sprintf("L%d", L), @sprintf("D%d", D)))
                        for f in files2
                            if startswith(f, "stats_") && (occursin(@sprintf("%0.5f", ang), f) || occursin(@sprintf("%0.5fp", ang), f))
                                if occursin(@sprintf("R%d", Int(R)), root2)
                                    found = joinpath(root2, f); break
                                end
                            end
                        end
                        found !== nothing && break
                    end
                    found === nothing && continue
                    push!(xs, R)
                    push!(ys, read_mean_certified(found))
                end
                isempty(xs) && continue
                idx_col = findmin(abs.(global_angles .- ang))[2]
                col = pal[idx_col]
                plot!(ax, xs, ys, label = false, color = col, marker = :circle)
            end

            plot!(ax, yscale = :log10, ylim = (1e-1, 1e3))
            plot!(ax, xlim = (0, maximum([maximum(Rs)+3, 2])))
        end
    end
    # create a horizontal colorbar-like heatmap on top showing the angle palette
    cb_mat = reshape(global_angles, 1, n_slices)
    # compute tick positions corresponding to angles 0, pi/4, pi/2 within the sampled slices
    tick_pos = [1, findmin(abs.(global_angles .- (pi/4)))[2], n_slices]
    tick_labels = [L"0", L"\frac{\pi}{8}", L"\frac{\pi}{4}"]
    # show the angle label as a centered title above the slim colorband and draw a thin box
    cb = heatmap(cb_mat; color = cgrad(:viridis, n_slices, rev=true), axis = true,
                  legend = false, title = "angle (rad)", titlefont = font(22),
                  framestyle = :box, xticks = (tick_pos, tick_labels), yticks = false,
                  xtickfont = font(18))

    # compose with colorbar on top (small height) and main grid below
    combined = plot(cb, plt; layout = grid(2, 1, heights=[0.02 ,0.98]))

    outpng = joinpath(out_dir, "certify_grid_LxD.png")
    savefig(combined, outpng)
    println("Saved: ", outpng)
end


"""Plot certified p_over_q vs angle for each R as a grid with rows = L and columns = D.

Usage:
  plot_grid_by_R("data", "figures/certify_byR")
"""
function plot_grid_by_R(data_root::AbstractString = "data", out_dir::AbstractString = "figures/certify_byR")
    nested = collect_nested(data_root)
    isempty(nested) && error("No data found under ", data_root)
    mkpath(out_dir)

    Ls = sort(collect(keys(nested)))
    Ds_set = Set{Int}()
    for L in Ls
        for D in keys(nested[L])
            push!(Ds_set, D)
        end
    end
    Ds = sort(collect(Ds_set))

    # collect all R values present anywhere
    Rs_set = Set{Float64}()
    for L in Ls, D in Ds
        for R in keys(nested[L][D])
            push!(Rs_set, R)
        end
    end
    Rs = sort(collect(Rs_set))
    isempty(Rs) && return

    for R in Rs
        nrows = max(1, length(Ls)); ncols = max(1, length(Ds))
        plt = plot(layout = (nrows, ncols), size = (600*ncols, 500*nrows),
                   left_margin = 25mm, right_margin = 10mm, top_margin = 10mm, bottom_margin = 10mm,
                   xtickfont = font(18), tickfont = font(15), guidefont = font(16))

        for (iL, L) in enumerate(Ls)
            for (jD, D) in enumerate(Ds)
                idx = (iL - 1) * ncols + jD
                ax = plt[idx]
                xlabel!(ax, "angle (rad)", fontsize=font(18))
                ylabel!(ax, "p/q", fontsize=font(18))
                title!(ax, @sprintf("L=%d  D=%d", L, D))

                # gather angle -> mean for this L,D,R by scanning files under L/D
                angles = Float64[]; means = Float64[]
                sample_dir = joinpath(data_root, @sprintf("L%d", L), @sprintf("D%d", D))
                for (root, dirs, files) in walkdir(sample_dir)
                    occursin(@sprintf("R%d", Int(R)), root) || continue
                    for f in files
                        startswith(f, "stats_") || continue
                        a = parse_angle_from_filename(basename(f))
                        a === nothing && continue
                        m = read_mean_certified(joinpath(root, f))
                        isnan(m) && continue
                        push!(angles, a); push!(means, m)
                    end
                end

                isempty(angles) && continue
                order = sortperm(angles)
                a_s = angles[order]; m_s = means[order]
                plot!(ax, a_s, m_s, label = false, marker = :circle)
                plot!(ax, yscale = :log10, ylim = (1e-1, 1e3))
                plot!(ax, xlim = (0, pi/2), xticks = ([0, pi/4, pi/2], [L"0", L"\frac{\pi}{4}", L"\frac{\pi}{2}"]))
            end
        end

        outpng = joinpath(out_dir, @sprintf("certify_byR_R%d.png", Int(R)))
        savefig(plt, outpng)
        println("Saved: ", outpng)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. plotting/plot_certify_byL.jl <data_root> [out_dir]")
        exit(1)
    end
    data_root = ARGS[1]
    out_dir = length(ARGS) >= 2 ? ARGS[2] : "figures"
    plot_grid_by_LD(data_root, out_dir)
    plot_grid_by_R(data_root, out_dir)
end
