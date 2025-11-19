#!/usr/bin/env julia
# Run multiple experiments in a single Julia process to amortize precompilation and JIT costs.
# Usage:
#   julia --project=. batch_experiments.jl "L_list" "D_list" [nsamples] [output_dir] [seed]
# where L_list and D_list are comma-separated integers, e.g. "4,6,8" and "32,64".
# Example:
#   julia --project=. batch_experiments.jl "2,3" "4,8" 10 data 12345

using Printf
using Dates
using Random

# Bring in the `main` function from experiment.jl without triggering its CLI entrypoint
include("experiment.jl")

function parse_list(s::AbstractString)
    isempty(s) && error("Empty list argument")
    return parse.(Int, split(strip(s), ","))
end

function batch_main(Ls::Vector{Int}, Ds::Vector{Int}; nsamples::Int=1000, outdir::AbstractString="data", seed::Union{Nothing,Int}=nothing)
    mkpath(outdir)
    if seed !== nothing
        Random.seed!(seed)
        @info "Seeding RNG for batch" seed
    end
    @info "Starting batch" Ls Ds nsamples outdir seed
    for L in Ls
        for D in Ds
            outfile = seed === nothing ?
                joinpath(outdir, @sprintf("L%d_D%d_NS%d.csv", L, D, nsamples)) :
                joinpath(outdir, @sprintf("L%d_D%d_NS%d_SEED%d.csv", L, D, nsamples, seed))
            @info "Running" L D nsamples outfile
            try
                # We set the RNG once above for reproducibility across the batch.
                # Pass seed=nothing to avoid re-seeding per run.
                main(L, D, outfile, nsamples; seed=seed)
            catch err
                @error "Experiment failed" L D exception=(err, catch_backtrace())
            end
        end
    end
    @info "Batch complete" outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia --project=. batch_experiments.jl \"L_list\" \"D_list\" [nsamples] [output_dir] [seed]")
        println("Example: julia --project=. batch_experiments.jl \"2,3\" \"4,8\" 10 data 12345")
        # tiny default
        batch_main([2], [4]; nsamples=4, outdir="data")
    else
        Ls = parse_list(ARGS[1])
        Ds = parse_list(ARGS[2])
        nsamples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
        outdir = length(ARGS) >= 4 ? ARGS[4] : "data"
        seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
        batch_main(Ls, Ds; nsamples=nsamples, outdir=outdir, seed=seed)
    end
end
