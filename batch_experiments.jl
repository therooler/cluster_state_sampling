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
include("experiment_sample.jl")
include("experiment_certify.jl")

function parse_list(s::AbstractString)
	isempty(s) && error("Empty list argument")
	return parse.(Int, split(strip(s), ","))
end

function batch_main_sample(Ls::Vector{Int}, Ds::Vector{Int}; nsamples::Int = 1000, outdir::AbstractString = "data", seed::Union{Nothing, Int} = nothing)
	mkpath(outdir)
	if seed !== nothing
		Random.seed!(seed)
		@info "Seeding RNG for batch" seed
	end
	@info "Starting batch sampling" Ls Ds nsamples outdir seed
	for L in Ls
		for D in Ds
			outfile = seed === nothing ?
					  joinpath(outdir, @sprintf("L%d/D%d/NS%d", L, D, nsamples)) :
					  joinpath(outdir, @sprintf("L%d/D%d/NS%d/SEED%d", L, D, nsamples, seed))
			@info "Running" L D nsamples outfile
			try
				# We set the RNG once above for reproducibility across the batch.
				# Pass seed=nothing to avoid re-seeding per run.
				main_sample(L, D, nsamples, seed; output_dir = outfile)
			catch err
				@error "Experiment failed" L D exception=(err, catch_backtrace())
			end
		end
	end
	@info "Batch complete" outdir
end

function batch_main_certify(Ls::Vector{Int}, Ds::Vector{Int}, Rs::Vector{Int}; nsamples::Int = 1000, outdir::AbstractString = "data", seed::Int = 0)
	mkpath(outdir)
	if seed !== nothing
		Random.seed!(seed)
		@info "Seeding RNG for batch" seed
	end
	@info "Starting batch certification" Ls Ds nsamples outdir seed
	for L in Ls
		for D in Ds
			for R in Rs
				outfile = joinpath(outdir, @sprintf("L%d/D%d/NS%d/SEED%d/", L, D, nsamples, seed))
				outfile_R = joinpath(outdir, @sprintf("L%d/D%d/NS%d/SEED%d/R%d", L, D, nsamples, seed, R))

				@info "Running" L D R nsamples outfile outfile_R
				try
					# We set the RNG once above for reproducibility across the batch.
					# Pass seed=nothing to avoid re-seeding per run.
					main_certify(L, D, R, nsamples, seed; output_dir = outfile, output_dir_R=outfile_R)
				catch err
					@error "Experiment failed" L D exception=(err, catch_backtrace())
				end
			end
		end
	end
	@info "Batch complete" outdir
end
if abspath(PROGRAM_FILE) == @__FILE__
	if length(ARGS) < 2
		println("Usage: julia --project=. batch_experiments.jl \"L_list\" \"D_list\" \"sample_or_certify\" [nsamples] [seed] [output_dir] ")
		println("Example: julia --project=. batch_experiments.jl \"2,3\" \"4,8\" 10 12345 data")
		# tiny default
	else
		Ls = parse_list(ARGS[1])
		Ds = parse_list(ARGS[2])
		sample_or_certify = parse_list(ARGS[3])
		nsamples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
		seed = length(ARGS) >= 5 ? parse(Int, ARGS[6]) : nothing
		outdir = length(ARGS) >= 6 ? ARGS[6] : "data"
		if sample_or_certify=="sample"
			batch_main_sample(Ls, Ds; nsamples = nsamples, outdir = outdir, seed = seed)
		elseif sample_or_certify=="certify"
			if length(ARGS!=7)
				println("ARGS[7] must contain Rs for \"certify\"")
			else
				Rs = parse_list(ARGS[7])
				batch_main_certify(Ls, Ds, Rs; nsamples = nsamples, outdir = outdir, seed = seed)
			end
		else
			println("ARGS[3] must be \"sample\" or \"certify\"")
			println("Usage: julia --project=. batch_experiments.jl \"L_list\" \"D_list\" \"sample_or_certify\" [nsamples] [seed] [output_dir] ")
			println("Example: julia --project=. batch_experiments.jl \"2,3\" \"4,8\" 10 12345 data")
		end
	end
end
