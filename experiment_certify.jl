using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
# Try to load CUDA and detect availability; set HAVE_CUDA accordingly
# const HAVE_CUDA = try
# 	@eval begin
# 		using CUDA
# 	end
# 	CUDA.has_cuda()
# catch err
# 	@warn "CUDA not available or failed to load: $err"
# 	false
# end
HAVE_CUDA=false
using Adapt
using Printf
using JSON3
using NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: ITensorNetworks

using Dictionaries: Dictionary

using Statistics
using DelimitedFiles: writedlm

using Random

"""
Run one experiment at lattice size L x L with a given MPS bond dimension.

Arguments
- L::Int: linear lattice size (nx = ny = L)
- certification_bond_dimension::Int: MPS bond dimension used for contraction/sampling
- output_dir::Union{Nothing,String}: optional path to write a CSV with mean and std
- nsamples::Int: number of certified samples to draw (default 1000)
- seed::Union{Nothing,Int}: RNG seed; if provided, Random.seed!(seed) is called before any sampling

Outputs
- Prints mean and std of the sampled statistic to stdout
- If output_dir is set, writes a 1-row CSV with columns: metric,mean,std
"""
function main_certify(L::Int, sampling_bond_dimension::Int, certification_bond_dimension::Int, nsamples::Int = 1000, 
	seed::Union{Nothing, Int} = nothing; output_dir::Union{Nothing, String} = nothing,
	output_dir_R::Union{Nothing, String} = nothing)
	# Ensure deterministic randomness if seed is set (no environment fallback).
	if seed !== nothing
		Random.seed!(seed)
		@info "Seeding RNG with SEED=$(seed)"
	end
	nx=ny=L
	g = named_grid((nx, ny))
	ITensors.disable_warn_order()

	# If an output CSV path is provided and the file already exists, skip the expensive
	# experiment and return immediately to avoid re-running completed jobs.
	if isfile(output_dir_R * "/stats_1.57080.csv")
		@info "Output data exists at: $(output_dir_R). Skipping experiment."
		return
	end
	if !isfile(output_dir * "/stats_1.57080.csv")
		@info "Sample data does not exist exists at: $(output_dir). Skipping experiment."
		return
	end
	mkpath(output_dir_R)

	@info "Building the PEPS state with nx = $(nx), ny = $(ny)"
	flush(stderr)
	sinds = TN.siteinds("S=1/2", g)
	nqubits = length(vertices(g))
	betas = Dictionary(collect(vertices(g)), [rand([0, pi / 4]) for v in vertices(g)])
	ψ0 = tensornetworkstate(ComplexF64, v -> (1 / sqrt(2)) * [1, exp(im * betas[v])], g, sinds)

	layer = []
	append!(layer, ("Rzz", e, (pi / 2)) for e in edges(g))
	append!(layer, ("Rz", [v], (-degree(g, v) * pi / 2)) for v in vertices(g))

	layer = TN.toitensor(layer, sinds)
	t = @timed ψ, errors = apply_gates(layer, ψ0; apply_kwargs = (; maxdim = 100, cutoff = 1.0e-18, normalize_tensors = false), update_cache = true, verbose = false)

	@info "State built exactly. It's bond dimension is $(ITensorNetworks.maxlinkdim(ψ))."
	@info "Taking $nsamples samples from the state now by contracting the PEPS with a MPS bond dimension of $(certification_bond_dimension) for a range of Ry angles"
	flush(stderr)
	# Define angles from 0 to pi/2 in 20 steps
	angles = collect(range(0, stop = pi/2, length = 20))

	for ang in angles
		output_bitstrings = joinpath(output_dir, @sprintf("bitstrings_%0.5f.json", ang))
		bitstrings = Vector{Dictionary{Tuple{Int64, Int64}, Int64}}
		open(output_bitstrings) do f
			bitstrings = JSON3.read(f, Vector{Dictionary{Tuple{Int64, Int64}, Int64}});
		end
		output_stats = joinpath(output_dir, @sprintf("stats_%0.5f.csv", ang))
		poverqs = ComplexF64[]
		logqs = Float64[]
		open(output_stats, "r") do io
			# skip header
			try
				readline(io)
			catch
				# empty file
			end
			for line in eachline(io)
				line = strip(line)
				isempty(line) && continue
				cols = split(line, ',')
				if length(cols) >= 2
					push!(poverqs, parse(ComplexF64, strip(cols[1])))
					push!(logqs, parse(Float64, strip(cols[2])))
				end
			end
		end
		probs_and_bitstrings = NamedTuple[]
		for j in 1:nsamples
			push!(probs_and_bitstrings, (poverq = poverqs[j], logq = logqs[j], bitstring = bitstrings[j]))
		end
		@info "Loaded bitstrings"

		@info "Applying angle $(ang)"
		flush(stderr)
		# Skip case where angle is close to zero.
		rotation_layer = vcat([("Z", [v]) for v in vertices(g)], [("Ry", [v], ang) for v in vertices(g)])
		t = @timed ψ_rotated, errors = apply_gates(rotation_layer, ψ; apply_kwargs = (; maxdim = 100, cutoff = 1.0e-18, normalize_tensors = false), update_cache = false, verbose = false)
		# For each angle, apply a Z to every qubit first, then an Ry(ang) to every qubit
		# Move the rotated state to the GPU only when CUDA is available on this node.
		ψ_rotated = gauge_and_scale(ψ_rotated)
		if HAVE_CUDA
			ψ_rotated = CUDA.cu(ψ_rotated)
			ψ_rotated = adapt(CuArray{ComplexF64})(ψ_rotated)
			@info "Put state on GPU"
		end
		R = 10
		outs = TN.certify_samples(ψ_rotated, probs_and_bitstrings; certification_mps_bond_dimension = certification_bond_dimension, symmetrize_and_normalize = false)
		certified_poverq = [o.poverq for o in outs]
		# error("HRE")
		# 	poverqs = getfield.(outs, :poverq)
		# logqs = getfield.(outs, :logq)
		# bitstrings = getfield.(outs, :bitstring)
		if output_dir_R !== nothing
			# use the loop variable `ang`; format to 3 decimal places
			output_stats = joinpath(output_dir_R, @sprintf("stats_%0.5f.csv", ang))
			open(output_stats, "w") do io
				println(io, "certified_poverq")
				for poverq in zip(certified_poverq)
					println(io, "$(poverq)")
				end
			end			
			println("Saved results to: $(output_stats)")
		end
	end



end

# Allow running from the command line as: julia --project=. experiment.jl L D [output_dir] [nsamples]
if abspath(PROGRAM_FILE) == @__FILE__
	if length(ARGS) < 2
		println("Usage: julia --project=. experiment.jl L D R [nsamples] [seed] [output_dir]")
		println("Example: julia --project=. experiment.jl 6 64 data/L6_D64_NS1000.csv 1000 12345")
		# Run a tiny default for interactive exploration
	else
		L = parse(Int, ARGS[1])
		D = parse(Int, ARGS[2])
		RE = parse(Int, ARGS[3])
		nsamples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
		seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
		output_dir = length(ARGS) >= 6 ? ARGS[6] : nothing
		main_sample(L, D, R, seed, nsamples; output_dir = output_dir)
	end
end
