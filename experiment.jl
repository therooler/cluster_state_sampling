using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
# using CUDA
using NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: ITensorNetworks

using Dictionaries: Dictionary

using Statistics
using DelimitedFiles: writedlm

using Random

function measure_energy(g::NamedGraph, bitstrings::Vector{<:Dictionary}, siteinds::Dictionary, betas::Dictionary, sublattice::Vector)
    return sum(measure_energy(g, b, siteinds, betas, sublattice) for b in bitstrings) / length(bitstrings)
end

#Measure the energy of a bitstring, vertices in the sublattice are rotated into the X basis, the rest in Z
function measure_energy(g::NamedGraph, bitstring::Dictionary, siteinds::Dictionary, betas::Dictionary, sublattice::Vector)
    mapped_bitstring = map(b -> b == 0 ? 1 : -1, bitstring)
    e = 0
    for v in sublattice
        if abs(betas[v] - pi / 4) < 1.0e-8
            #Sqrt(2) here because we want to project onto the rotated X axis spanned by (|0> \pm e^{i pi/4} |1>) / sqrt(2)
            z = mapped_bitstring[v] == -1 ? -sqrt(2) : sqrt(2)
        elseif abs(betas[v]) < 1.0e-8
            z = mapped_bitstring[v] == -1 ? -1 : 1
        else
            error("Beta value not supported")
        end
        e_local = prod([mapped_bitstring[vn] for vn in neighbors(g, v)]) * z
        e += e_local
    end
    return e
end

"""
Run one experiment at lattice size L x L with a given MPS bond dimension.

Arguments
- L::Int: linear lattice size (nx = ny = L)
- mps_bond_dimension::Int: MPS bond dimension used for contraction/sampling
- output_csv::Union{Nothing,String}: optional path to write a CSV with mean and std
- nsamples::Int: number of certified samples to draw (default 1000)
- seed::Union{Nothing,Int}: RNG seed; if provided, Random.seed!(seed) is called before any sampling

Outputs
- Prints mean and std of the sampled statistic to stdout
- If output_csv is set, writes a 1-row CSV with columns: metric,mean,std
"""
function main(L::Int, mps_bond_dimension::Int, output_csv::Union{Nothing,String}=nothing, nsamples::Int=1000; seed::Union{Nothing,Int}=nothing)
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
    if output_csv !== nothing && isfile(output_csv)
        @info "Output CSV already exists at: $(output_csv). Skipping experiment."
        return
    end

    @info "Building the PEPS state with nx = $(nx), ny = $(ny)"
    flush(stderr)
    sinds = TN.siteinds("S=1/2", g)
    nqubits = length(vertices(g))
    betas = Dictionary(collect(vertices(g)), [rand([0, pi / 4]) for v in vertices(g)])
    ψ0 = tensornetworkstate(ComplexF64, v -> (1 / sqrt(2)) * [1, exp(im * betas[v])], g, sinds)
    # ψ0 = CUDA.cu(ψ0)

    layer = []
    append!(layer, ("Rzz", e, (pi / 2)) for e in edges(g))
    append!(layer, ("Rz", [v], (-degree(g, v) * pi / 2)) for v in vertices(g))

    layer = TN.toitensor(layer, sinds)
    t = @timed ψ, errors = apply_gates(layer, ψ0; apply_kwargs=(; maxdim=100, cutoff=1.0e-18, normalize_tensors=false), update_cache=true, verbose=false)

    @info "State built exactly. It's bond dimension is $(ITensorNetworks.maxlinkdim(ψ))."
    @info "Taking $nsamples samples from the state now by contracting the PEPS with a MPS bond dimension of $(mps_bond_dimension) for a range of Ry angles"
    flush(stderr)
    # Define angles from 0 to pi/2 in 20 steps
    angles = collect(range(0, stop=pi/2, length=20))
    results = Vector{Tuple{Float64, Float64, Float64}}()

    for ang in angles
        
        @info "Applying angle $(ang)"
        flush(stderr)
        # Skip case where angle is close to zero.
        rotation_layer = vcat([("Z", [v]) for v in vertices(g)], [("Ry", [v], ang) for v in vertices(g)])
        t = @timed ψ_rotated, errors = apply_gates(rotation_layer, ψ; apply_kwargs=(; maxdim=100, cutoff=1.0e-18, normalize_tensors=false), update_cache=false, verbose=false)
        # For each angle, apply a Z to every qubit first, then an Ry(ang) to every qubit


        bitstrings = TN.sample_certified(ψ_rotated, nsamples; norm_mps_bond_dimension=mps_bond_dimension, projected_mps_bond_dimension=mps_bond_dimension, certification_mps_bond_dimension=100)

        # Compute summary statistics of the sampled quantity currently used in this experiment
        xs = first.(bitstrings)
        mu = Statistics.mean(xs)
        sigma = Statistics.std(xs)
        @show ang
        @show sigma
        @show mu

        push!(results, (ang, mu, sigma))
    end

    if output_csv !== nothing
        # Ensure the output directory exists
        mkpath(dirname(output_csv))
        # Write a CSV with rows for each angle: metric,angle,mean,std
        open(output_csv, "w") do io
            println(io, "metric,angle,mean,std")
            for (ang, mu, sigma) in results
                println(io, "sample_first,$(ang),$(mu),$(sigma)")
            end
        end
        println("Saved results to: $(output_csv)")
    end

end

# Allow running from the command line as: julia --project=. experiment.jl L D [output_csv] [nsamples]
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia --project=. experiment.jl L D [output_csv] [nsamples] [seed]")
        println("Example: julia --project=. experiment.jl 6 64 data/L6_D64_NS1000.csv 1000 12345")
        # Run a tiny default for interactive exploration
        main(6, 6)
    else
        L = parse(Int, ARGS[1])
        D = parse(Int, ARGS[2])
        output_csv = length(ARGS) >= 3 ? ARGS[3] : nothing
        nsamples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
        seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
        main(L, D, output_csv, nsamples; seed=seed)
    end
end
