using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: sample_certified
const TN = TensorNetworkQuantumSimulator

using ITensors
using CUDA
using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: ITensorNetworks

using Dictionaries: Dictionary

using Statistics

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

function main(nx::Int, ny::Int)
    g = named_grid((nx, ny))

    ITensors.disable_warn_order()

    println("Building the PEPS state with nx = $(nx), ny = $(ny) \n")

    sinds = TN.siteinds("S=1/2", g)
    nqubits = length(vertices(g))
    betas = Dictionary(collect(vertices(g)), [rand([0, pi / 4]) for v in vertices(g)])
    ψ0 = tensornetworkstate(ComplexF64, v -> (1 / sqrt(2)) * [1, exp(im * betas[v])], g, sinds)
    ψ0 = CUDA.cu(ψ0)

    layer = []
    append!(layer, ("Rzz", e, (pi / 2)) for e in edges(g))
    append!(layer, ("Rz", [v], (- degree(g, v) * pi / 2)) for v in vertices(g))

    layer = TN.toitensor(layer, sinds)
    t = @timed ψ, errors = apply_gates(layer, ψ0; apply_kwargs = (; maxdim = 100, cutoff = 1.0e-18, normalize_tensors = false), update_cache = true, verbose = false)
    
    println("State built exactly. It's bond dimension is $(ITensorNetworks.maxlinkdim(ψ)). \n")

    mps_bond_dimension = 16
    nsamples = 10
    println("Taking $nsamples samples from the state now in the X-basis by contracting the PEPS with a MPS bond dimension of $(mps_bond_dimension) \n")

    A_vertices = filter(v -> isodd(sum(v)), collect(vertices(g)))
    B_vertices = filter(v -> iseven(sum(v)), collect(vertices(g)))

    A_rotation_layer = [("H", [v]) for v in A_vertices]
    t = @timed ψ_rotated, errors = apply_gates(A_rotation_layer, ψ; apply_kwargs = (; maxdim = 100, cutoff = 1.0e-18, normalize_tensors = false), update_cache = false, verbose = false)
    bitstrings = sample_certified(ψ_rotated, nsamples; norm_mps_bond_dimension = mps_bond_dimension, projected_mps_bond_dimension = mps_bond_dimension, certification_mps_bond_dimension = 100)
    @show Statistics.std(first.(bitstrings))
    @show Statistics.mean(first.(bitstrings))
    e_A = measure_energy(g, last.(bitstrings), TN.siteinds(ψ), betas, A_vertices)

    B_rotation_layer = [("H", [v]) for v in B_vertices]
    t = @timed ψ_rotated, errors = apply_gates(B_rotation_layer, ψ; apply_kwargs = (; maxdim = 100, cutoff = 1.0e-18, normalize_tensors = false), update_cache = false, verbose = false)
    bitstrings = sample_certified(ψ_rotated, nsamples; norm_mps_bond_dimension = mps_bond_dimension, projected_mps_bond_dimension = mps_bond_dimension, certification_mps_bond_dimension = 100)
    @show Statistics.std(first.(bitstrings))
    @show Statistics.mean(first.(bitstrings))
    e_B = measure_energy(g, last.(bitstrings), TN.siteinds(ψ), betas, B_vertices)


    @show e_A / length(vertices(g)), e_B / length(vertices(g))
    return @show -(e_A + e_B) / length(vertices(g))

end

main(6, 6)
