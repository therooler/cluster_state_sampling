using LinearAlgebra, SparseArrays
using NamedGraphs
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: vertices, edges, src, dst, neighbors, degree

global const Z = sparse([1 0; 0 -1])
global const I2 = sparse(I, 2, 2)
global const X = sparse([0 1; 1 0])

function X_rotated(beta::Number)
    return sparse([0 exp(- im * beta); exp(im * beta) 0])
end

"Kronecker product for a vector of small (sparse) matrices."
function kronall(ops::Vector{SparseMatrixCSC{T,Int}}) where T
    K = ops[1]
    @inbounds for k in 2:length(ops)
        K = kron(K, ops[k])
    end
    return K
end

"Embed a 2x2 operator op acting on `site` (1-based) within N spins."
function embed_single(op::SparseMatrixCSC, N::Int, site::Int)
    ops = Vector{SparseMatrixCSC{eltype(op),Int}}(undef, N)
    @inbounds for n in 1:N
        ops[n] = (n == site) ? op : I2
    end
    return kronall(ops)
end

"Embed a two-site operator opA⊗opB acting on (i,j) with i<j in N spins."
function embed_pair(opA::SparseMatrixCSC, i::Int, opB::SparseMatrixCSC, j::Int, N::Int)
    ops = Vector{SparseMatrixCSC{eltype(opA),Int}}(undef, N)
    @inbounds for n in 1:N
        ops[n] = n == i ? opA : n == j ? opB : I2
    end
    return kronall(ops)
end

"""
Build H = (sum Z_i Z_j) - (sum Z_i)
"""
function build_H(g::NamedGraph, site_map::Dict)
    N = length(vertices(g))
    H = spzeros(ComplexF64, 2^N, 2^N)

    for e in edges(g)
        si, sj = site_map[src(e)], site_map[dst(e)]
        H += embed_pair(Z, si, Z, sj, N)
    end

    # - sum Z_i
    for v in vertices(g)
        H -= degree(g, v) * embed_single(Z, N, site_map[v])
    end

    return H
end

"Convenience single-qubit kets."
ket0() = sparse([1.0; 0.0])
ket1() = sparse([0.0; 1.0])
ketp(beta::Number) = (ket0() .+ exp(1*im*beta).*ket1()) ./ √2  # |+>

"Construct a product state from a vector of 2-vectors."
function product_state(kets::Vector{SparseVector{ComplexF64,Int}})
    ψ = SparseVector{ComplexF64,Int}(kets[1])
    @inbounds for k in 2:length(kets)
        ψ = kron(ψ, kets[k])
    end
    # promote to Complex for unitary evolution
    return ComplexF64.(ψ)
end

"Apply U = exp(-i * (π/4) * H) to initial state ψ0."
function evolve_with_circuit(H::SparseMatrixCSC, ψ0::AbstractVector{<:Complex})
    θ = π/4
    # Use dense exp for reliability on modest sizes; ED scales as 2^N
    U = exp(im * θ * Matrix(H))
    return U * ψ0
end

"Measure probabilities in the computational basis."
function probs(ψ::AbstractVector{<:Complex})
    return real.(abs2.(ψ))
end

function expect(ψ::AbstractVector{<:Complex}, ops::Vector, verts::Vector, site_map::Dict)
    N = length(keys(site_map))
    op = prod(embed_single(op, N, site_map[v]) for (op, v) in zip(ops, verts))
    return ψ' * op * ψ
end

function energy(ψ::AbstractVector{<:Complex}, g::NamedGraph, site_map::Dict,betas::Dict)
    e= 0
    for v in vertices(g)
        vn = neighbors(g, v)
        ops = vcat([Z for _ in 1:length(vn)], [X_rotated(betas[v])])
        verts = vcat(vn, [v])
        e += expect(ψ, ops, verts, site_map)
    end

    return e
end

# ----------------- Example usage -----------------
function main()
    nx, ny = 3,3  # number of spins (adjust as needed)
    N = nx*ny
    g = named_grid((nx, ny))

    site_map = Dict([v => i for (i,v) in enumerate(vertices(g))])
    betas = Dict([v => rand([0.0, 0.0]) for v in collect(vertices(g))])

    # Example initial product state: |+>^{⊗N}
    ψ0 = product_state([ketp(betas[v]) for v in vertices(g)])
    # for c in ψ0
    #     println(c)
    # end
    # Build H with all-to-all ZZ (default). To use nearest-neighbor:
    H = build_H(g, site_map)

    ψf = evolve_with_circuit(H, ψ0)
    # for c in ψf
    #     println(c)
    # end
    println("Final energy density with nx = $(nx) and ny = $(ny) is $(energy(ψf, g, site_map, betas))")

end

main()