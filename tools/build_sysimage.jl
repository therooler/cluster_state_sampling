#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
try
	proj = Pkg.project()
	println("Project: ", get(proj, "name", "(unknown)"))
catch
	println("Project: (unable to read project metadata)")
end

using PackageCompiler

sysimage_path = joinpath(@__DIR__, "..", "sys_bp_cluster_state.so")
precompile_file = joinpath(@__DIR__, "precompile_execution.jl")

println("Creating sysimage at: ", sysimage_path)
println("Precompile execution file: ", precompile_file)

# List of modules to include in the sysimage. Adjust if you want fewer/more.
mods = [
	:TensorNetworkQuantumSimulator,
	:ITensors,
	:ITensorNetworks,
	:CUDA,
	:NamedGraphs,
	:Graphs,
	:Dictionaries,
	:Measures,
	:Plots,
	:Adapt,
	:ColorSchemes,
	:JSON3,
	:LaTeXStrings,
]

# Create sysimage (this can take many minutes). The function will print progress.
PackageCompiler.create_sysimage(mods; sysimage_path = sysimage_path, precompile_execution_file = precompile_file)

println("Sysimage creation finished: ", sysimage_path)
