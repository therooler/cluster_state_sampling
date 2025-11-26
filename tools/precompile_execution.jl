println("Running precompile execution to warm PackageCompiler cache...")
try
    # Load the packages used by the project so precompilation records are created
    using TensorNetworkQuantumSimulator
    using ITensors
    using ITensorNetworks
    using NamedGraphs
    using Graphs
    using Dictionaries
    using Measures
    using Plots
    using PackageCompiler
	using ColorSchemes
	using JSON3
	using LaTeXStrings
    println("Finished using target packages for precompilation")
catch e
    @warn "Precompile execution encountered an error: $e"
    rethrow()
end
