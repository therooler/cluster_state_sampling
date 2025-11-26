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

# Detect CPU name to produce an informative, sanitized sysimage filename.
function detect_cpu_name()
	# Allow explicit override via environment variable (conventional for Julia builds)
	if haskey(ENV, "JULIA_CPU_TARGET") && !isempty(ENV["JULIA_CPU_TARGET"]) 
		return ENV["JULIA_CPU_TARGET"]
	end

	# Try /proc/cpuinfo (works on Linux)
	try
		if isfile("/proc/cpuinfo")
			for line in eachline("/proc/cpuinfo")
				if occursin("model name", line)
					parts = split(line, ':', limit=2)
					if length(parts) == 2
						return strip(parts[2])
					end
				end
			end
		end
	catch
		# fall through to lscpu
	end

	# Fallback to lscpu if available
	try
		out = readchomp(`lscpu`)
		for ln in split(out, '\n')
			if occursin("Model name", ln) || occursin("model name", ln)
				parts = split(ln, ':', limit=2)
				if length(parts) == 2
					return strip(parts[2])
				end
			end
		end
	catch
		# give up
	end

	return "unknowncpu"
end

function sanitize_cpu_name(name::AbstractString)
	s = lowercase(name)
	# replace whitespace with underscore
	s = replace(s, r"\s+" => "_")
	# remove characters that are unsafe for filenames
	s = replace(s, r"[^a-z0-9_\-]" => "")
	# collapse multiple underscores
	s = replace(s, r"_+" => "_")
	s = strip(s, '_')
	return isempty(s) ? "unknowncpu" : s
end

# choose sysimage filename based on detected CPU
detected_cpu = detect_cpu_name()
cpu_tag = sanitize_cpu_name(detected_cpu)
sysimage_name = "sys_bp_cluster_state_$(cpu_tag).so"
sysimage_path = joinpath(@__DIR__, "..", sysimage_name)

println("Creating sysimage at: ", sysimage_path)
println("Precompile execution file: ", precompile_file)
println("Detected CPU: ", detected_cpu, " -> using tag: ", cpu_tag)

# List of modules to include in the sysimage. Adjust if you want fewer/more.
mods = [
	:TensorNetworkQuantumSimulator,
	:ITensors,
	:ITensorNetworks,
	:NamedGraphs,
	:Graphs,
	:Dictionaries,
	:Measures,
	:Plots,
	:ColorSchemes,
	:JSON3,
	:LaTeXStrings,
]

# Allow opting out of CUDA-related modules at build time. Set SKIP_CUDA=1 to remove
# any modules whose symbol name contains "CUDA" from the precompile list.
mods = filter(m -> !occursin("CUDA", String(m)), mods)

# Create sysimage (this can take many minutes). The function will print progress.
PackageCompiler.create_sysimage(mods; sysimage_path = sysimage_path, precompile_execution_file = precompile_file)

println("Sysimage creation finished: ", sysimage_path)
