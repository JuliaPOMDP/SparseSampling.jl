module SparseSampling

using Random
using POMDPs
using Parameters
using StaticArrays
using POMDPModelTools

export
    SSOptions,
    SparseSamplingSolver,
    POWSSOptions,
    SSPlanner,
    valuepairs

abstract type AbstractSSOptions end

# accessors
maxdepth(opt::AbstractSSOptions) = opt.maxdepth
width(opt::AbstractSSOptions) = opt.width

@with_kw struct SSOptions <: AbstractSSOptions
    maxdepth::Int
    width::Int
end

@with_kw struct POWSSOptions <: AbstractSSOptions
    maxdepth::Int
    width::Int
end

struct SparseSamplingSolver
    opt::AbstractSSOptions
    rng::AbstractRNG
end

SparseSamplingSolver(opt::AbstractSSOptions; rng=Random.GLOBAL_RNG) = SparseSamplingSolver(opt, rng)
SparseSamplingSolver(;rng=Random.GLOBAL_RNG, kwargs...) = SparseSamplingSolver(SSOptions(kwargs...), rng)

function POMDPs.solve(s::SparseSamplingSolver, m::Union{MDP,POMDP})
    if s.opt isa POWSSOptions
        @assert !(Nothing <: statetype(m)) "POWSS does not support problems where the state can be nothing (this can be fixed by adding Some in appropriate places)."
        # @assert !(Nothing <: obstype(m)) "POWSS does not support problems where the observation can be nothing (this can be fixed by adding Some in appropriate places)."
    end
    SSPlanner(m, s.opt, s.rng)
end
struct SSPlanner{M<:Union{MDP,POMDP},O<:AbstractSSOptions,R<:AbstractRNG}
    m::M
    opt::O
    rng::R
end

include("pomdp_sparse_sampling.jl")

end # module
