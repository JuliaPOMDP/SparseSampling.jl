module SparseSampling

using Random
using POMDPs
using Parameters

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

SparseSamplingSolver(opt::AbstractSSOptions; rng=Random.GLOBAL_RNG) = SparseSamplingSolver(SSOptions(kwargs...), rng)
SparseSamplingSolver(;rng=Random.GLOBAL_RNG, kwargs...) = SparseSamplingSolver(SSOptions(kwargs...), rng)

POMDPs.solve(s::SparseSamplingSolver, m::Union{MDP,POMDP}) = SSPlanner(m, s.opt, s.rng)

struct SSPlanner{M<:Union{MDP,POMDP},O<:AbstractSSOptions,R<:AbstractRNG}
    m::M
    opt::O
    rng::R
end

include("pomdp_sparse_sampling.jl")

end # module
