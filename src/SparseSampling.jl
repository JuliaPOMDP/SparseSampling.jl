module SparseSampling

struct SSOptions
    maxdepth::Int
    width::Int
end

struct SSPlanner{M<:Union{MDP,POMDP},R<:AbstractRNG}
    m::M
    opt::SSOptions
    rng::R
end

function estimate_v(m::POMDP, belief::Vector, opt::SSOptions, rng::AbstractRNG)
    if depth(h) == p.maxdepth
        return 0.0
    end

    vstar = -Inf
    for a in actions(m)
        q = estimate_q(m, belief, a, opt, rng)
        vstar = max(q, vstar)
    end

    return vstar
end

function estimate_q(m::POMDP, belief::Vector, a, opt::SSOptions, rng::AbstractRNG)
    q = 0.0
    children = Dict{obstype(m), Vector{statetype(m)}}()

    for i in 1:p.width
        s = belief[mod1(i, length(belief))]
        sp, o, r = generate_sor(m, s, a, p.rng)
        q += r
        if haskey(children, o)
            push!(children[o], s)
        else
            children[o] = [s]
        end
    end

    for c in values(children)
        vp = estimate_v(p, m, c)
        q += discount(m)*vp*length(c)/p.width
    end
    return q
end

function action(p::SSPlanner, b)
    qstar = -Inf
    astar::actiontype(p.m)
    belief = collect(rand(rng, b) for i in 1:p.opt.width)
    for a in actions(p.m)
        q = estimate_q(p.m, belief, a, p.opt, p.rng)
        if q >= qstar
            qstar = q
            astar = a
        end
    end
    return astar
end

end # module
