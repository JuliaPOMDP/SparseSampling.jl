function estimate_v(opt::AbstractSSOptions, m::POMDP, belief::AbstractVector, depth::Int, rng::AbstractRNG)
    if depth == maxdepth(opt)
        return 0.0
    end

    as = actions(m)
    qs = MVector{length(as), Float64}(undef)
    for (i,a) in enumerate(as)
        qs[i] = estimate_q(opt, m, belief, a, depth, rng)
    end
    return maximum(qs)
end

function estimate_q(opt::SSOptions, m::POMDP, belief::AbstractVector, a, depth::Int, rng::AbstractRNG)
    qsum = 0.0
    children = Dict{obstype(m), Vector{statetype(m)}}()

    for i in 1:width(opt)
        s = belief[mod1(i, length(belief))]
        if !isterminal(m, s)
            sp, o, r = gen(DDNOut(:sp,:o,:r), m, s, a, rng)
            qsum += r
            if haskey(children, o)
                push!(children[o], sp)
            else
                children[o] = [sp]
            end
        end
    end

    for c in values(children)
        vp = estimate_v(opt, m, c, depth+1, rng)
        qsum += discount(m)*vp*length(c)
    end
    return qsum/width(opt)
end

function valuepairs(p::SSPlanner{M}, b) where M <: POMDP
    belief = collect(rand(p.rng, b) for i in 1:p.opt.width)
    eq(a) = estimate_q(p.opt, p.m, belief, a, 0, p.rng)
    return (a=>eq(a) for a in actions(p.m))
end

function POMDPs.action(p::SSPlanner{M}, b) where M <:POMDP
    avps = collect(valuepairs(p, b))
    best = avps[1]
    for av in avps[2:end]
        if last(av) > last(best)
            best = av
        end
    end
    return first(best)
end

# for POWSS, belief is either a vector of states, in which case each is equally weighted, or a vector of state=>weight pairs
function estimate_q(opt::POWSSOptions, m::POMDP, belief::AbstractVector, a, depth::Int, rng::AbstractRNG)
    q = 0.0
    
    predictions = Vector{statetype(m)}(undef, opt.width)
    observations = Vector{obstype(m)}(undef, opt.width)

    allterminal = true
    wsum = 0.0
    for i in 1:width(opt)
        s, w = weighted_state(belief, i)
        if !isterminal(m, s)
            allterminal = false
            sp, o, r = gen(DDNOut(:sp,:o,:r), m, s, a, rng)
            predictions[i] = sp
            observations[i] = o
            q += w*r
        end
        wsum += w
    end
    if allterminal
        return 0.0
    end

    nextbelief = Vector{Pair{statetype(m), Float64}}(undef, opt.width)

    for i in 1:width(opt) # needs to be a separate for loop because it needs all predictions
        s, ow = weighted_state(belief, i)
        if !isterminal(m, s)
            o = observations[i]
            for j in 1:width(opt)
                s, w = weighted_state(belief, j)
                if isterminal(m, s)
                    nextbelief[j] = s=>0.0
                else
                    sp = predictions[j]
                    nextbelief[j] = sp=>w*obs_weight(m, s, a, sp, o)
                end
            end
            vp = estimate_v(opt, m, nextbelief, depth+1, rng)
            q += ow*discount(m)*vp
        end
    end
    return q/wsum
end

weighted_state(b::AbstractVector, i) = b[i]=>1/length(b)
weighted_state(b::AbstractVector{Pair{S,Float64}}, i) where {S} = b[i]
