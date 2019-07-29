function estimate_v(opt::AbstractSSOptions, m::POMDP, belief::AbstractVector, depth::Int, rng::AbstractRNG)
    if depth == maxdepth(opt)
        return 0.0
    end

    return maximum(estimate_q(opt, m, belief, a, depth, rng) for a in actions(m))
end

function estimate_q(opt::SSOptions, m::POMDP, belief::AbstractVector, a, depth::Int, rng::AbstractRNG)
    qsum = 0.0
    children = Dict{obstype(m), Vector{statetype(m)}}()

    for i in 1:width(opt)
        s = belief[mod1(i, length(belief))]
        sp, o, r = generate_sor(m, s, a, rng)
        qsum += r
        if haskey(children, o)
            push!(children[o], sp)
        else
            children[o] = [sp]
        end
    end

    for c in values(children)
        vp = estimate_v(opt, m, c, depth+1, rng)
        qsum += discount(m)*vp*length(c)
    end
    return qsum/width(opt)
end

function actionvaluepairs(p::SSPlanner{M}, b) where M <: POMDP
    belief = collect(rand(p.rng, b) for i in 1:p.opt.width)
    eq(a) = estimate_q(p.opt, p.m, belief, a, 0, p.rng)
    return (a=>eq(a) for a in actions(p.m))
end

function POMDPs.action(p::SSPlanner{M}, b) where M <:POMDP
    avps = collect(actionvaluepairs(p, b))
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
    
    predictions = MVector{opt.width, statetype(m)}(undef)
    observations = MVector{opt.width, obstype(m)}(undef)

    wsum = 0.0
    for i in 1:width(opt)
        s, w = weighted_state(belief, i)
        sp, o, r = generate_sor(m, s, a, rng)
        predictions[i] = sp
        observations[i] = o
        wsum += w
        q += w*r
    end

    nextbelief = MVector{opt.width, Pair{statetype(m), Float64}}(undef)

    for i in 1:width(opt) # needs to be a separate for loop because it needs all predictions
        o = observations[i]
        _, w = weighted_state(belief, i)
        for i in 1:width(opt)
            s, w = weighted_state(belief, i)
            sp = predictions[i]
            nextbelief[i] = sp=>w*obs_weight(m, s, a, sp, o)
        end
        vp = estimate_v(opt, m, nextbelief, depth+1, rng)
        q += w*discount(m)*vp
    end
    return q/wsum
end

weighted_state(b::AbstractVector, i) = b[i]=>1/length(b)
weighted_state(b::AbstractVector{Pair{S,Float64}}, i) where {S} = b[i]
