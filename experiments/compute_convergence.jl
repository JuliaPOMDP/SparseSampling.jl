using Distributed
using ContinuousObservationToyPOMDPs
@everywhere using SparseSampling
using QMDP
@everywhere using POMDPs
using BeliefUpdaters
using POMDPPolicies
using IncrementalPruning
using SARSOP
using DataFrames
using Statistics
using POMDPModelTools
using ProgressMeter
using Dates
using CSV
using POMCPOW

n = 200
maxwidth = 41
widthstep = 4
depth = 3
p_correct = 0.85

datestring = Dates.format(now(), "e_d_u_Y_HH_MM")

m = TimedCOTigerPOMDP(horizon=depth, p_correct=p_correct)
qp = solve(QMDPSolver(), m)
b = initialize_belief(updater(qp), initialstate_distribution(m))
@show qp.action_map
@show actionvalues(qp, b)

dm = TimedDOTigerPOMDP(horizon=depth, p_correct=p_correct)
op  = solve(SARSOPSolver(), dm)
# op  = solve(PruneSolver(), dm)
b = initialize_belief(updater(op), initialstate_distribution(dm))
@show op.action_map
@show actionvalues(op, b)

recalc = true
# if !@isdefined recalc
#     recalc = true
# end

if recalc
    results = DataFrame(mean=Float64[], std=Float64[], sem=Float64[], max=Float64[], min=Float64[], width=Int[], depth=Int[], solver=String[], action=Symbol[])

    for width in 1:widthstep:maxwidth

        ad = Dict{Symbol, Vector{Float64}}(a=>Float64[] for a in actions(m))
        @show width
        ssresults = @showprogress pmap(1:n) do i
            opt = SSOptions(depth, width)
            p = solve(SparseSamplingSolver(opt), m)
            return collect(valuepairs(p, b))
        end
        for avpairs in ssresults
            for (a, v) in avpairs
                push!(ad[a], v)
            end
        end
        for (a, vs) in ad
            sem = std(vs)/sqrt(n)
            mn = mean(vs)
            push!(results, (mean=mn,
                            std=std(vs),
                            sem=sem,
                            max=mn+3*sem,
                            min=mn-3*sem,
                            width=width,
                            depth=depth,
                            solver="POSS",
                            action=a
                           ))
        end
        
        ad = Dict{Symbol, Vector{Float64}}(a=>Float64[] for a in actions(m))
        powresults = @showprogress pmap(1:n) do i
            powopt = POWSSOptions(depth, width)
            powp = solve(SparseSamplingSolver(powopt), m)
            return collect(valuepairs(powp, b))
        end
        for avpairs in powresults
            for (a, v) in avpairs
                push!(ad[a], v)
            end
        end
        for (a, vs) in ad
            sem = std(vs)/sqrt(n)
            mn = mean(vs)
            if a in (:wait, :listen)
                @show "$a: $mn ± $(std(vs))"
            end
            push!(results, (mean=mn,
                            std=std(vs),
                            sem=sem,
                            max=mn+3*sem,
                            min=mn-3*sem,
                            width=width,
                            solver="POWSS",
                            depth=depth,
                            action=a
                           ))
        end

        ad = Dict{Symbol, Vector{Float64}}(a=>Float64[] for a in actions(m))
        pomcpow_results = @showprogress pmap(1:n) do i
            solver = POMCPOWSolver(criterion=MaxUCB(200.0),
                                   tree_queries=100_000,
                                   # max_depth=4,
                                   # max_time=0.1,
                                   enable_action_pw=false,
                                   k_observation=width,
                                   alpha_observation=0.0
                                  )
            pomcpow_planner = solve(solver, m)
            oa = ordered_actions(m)
            avs = actionvalues(pomcpow_planner, b)
            return [oa[i]=>avs[i] for i in 1:length(oa)]
        end
        for avpairs in pomcpow_results
            for (a, v) in avpairs
                push!(ad[a], v)
            end
        end
        for (a, vs) in ad
            sem = std(vs)/sqrt(n)
            mn = mean(vs)
            if a in (:wait, :listen)
                @show "$a: $mn ± $(std(vs))"
            end
            push!(results, (mean=mn,
                            std=std(vs),
                            sem=sem,
                            max=mn+3*sem,
                            min=mn-3*sem,
                            width=width,
                            solver="POMCPOW",
                            depth=depth,
                            action=a
                           ))
        end

        fname = joinpath(dirname(@__FILE__()), "data", "width_$(width)_"*datestring*".csv")
        @show CSV.write(fname, results)
    end
    recalc = false
end

# Gadfly plots
#=
# plotted = results[results[:solver].=="POWSS" & (results[:action] (:listen1, :listen2)), :]
plotted = results |> @filter(_.solver=="POWSS") |> @filter(_.action in (:listen1, :listen2)) |> DataFrame
@show plotted

optimal = DataFrame(action=ordered_actions(m), value=collect(actionvalues(op, b)))
optimal = optimal |> @filter(_.action in (:listen1, :listen2)) |> DataFrame
@show optimal

p1 = plot(
          layer(optimal, yintercept=:value, color=:action, Geom.hline),
          layer(plotted, x=:width, y=:mean, ymin=:min, ymax=:max, color=:action, Geom.line, Geom.ribbon),
         )
=#
