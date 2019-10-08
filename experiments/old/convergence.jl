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

datestring = Dates.format(now(), "e_d_u_Y_HH_MM")

m = COTigerPOMDP()
qp = solve(QMDPSolver(), m)
b = initialize_belief(updater(qp), initialstate_distribution(m))
@show qp.action_map
@show actionvalues(qp, b)

dm = DOTigerPOMDP()
op  = solve(SARSOPSolver(), dm)
# op  = solve(PruneSolver(), dm)
b = initialize_belief(updater(op), initialstate_distribution(dm))
@show op.action_map
@show actionvalues(op, b)

n = 200
maxwidth = 6
if !@isdefined recalc
    recalc = true
end

if recalc
    results = DataFrame(mean=Float64[], max=Float64[], min=Float64[], width=Int[], solver=String[], action=Symbol[])

    for width in 1:maxwidth
        ad = Dict{Symbol, Vector{Float64}}(a=>Float64[] for a in actions(m))
        @show width
        ssresults = @showprogress pmap(1:n) do i
            opt = SSOptions(5, width)
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
                            max=mn+3*sem,
                            min=mn-3*sem,
                            width=width,
                            solver="POSS",
                            action=a
                           ))
        end
        
        ad = Dict{Symbol, Vector{Float64}}(a=>Float64[] for a in actions(m))
        powresults = @showprogress pmap(1:n) do i
            powopt = POWSSOptions(5, width)
            powp = solve(SparseSamplingSolver(powopt), m)
            return valuepairs(powp, b)
        end
        for avpairs in powresults
            for (a, v) in avpairs
                push!(ad[a], v)
            end
        end
        for (a, vs) in ad
            sem = std(vs)/sqrt(n)
            mn = mean(vs)
            push!(results, (mean=mn,
                            max=mn+3*sem,
                            min=mn-3*sem,
                            width=width,
                            solver="POWSS",
                            action=a
                           ))
        end
    end
    recalc = false
end

fname = joinpath(dirname(@__FILE__()), "data", "results_"*datestring*".csv")
CSV.write(fname, results)

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
