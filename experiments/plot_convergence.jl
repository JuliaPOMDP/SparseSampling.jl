using Plots
using StatsPlots
using CSV
using Query
using DataFrames
using IncrementalPruning
using SARSOP
using POMDPSolve
using ContinuousObservationToyPOMDPs
using POMDPs
using POMDPPolicies
using POMDPModelTools
using LinearAlgebra
using QMDP

n = 200
maxwidth = 41
widthstep = 4
depth = 3
p_correct = 0.85

fname = joinpath(dirname(@__FILE__()), "data", "width_41_Mon_7_Oct_2019_16_55.csv") # no pomcpow
# fname = joinpath(dirname(@__FILE__()), "data", "width_41_Wed_29_Apr_2020_15_46.csv") # -10 penalty
# fname = joinpath(dirname(@__FILE__()), "data", "width_41_Wed_29_Apr_2020_15_46.csv") # -100 penalty

# pgfplots()
pyplot()

color_d = Dict("wait" => "blue",
               "listen" => "red")
color_f(a) = get(color_d, string(a), "black")

label_d = Dict("wait" => "Wait",
               "listen" => "Listen",
              )
label_f(a) = label_d[string(a)]

results = CSV.read(fname)
# @show results

plotted = results |> @filter(_.solver=="POWSS") |> @filter(_.action in ("wait", "listen")) |> DataFrame

p = plot(xlabel="Width (C)", ylabel="Estimated Q Value", size=(400,300), legend=(0.6,0.45))
for a in ("wait", "listen")
    df = plotted |> @filter(_.action == a) |> DataFrame
    plot!(p, df[!,:width], df[!,:mean],
          ribbon=df[!,:std],
          color=color_f(a),
          label="POWSS "*label_f(a)
        )
end

dm = TimedDOTigerPOMDP(horizon=depth, p_correct=p_correct)
op  = solve(PruneSolver(), dm)
b = initialize_belief(updater(op), initialstate_distribution(dm))
optpairs = Dict(first(item)=>last(item) for item in zip(ordered_actions(dm), actionvalues(op, b)))

a = :wait
Sz = Vector{Set{Vector{Float64}}}(undef, length(observations(dm)))
for (zind, z) in enumerate(ordered_observations(dm))
    V = Set(IncrementalPruning.dpval(α,a,z,dm) for α in op.alphas)
    Sz[zind] = IncrementalPruning.filtervec(V, PruneSolver().optimizer_factory)
end
Sa = Set([IncrementalPruning.AlphaVec(α,a) for α in IncrementalPruning.incprune(Sz, PruneSolver().optimizer_factory)])

bvec = zeros(length(states(dm)))
bvec[stateindex(dm, (:left, 0))] = 0.5 
bvec[stateindex(dm, (:right, 0))] = 0.5
@show optpairs[:wait] = maximum(dot(bvec, α.alpha) for α in Sa)

for a in [:wait, :listen]
    plot!(p, [1, maximum(results[!, :width])], fill(optpairs[a], 2),
          color=color_f(a),
          linestyle=:dash,
          label="Optimal "*label_f(a)
         )
end
savefig(p, "fig/powss_convergence_plot.pdf")

# POSS/QMDP
plotted = results |> @filter(_.solver=="POSS") |>  DataFrame

for a in ("wait", "listen")
    df = plotted |> @filter(_.action == a) |> DataFrame
    plot!(p, df[!,:width], df[!,:mean],
          color=color_f(a),
          label="POSS "*label_f(a),
          linestyle=:dot
        )
end
@show savefig(p, "fig/convergence_plot.pdf")
