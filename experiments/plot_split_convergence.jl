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
r_findtiger = -10.0

# fname = joinpath(dirname(@__FILE__()), "data", "width_41_Mon_7_Oct_2019_16_55.csv") # no pomcpow
fname = joinpath(dirname(@__FILE__()), "data", "width_41_Wed_29_Apr_2020_15_46.csv") # -10 penalty
# fname = joinpath(dirname(@__FILE__()), "data", "width_41_Wed_29_Apr_2020_15_46.csv") # -100 penalty

# pgfplots()
pyplot()

color_d = Dict("wait" => "black",
               "listen" => "black")
color_f(a) = get(color_d, string(a), "black")

label_d = Dict("wait" => "Wait",
               "listen" => "Listen",
              )
label_f(a) = label_d[string(a)]

results = CSV.read(fname)
# @show results

plotted = results |> @filter(_.solver=="POWSS") |> @filter(_.action in ("wait", "listen")) |> DataFrame

plots = Dict()
ylims = (2.0,9.0)
plots["wait"] = plot(xlabel="Width (C)", legend=(0.4,0.45), title="Wait", ylims=ylims)
plots["listen"] = plot(xlabel="Width (C)", ylabel="Estimated Q Value", title="Listen", legend=false, ylims=ylims)

for a in ("wait", "listen")
    df = plotted |> @filter(_.action == a) |> DataFrame
    plot!(plots[a], df[!,:width], df[!,:mean],
          ribbon=df[!,:std],
          color=color_f(a),
          # label="POWSS "*label_f(a)
          label="POWSS"
        )
end

dm = TimedDOTigerPOMDP(horizon=depth, p_correct=p_correct, r_findtiger=r_findtiger)
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
@assert optpairs[:wait] == maximum(dot(bvec, α.alpha) for α in Sa)
@show optpairs

for a in [:wait, :listen]
    plot!(plots[string(a)], [1, maximum(results[!, :width])], fill(optpairs[a], 2),
          color="red",
          linestyle=:dash,
          label="Optimal"
         )
end
# savefig(p, "fig/powss_convergence_plot.pdf")

# POSS/QMDP
plotted = results |> @filter(_.solver=="POSS") |>  DataFrame

for a in ("wait", "listen")
    df = plotted |> @filter(_.action == a) |> DataFrame
    plot!(plots[a], df[!,:width], df[!,:mean],
          color=color_f(a),
          label="POSS",
          linestyle=:dot
        )
end

# POMCPOW
plotted = results |> @filter(_.solver=="POMCPOW") |>  DataFrame

for a in ("wait", "listen")
    df = plotted |> @filter(_.action == a) |> DataFrame
    plot!(plots[a], df[!,:width], df[!,:mean],
          # ribbon=df[!,:std],
          color=color_f(a),
          label="POMCPOW",
          linestyle=:dashdot
        )
end


p = plot(plots["listen"], plots["wait"], size=(400,300))
@show savefig(p, "fig/convergence_plot_split.pdf")
run(`xdg-open fig/convergence_plot_split.pdf`)
