using ContinuousObservationToyPOMDPs
using SparseSampling
using QMDP
using POMDPs
using BeliefUpdaters
using POMDPPolicies

m = COTigerPOMDP()
qp = solve(QMDPSolver(), m)
b = initialize_belief(updater(qp), initialstate_distribution(m))
@show qp.action_map
@show actionvalues(qp, b)

for width in 5:5:100
    @show opt = SSOptions(3, width)
    p = solve(SparseSamplingSolver(opt), m)
    @show collect(valuepairs(p, b))
end

for width in 5:5:50
    @show opt = POWSSOptions(5, width)
    p = solve(SparseSamplingSolver(opt), m)
    @show collect(valuepairs(p, b))
end
