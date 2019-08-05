using ContinuousObservationToyPOMDPs
using SparseSampling
using QMDP
using POMDPs
using BeliefUpdaters
using POMDPPolicies
using ProfileView
using Profile

m = COTigerPOMDP()
qp = solve(QMDPSolver(), m)
b = initialize_belief(updater(qp), initialstate_distribution(m))
@show qp.action_map
@show actionvalues(qp, b)

function f(width)
    opt = SSOptions(3, width)
    p = solve(SparseSamplingSolver(opt), m)
    collect(valuepairs(p, b))
end

# @code_warntype observation(m, :left, :listen_1)

@time f(20)

Profile.clear()
@profile f(20)
ProfileView.view()
