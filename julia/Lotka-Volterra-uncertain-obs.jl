using DifferentialEquations, DiffEqParamEstim, Optimization, OptimizationBBO
f1 = function (du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3.0 * u[2] + u[1] * u[2]
end
p = [1.5, 1.0]
u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
prob1 = ODEProblem(f1, u0, tspan, p)
sol = solve(prob1, Tsit5())

using RecursiveArrayTools # for VectorOfArray
t = collect(range(0, stop = 10, length = 200))
function generate_data(sol, t)
    randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
    data = convert(Array, randomized)
end
aggregate_data = convert(Array, VectorOfArray([generate_data(sol, t) for i in 1:100]))

using Distributions
distributions = [fit_mle(Normal, aggregate_data[i, j, :]) for i in 1:2, j in 1:200]