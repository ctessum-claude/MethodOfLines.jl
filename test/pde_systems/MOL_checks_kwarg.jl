# Test that checks=false kwarg on discretize is properly propagated

using ModelingToolkit, MethodOfLines, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

@testset "discretize with checks=false" begin
    # Simple 1D diffusion problem
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, Float64(π)) ~ -exp(-t),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc = MOLFiniteDifference([x => dx], t)

    # Test 1: checks=false via discretize kwarg
    prob = discretize(pdesys, disc; checks = false)
    @test prob isa ODEProblem

    # Test 2: default (checks=true) still works
    prob2 = discretize(pdesys, disc)
    @test prob2 isa ODEProblem

    # Test 3: solve with checks=false to verify the problem is well-formed
    sol = solve(prob, Tsit5(), saveat = 0.1)
    @test sol.retcode == SciMLBase.ReturnCode.Success
end
