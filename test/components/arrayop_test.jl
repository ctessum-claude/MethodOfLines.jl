using Test
using Symbolics
using SymbolicUtils: @arrayop

@testset "arrayop infrastructure" begin
    @testset "Symbolics @arrayop scalarize roundtrip" begin
        @variables x[1:4] y[1:4]
        i = only(@variables i::Int)

        # Element-wise addition via @arrayop
        arr_add = @arrayop (i,) x[i] + y[i]
        sc = Symbolics.scalarize(arr_add)
        @test length(sc) == 4
        @test isequal(sc[1], x[1] + y[1])
        @test isequal(sc[4], x[4] + y[4])

        # Element-wise multiply via @arrayop
        arr_mul = @arrayop (i,) x[i] * y[i]
        sc_mul = Symbolics.scalarize(arr_mul)
        @test length(sc_mul) == 4
        @test isequal(sc_mul[2], x[2] * y[2])
    end

    @testset "arrayop_equations helper" begin
        using MethodOfLines: arrayop_equations

        @variables a b c d e f
        lhs_vec = Num[a + b, c, d^2]
        rhs_vec = Num[e, f, a * c]

        eqs = arrayop_equations(lhs_vec, rhs_vec)
        @test length(eqs) == 3
        @test isequal(eqs[1].lhs, a + b)
        @test isequal(eqs[1].rhs, e)
        @test isequal(eqs[2].lhs, c)
        @test isequal(eqs[2].rhs, f)
        @test isequal(eqs[3].lhs, d^2)
        @test isequal(eqs[3].rhs, a * c)

        # Results match direct broadcast
        direct = lhs_vec .~ rhs_vec
        for i in 1:3
            @test isequal(eqs[i].lhs, direct[i].lhs)
            @test isequal(eqs[i].rhs, direct[i].rhs)
        end
    end

    @testset "arrayop_equations empty input" begin
        using MethodOfLines: arrayop_equations
        eqs = arrayop_equations(Num[], Num[])
        @test isempty(eqs)
    end
end
