
using BioLizard.ZeroNorm
using Zygote

@testset "ZeroNorm" begin

    y = randn(100)
    X = randn(100, 5)

    w = randn(5)

    l(w) = sum((y - X * w).^2)

    s = ones(Bool, 5)

    @testset "momentum" begin
        dw = zeros(5)
        dl!(w, dw, β) = dw .= (1.0 - β) .* l'(w) .+ β .* dw
        momentum!(w, s, dl!, dw)
        @test sum(dw.^2) < 1e-2 
    end

    @testset "find zero-norm" begin
        # SA
        w0 = zeros(5)
        findzeronorm(w0, l, 0.1, s, SimulatedAnnealing(kT=1), verbose=true)

        w0 = zeros(5)
        findzeronorm(w0, l, 0.1, s, HillClimbing(), verbose=true)

    end
end