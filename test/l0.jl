
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
        dl!(w, dw, β) = dw .+= β * l'(w)
        momentum!(w, s, dl!, dw)
        @test sum(dw.^2) < 1e-2 
    end
end