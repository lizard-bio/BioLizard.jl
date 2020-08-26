@testset "ordinal SVM" begin

using BioLizard: OrdinalRegr
    
L = [:A, :B, :C]

X = randn(100, 5)

y = [xi + 0.1randn() < -2 ? :A : (xi + 0.1randn() < 0 ? :B : :C) for xi in X[:,4]]

model = fitordinalsvm(y, X, L)

@test predict(model, X) isa Vector{Symbol}
@test score(model, X) isa Vector{Float64}

end