using Distributions, Plots
using BioLizard.OrdinalRegr: fitordinalsvm, OrdinalSVM, predict, score

Σ = [3.0 1; 1 2]

dist1 = MultivariateNormal([0, 0], Σ)
dist2 = MultivariateNormal([1, 3], Σ)
dist3 = MultivariateNormal([3, 9], Σ)

X = [rand(dist1, 20) rand(dist2, 20) rand(dist3, 20)]'
y = vcat([3 for i in 1:20], [5 for i in 1:20], [8 for i in 1:20])

pcl = scatter(X[1:20,1], X[1:20,2], label="class 1")
scatter!(X[21:40,1], X[21:40,2], label="class 2")
scatter!(X[41:60,1], X[41:60,2], label="class 3")

model = fitordinalsvm(y, X)

phist = histogram(score(model, X[1:20,:]), label="class 1", alpha=0.8)
histogram!(score(model, X[21:40,:]), label="class 2", alpha=0.8)
histogram!(score(model, X[41:60,:]), label="class 3", alpha=0.8)
vline!(model.b, label="b")

plot(pcl, phist, layout=(2,1))
