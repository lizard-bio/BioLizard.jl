#=
Created on Tuesday 11 August 2020
Last update: Tuesday 18 August 2020

@author: Michiel Stock
michielfmstock@gmail.com

Simple demonstration of linear regression with zero-norm minimization.
=#

using BioLizard.ZeroNorm
using Plots, StatsBase

# generate some data

n, p, σ = 150, 50, 1

X = randn(n, p)
strue = rand(p) .< 0.2  # use 20% of the features
wtrue = randn(p)
y = X * (wtrue .* strue) .+ σ * randn(n)

# simple squared loss, easy as pie
l(w) = mean((X * w .- y).^2)

# initial guess for w
w = zeros(p)
λ = 0.05

# use the method to find the norm
s, w = findzeronorm(w, l, λ, SimulatedAnnealing(), verbose=true)

# can also use an initial s
#s0 = ones(Bool, p)
#s, obj_vals = findzeronorm!(w, l, λ, s0, verbose=true)

# get confusion table
tp = sum(strue[s])  # true pos
fp = sum(.!strue[s])
tn = sum(.!strue[.!s])
fn = sum(strue[.!s])

println("""
TP = $tp
FP = $fp
TN = $tn
FN = $fn
""")

scatter(w[strue], wtrue[strue], color=1s[strue])
xlabel!("w fitted")
ylabel!("w true")
