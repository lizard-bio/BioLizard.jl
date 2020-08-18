#=
Created on Tuesday 11 August 2020
Last update: Tuesday 18 August 2020

@author: Michiel Stock
michielfmstock@gmail.com

Linear models with zero-"norm" regularization.
Useful for feature selection.
=#

module ZeroNorm

export momentum!, flip, findzeronorm, SimulatedAnnealing, L0Solver, HillClimbing

using Zygote, LinearAlgebra

"""
momentum!(w, s, dl!, dw;
		α=0.01, β=0.5, niter=100, ϵ=1e-4)

Gradient descent with momentum to fit sparse linear models. Fits a parameter
vector `w` with sparsity structure `s`. Requires a function `dl!` (in-place
gradient of the loss function) and a vector `dw` to allocate the gradients.
"""
function momentum!(w::AbstractVecOrMat,
					s::AbstractVecOrMat{Bool},
					dl!,
					dw::AbstractVecOrMat;
					α=0.01, β=0.5, niter=100, ϵ=1e-4)
	@assert size(w) == size(s) == size(dw)
	dw .= 0.0
	for i in 1:niter
		# update gradient
		dl!(w, dw, β)
		dw .*= s
		norm(dw) < ϵ && break
		w .-= α .* dw
	end
	return w
end

abstract type L0Solver end

# SIMULATED ANNEALING

struct SimulatedAnnealing <: L0Solver
	p₀
	p₁
	r
	kT
	Tmin
	Tmax
end

"""
	SimulatedAnnealing(;p₀=0.05, p₁=0.05, r=0.8, kT=1000, Tmin=1e-3, Tmax=1e3)

Solver for L0-norm minimization. Uses parameters:

	- `p₀` : probability of true -> false in `s`
	- `p₁`: probability of false -> true in `s`
	- `r` : cooling rate
	- `kT` : number of repetitions per temperature
	- `Tmax` : starting temperature
	- `Tmin` : stopping temperature
"""
SimulatedAnnealing(;p₀=0.05, p₁=0.05, r=0.8, kT=1000,
			Tmin=1e-3, Tmax=1e3) = SimulatedAnnealing(p₀, p₁, r, kT, Tmin, Tmax)

struct HillClimbing <: L0Solver end

"""
Turns a `true` to a `false` with probability `p₀` and a `false` to a `true`
with probability `p₁`.
"""
flip(bit::Bool, p₀, p₁) = bit ? rand() > p₀ : rand() < p₁


function findzeronormSA(w, loss, λ, s, p₀, p₁, r, kT,
			Tmin, Tmax; verbose::Bool=false,
			momentumpars...)
	n = length(w)
	dw = similar(w)
	# compute gradient for momentum
	dl!(w, dw, β) = dw .= (1.0 - β) .* loss'(w) .+ β .* dw
	snew = copy(s)
	sbest = copy(s)
	j(w, s) = loss(w .* s) + λ * sum(s)
	T = Tmax
	best_obj = Inf
	current_obj = Inf
	# create structure to store the objective through the iterations
	obj_vals = Vector{Float64}(undef, ceil(Int, (log(Tmin) - log(Tmax)) / log(r)))
	iter = 0
	while T > Tmin
		iter += 1
		for rep in 1:kT
			# perturbate s
			snew .= flip.(s, p₀, p₁)
			# update w
			momentum!(w, snew, dl!, dw; momentumpars...)
			obj = j(w, snew)
			if obj < current_obj || rand() < exp((current_obj - obj)/T)
				current_obj = obj
				s .= snew
			end
			if obj < best_obj
				best_obj = obj
				sbest .= s
			end
		end
		verbose && println("T=$T : obj=$best_obj, nfeatures=$(sum(sbest))")
		obj_vals[iter] = best_obj
		T *= r
		end
	return sbest
end

# HILL CLIMBING

function findzeronormHC(w, loss, λ, s; verbose::Bool=false,
		momentumpars...)
	n = length(w)
	dw = similar(w)
	# compute gradient for momentum
	dl!(w, dw, β) = dw .= (1.0 - β) .* loss'(w) .+ β .* dw
	j(w, s) = loss(w .* s) + λ * sum(s)
	best_obj = Inf
	improved = true
	iter = 0
	while improved
		improved = false
		iter += 1
		best_flip = 0
		for i in 1:n
			# perturbate s
			s[i] = !s[i]
			# update w
			momentum!(w, s, dl!, dw; momentumpars...)
			obj = j(w, s)
			if obj < best_obj
				best_obj = obj
				improved = true
				best_flip = i
			end
			# change s back
			s[i] = !s[i]
		end
	# update s
	improved && (s[best_flip] = !s[best_flip])
	verbose && improved && println("iteration $iter, $(s[best_flip] ? "added $best_flip" : "removed $best_flip"), obj=$best_obj, nfeatures=$(sum(s))")
	end
return s, w
end



function findzeronorm(w0, loss, λ, s, sa::SimulatedAnnealing; kwargs...)
	return findzeronormSA(w0, loss, λ, s, sa.p₀, sa.p₁, sa.r, sa.kT, sa.Tmin,
				sa.Tmax; kwargs...), w0
end

findzeronorm(w0, loss, λ, s, ::HillClimbing; kwargs...) = findzeronormHC(w0, loss, λ, s; kwargs...)

findzeronorm(w0, loss, λ, sol::L0Solver; kwargs...) = findzeronorm(w0, loss, λ, ones(Bool, length(w0)), sol; kwargs...)

"""
	findzeronorm(w0, loss, λ[, s0], sol; verbose=true, momentumpars...)

Finds the zero norm for a model with loss function regularized with a zero-"norm":

	min_w loss(w) + ||w||_0

Uses Simulated Annealing or Hill Climbing and returns `s` a binary vector containing the sparsity
and a vector with the objective value throughout the iterations.

For SA, use `sol=SimulatedAnnealing()` (with optional parameters), for HC, use `sol=HillClimbing()`.

Parameters:
	- `w0` : starting value for the parameter vector
	- `loss` : a differentiable loss function (we use autodiff)
	- `λ` : regularization parameter
	- `s0` : initial guess for `s`, optional, uses all features initially by default
	- `verbose` : flag to print the convergence during fitting
	- `momentumpars` : parameters to determine behaviour for momentum
"""
findzeronorm(w0, loss, λ, s, sol::L0Solver; kwargs...) = nothing



end  # module ZeroNorm
