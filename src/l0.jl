#=
Created on Tuesday 11 August 2020
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Linear models with zero-"norm" regularization.
Useful for feature selection.
=#

module ZeroNorm

export momentum!, flip, findzeronorm!, LinearAlgebra

using Zygote

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

"""
Turns a `true` to a `false` with probability `p₀` and a `false` to a `true`
with probability `p₁`.
"""
flip(bit::Bool, p₀, p₁) = bit ? rand() > p₀ : rand() < p₁

"""
	findzeronorm!(w0, loss, λ[, s0]; p₀=0.05, p₁=0.05, r=0.8, kT=1000,
							Tmin=1e-3, Tmax=1e3, verbose=true, momentumpars...)

Finds the zero norm for a model with loss function regularized with a zero-"norm":

	min_w loss(w) + ||w||_0

Uses Simulated Annealing and returns `s` a binary vector containing the sparsity
and a vector with the objective value throughout the iterations.

Parameters:
	- `w0` : starting value for the parameter vector
	- `loss` : a differentiable loss function (we use autodiff)
	- `λ` : regularization parameter
	- `s0` : initial guess for `s`, optional, uses all features initially by default
	- `p₀` : probability of true -> false in `s`
	- `p₁`: probability of false -> true in `s`
	- `r` : cooling rate
	- `kT` : number of repetitions per temperature
	- `Tmax` : starting temperature
	- `Tmin` : stopping temperature
	- `verbose` : flag to print the convergence during fitting
	- `momentumpars` : parameters to determine behaviour for momentum
"""
function findzeronorm!(w0, loss, λ, s0; p₀=0.01, p₁=0.01, r=0.8, kT::Int=1000,
							Tmin=1e-3, Tmax=1e3, verbose::Bool=false,
							momentumpars...)
	w = w0
	n = length(w)
	dw = similar(w)
	# compute gradient for momentum
	dl!(w, dw, β) = dw .= (1.0 - β) .* loss'(w) .+ β .* dw
	s = s0  # depart from using all variables
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
	return sbest, obj_vals
end

findzeronorm!(w0, loss, λ; kwargs...) = findzeronorm!(w0, loss, λ, ones(Bool, length(w0)); kwargs...)

end  # module ZeroNorm
