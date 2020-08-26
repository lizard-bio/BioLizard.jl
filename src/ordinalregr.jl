module OrdinalRegr

using Convex, SCS

struct OrdinalSVM
    w
    b
    L
end

function fitordinalsvm(y, X, L; C=1e-2)
    n, p = size(X)
    l = length(L)
    w = Variable(p)
    b = Variable(l - 1)
    # get indexes of the labels
    inds = convert(Vector{Int}, indexin(y, L))
    # compute objective
    obj = C * sum(w .* w)
    f = X * w
    for (j, lj) in enumerate(L[1:end-1])
        for (i, yi) in enumerate(y)
            obj += yi ≤ lj ? max(0.0, 1.0 + (f[i] - b[j])) : max(0.0, 1.0 - (f[i] - b[j]))
        end
    end
    problem = minimize(obj, (b[i] <= b[i+1] for i in 1:l-2)...)
    solve!(problem, () -> SCS.Optimizer(verbose=false))
    return OrdinalSVM(w.value[:], b.value[:], L)
end

fitordinalsvm(y, X; C=1e-2) = fitordinalsvm(y, X, sort!(unique(y)); C=1e-2)

"""
    score(model::OrdinalSVM, X::AbstractMatrix)

Compute the scoring using an `OrdinalSVM` model.
"""
function score(model::OrdinalSVM, X::AbstractMatrix)
    X * model.w
end

"""
predict(model::OrdinalSVM, X::AbstractMatrix)

Predict the label using an `OrdinalSVM` model.
"""
function predict(model::OrdinalSVM, X::AbstractMatrix)
    l = [model.L[1] for i in 1:size(X, 1)]
    for (i, s) in enumerate(score(model, X))
        for (Lj, bj) in zip(model.L[2:end], model.b)
            s - bj ≥ 0 && (l[i] = Lj)
        end
    end
    return l

end


end