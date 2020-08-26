module OrdinalSVM

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
    b = Variable(l)
    # get indexes of the labels
    inds = indexin(y, L)
    # compute objective
    obj = sum(max(0, 1 - X * w  - b[inds])) + C * sum(w .* w)
    problem = minimize(obj, b[1]==0.0, (b[i] <= b[i+1] for i in 1:l-1)...)
    solve!(problem, () -> SCS.Optimizer(verbose=false))

    return OrdinalSVM(w.value, b.value, L)
end


end