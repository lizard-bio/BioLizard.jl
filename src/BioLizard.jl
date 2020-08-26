module BioLizard

include("l0.jl")
export ZeroNorm

include("ordinalregr.jl")
export OrdinalRegr, fitordinalsvm, score, predict

end # module
