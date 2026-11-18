# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationLBFGSBExt

using BAT
import OptimizationLBFGSB

BAT.pkgext(::Val{:OptimizationLBFGSB}) = BAT.PackageExtension{:OptimizationLBFGSB}()
BAT.ext_default(::BAT.PackageExtension{:OptimizationLBFGSB}, ::Val{:LBFGSB_ALG}) = OptimizationLBFGSB.LBFGSB()

# The Fortran backend requires Float64 input. The sampler converts fitted
# centers back to the context precision before forming proposal geometry.
function BAT._mw_mode(center::AbstractVector{Float32}, logtarget,
    mode::BAT.OptimizationAlg{<:OptimizationLBFGSB.LBFGSB}, remaining, context)
    return BAT._mw_mode(Float64.(center), logtarget, mode, remaining, context)
end

end
