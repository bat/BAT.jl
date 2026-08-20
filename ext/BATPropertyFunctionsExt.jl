# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATPropertyFunctionsExt

using PropertyFunctions: PropertyFunction, PPaths
import BAT

# PropertyFunctions' function-chain fusion absorbs functions that follow a
# property function, including likelihoods. Invert the wrappers: the
# transformed likelihood is recognized and precomposed, with the transform
# re-wrapped as a property function, so its property-access information
# stays available:
function BAT._split_density_transform(pf::PropertyFunction{Paths}) where {Paths<:PPaths}
    r = BAT._split_density_transform(pf.sel_prop_func)
    isnothing(r) && return nothing
    ℒ, g = r
    (ℒ, g === identity ? identity : PropertyFunction{Paths}(g))
end

end # module BATPropertyFunctionsExt
