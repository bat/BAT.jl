# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type PackageExtension{pkgname}

*Experimental feature, not part of stable public API.*

Represents a package extension that requires the package `pkgname` to be
loaded.

Do not construct instances of `PackageExtension` directly, use
`pkgext(:pkgname)` instead which will check that the required extension is
active.
"""
struct PackageExtension{pkgname} end


"""
    BAT.pkgext(:SomePackage)::PackageExtension
    BAT.pkgext(Val(:SomePackage))::PackageExtension

*Experimental feature, not part of stable public API.*

Returns the [`PackageExtension`](@ref) instance that depends on the package
`SomePackage`. Will throw an error if the extension is not active (because
`SomePackage`` hasn't been loaded).
"""
function pkgext end

pkgext(pkgname::Symbol) = pkgext(Val{pkgname}())

function pkgext(@nospecialize(v::Val{pkgname})) where pkgname
    throw(ErrorException("Requested functionality requires the Julia package $pkgname to be loaded."))
end


"""
    BAT.ext_default(::PackageExtension{SomePackage}, ::Val{:SomeConstant})
    
*Experimental feature, not part of stable public API.*

Returns the default value selected by `:SomeConstant` within the context of
the package extension that depends on `SomePackage`.
"""
function ext_default end
