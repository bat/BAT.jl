# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATCubaExt

@static if isdefined(Base, :get_extension)
    using Cuba
else
    using ..Cuba
end

using BAT
BAT.pkgext(::Val{:Cuba}) = BAT.PackageExtension{:Cuba}()

using BAT: AbstractMeasureOrDensity, CubaIntegration
using BAT: var_bounds, spatialvolume, log_volume, bat_integrate_impl
using BAT: fromuhc

using Base.Threads: @threads

using ArraysOfArrays
using DensityInterface, ValueShapes
import Measurements


BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:ATOL}) = Cuba.ATOL
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:BORDER}) = Cuba.BORDER
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:FLATNESS}) = Cuba.FLATNESS
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:KEY}) = Cuba.KEY
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:KEY1}) = Cuba.KEY1
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:KEY2}) = Cuba.KEY2
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:KEY3}) = Cuba.KEY3
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:LDXGIVEN}) = Cuba.LDXGIVEN
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:MAXCHISQ}) = Cuba.MAXCHISQ
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:MAXEVALS}) = Cuba.MAXEVALS
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:MAXPASS}) = Cuba.MAXPASS
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:MINDEVIATION}) = Cuba.MINDEVIATION
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:MINEVALS}) = Cuba.MINEVALS
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NBATCH}) = Cuba.NBATCH
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NEXTRA}) = Cuba.NEXTRA
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NGIVEN}) = Cuba.NGIVEN
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NINCREASE}) = Cuba.NINCREASE
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NMIN}) = Cuba.NMIN
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NNEW}) = Cuba.NNEW
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:NSTART}) = Cuba.NSTART
BAT.ext_default(::BAT.PackageExtension{:Cuba}, ::Val{:RTOL}) = Cuba.RTOL


struct CubaIntegrand{D<:AbstractMeasureOrDensity,T<:Real} <: Function
    density::D
    log_density_shift::T
    log_support_volume::T
end


function CubaIntegrand(density::AbstractMeasureOrDensity, log_density_shift::Real)
    vol = spatialvolume(var_bounds(density))
    isinf(vol) && throw(ArgumentError("CUBA integration doesn't support densities with infinite support"))
    log_support_volume = log_volume(vol)
    @assert _cuba_valid_value(log_support_volume)

    CubaIntegrand(density, float(log_density_shift), log_support_volume)
end


_cuba_valid_value(x) = !isnan(x) && x < typeof(x)(+Inf)


function (integrand::CubaIntegrand)(x::AbstractVector{<:Real}, f::AbstractVector{<:Real})
    idxs = axes(f, 1)
    @assert length(idxs) == 1

    vol = spatialvolume(var_bounds(integrand.density))
    x_trafo = fromuhc(x, vol)
    logd = logdensityof(integrand.density, x_trafo)
    @assert _cuba_valid_value(logd)

    f[first(idxs)] = exp(logd + integrand.log_density_shift)
    @assert all(_cuba_valid_value,f)

    f
end


function (integrand::CubaIntegrand)(X::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real})
    idxs1 = axes(f, 1)
    @assert length(idxs1) == 1
    idxs2 = axes(f, 2)
    @assert idxs2 == axes(X, 2)

    vol = spatialvolume(var_bounds(integrand.density))
    x_trafo = fromuhc(nestedview(X), vol)
    @threads for i in idxs2
        logd = logdensityof(integrand.density, x_trafo[i])
        @assert _cuba_valid_value(logd)
        y = exp(logd + integrand.log_density_shift)
        @assert _cuba_valid_value(y)
        f[first(idxs1), i] = y
    end

    f
end


function BAT.bat_integrate_impl(integrand::CubaIntegrand, algorithm::VEGASIntegration)
    r = Cuba.vegas(
        integrand, totalndof(integrand.density), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        nstart = algorithm.nstart, nincrease = algorithm.nincrease, nbatch = algorithm.nbatch
    )
end


function BAT.bat_integrate_impl(integrand::CubaIntegrand, algorithm::SuaveIntegration)
    Cuba.suave(
        integrand, totalndof(integrand.density), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        nnew = algorithm.nnew, nmin = algorithm.nmin, flatness = algorithm.flatness
    )
end


function BAT.bat_integrate_impl(integrand::CubaIntegrand, algorithm::DivonneIntegration)
    Cuba.divonne(
        integrand, totalndof(integrand.density), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        key1 = algorithm.key1, key2 = algorithm.key2, key3 = algorithm.key3,
        maxpass = algorithm.maxpass, border = algorithm.border, maxchisq = algorithm.maxchisq,
        mindeviation = algorithm.mindeviation, ngiven = algorithm.ngiven, ldxgiven = algorithm.ldxgiven,
        nextra = algorithm.nextra
    )
end


function BAT.bat_integrate_impl(integrand::CubaIntegrand, algorithm::CuhreIntegration)
    Cuba.cuhre(
        integrand, totalndof(integrand.density), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        key = algorithm.key
    )
end


function BAT.bat_integrate_impl(target::AnyMeasureOrDensity, algorithm::CubaIntegration)
    density_notrafo = convert(AbstractMeasureOrDensity, target)
    shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
    density = unshaped(shaped_density)
    integrand = CubaIntegrand(density, algorithm.log_density_shift)

    r_cuba = bat_integrate_impl(integrand, algorithm)

    if r_cuba.fail != 0
        buf = IOBuffer()
        Cuba.print_fail(buf, r_cuba)
        msg = String(take!(buf))
        if algorithm.strict
            throw(ErrorException(msg))
        else
            @warn(msg)
        end
    end

    log_renorm_corr = -integrand.log_density_shift + integrand.log_support_volume
    T = promote_type(BigFloat, typeof(log_renorm_corr))
    renorm_corr = exp(convert(T, log_renorm_corr))

    ival = first(r_cuba.integral) * renorm_corr
    ierr = first(r_cuba.error) * renorm_corr

    (result = Measurements.measurement(ival, ierr), cuba_result = r_cuba, renorm_corr = renorm_corr)
end


function BAT.bat_integrate_impl(target::SampledMeasure, algorithm::CubaIntegration)
    bat_integrate_impl(target.density, algorithm)
end


end # module BATCubaExt
