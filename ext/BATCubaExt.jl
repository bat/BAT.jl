# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATCubaExt

using Cuba

using BAT
BAT.pkgext(::Val{:Cuba}) = BAT.PackageExtension{:Cuba}()

using BAT: MeasureLike, BATMeasure
using BAT: CubaIntegration
using BAT: measure_support, bat_integrate_impl
using BAT: transform_and_unshape, auto_renormalize

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


struct CubaIntegrand{F<:Function} <: Function
    f_logdensity::F
    dof::Int
end

_cuba_valid_value(x) = !isnan(x) && x < typeof(x)(+Inf)

function (integrand::CubaIntegrand)(x::AbstractVector{<:Real}, f::AbstractVector{<:Real})
    idxs = axes(f, 1)
    @assert length(idxs) == 1
    logd = integrand.f_logdensity(x)
    @assert _cuba_valid_value(logd)
    y = exp(logd)
    @assert _cuba_valid_value(y)
    f[first(idxs)] = y
    @assert all(_cuba_valid_value,f)
    return f
end

function (integrand::CubaIntegrand)(X::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real})
    idxs1 = axes(f, 1)
    @assert length(idxs1) == 1
    idxs2 = axes(f, 2)
    @assert idxs2 == axes(X, 2)
    xs = nestedview(X)
    @threads for i in idxs2
        logd = integrand.f_logdensity(xs[i])
        @assert _cuba_valid_value(logd)
        y = exp(logd)
        @assert _cuba_valid_value(y)
        f[first(idxs1), i] = y
    end
    return f
end


function _integrate_impl_cuba(integrand::CubaIntegrand, algorithm::VEGASIntegration, context::BATContext)
    r = Cuba.vegas(
        integrand, integrand.dof, 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        nstart = algorithm.nstart, nincrease = algorithm.nincrease, nbatch = algorithm.nbatch
    )
end


function _integrate_impl_cuba(integrand::CubaIntegrand, algorithm::SuaveIntegration, context::BATContext)
    Cuba.suave(
        integrand, integrand.dof, 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        nnew = algorithm.nnew, nmin = algorithm.nmin, flatness = algorithm.flatness
    )
end


function _integrate_impl_cuba(integrand::CubaIntegrand, algorithm::DivonneIntegration, context::BATContext)
    Cuba.divonne(
        integrand, integrand.dof, 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        key1 = algorithm.key1, key2 = algorithm.key2, key3 = algorithm.key3,
        maxpass = algorithm.maxpass, border = algorithm.border, maxchisq = algorithm.maxchisq,
        mindeviation = algorithm.mindeviation, ngiven = algorithm.ngiven, ldxgiven = algorithm.ldxgiven,
        nextra = algorithm.nextra
    )
end


function _integrate_impl_cuba(integrand::CubaIntegrand, algorithm::CuhreIntegration, context::BATContext)
    Cuba.cuhre(
        integrand, integrand.dof, 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        key = algorithm.key
    )
end


function BAT.bat_integrate_impl(target::MeasureLike, algorithm::CubaIntegration, context::BATContext)
    measure = batmeasure(target)
    transformed_measure, _ = transform_and_unshape(algorithm.trafo, measure, context)

    if !BAT.has_uhc_support(transformed_measure)
        throw(ArgumentError("CUBA integration requires measures are supported only on the unit hypercube"))
    end

    renormalized_measure, logweight = auto_renormalize(transformed_measure)
    dof = totalndof(varshape(renormalized_measure))
    integrand = CubaIntegrand(logdensityof(renormalized_measure), dof)

    r_cuba = _integrate_impl_cuba(integrand, algorithm, context)

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

    (value, error) = first(r_cuba.integral), first(r_cuba.error)
    rescaled_value, rescaled_error = exp(BigFloat(log(value) - logweight)), exp(BigFloat(log(error) - logweight))
    result = Measurements.measurement(rescaled_value, rescaled_error)
    return (result = result, logweight = logweight, cuba_result = r_cuba)
end


end # module BATCubaExt
