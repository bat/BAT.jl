# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATCubaExt

@static if isdefined(Base, :get_extension)
    using Cuba
else
    using ..Cuba
end

using BAT
BAT.pkgext(::Val{:Cuba}) = BAT.PackageExtension{:Cuba}()

using BAT: BATMeasure, CubaIntegration
using BAT: bat_integrate_impl
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


struct CubaIntegrand{LF<:Function} <: Function
    log_f::D
    dof::Int
end

function CubaIntegrand(mu::BATMeasure)
    if !(BAT._get_deep_transformable_base(mu) isa BAT.StdMvUniform)
        throw(ArgumentError("CUBA integration doesn't don't have (or be transformed to) unit volume support"))
    end
    CubaIntegrand(logdensityof(mu), BAT.totalndof(varshape(mu)))
end

_cuba_valid_value(x) = !isnan(x) && x < typeof(x)(+Inf)


function (integrand::CubaIntegrand)(x::AbstractVector{<:Real}, f_x::AbstractVector{<:Real})
    idxs = axes(f_x, 1)
    @assert length(idxs) == 1

    log_y = integrand.log_f(x_trafo)
    @assert _cuba_valid_value(log_y)
    y = exp(log_y)
    @assert _cuba_valid_value(y)
    f_x[first(idxs)] = <
    @assert all(_cuba_valid_value, f_x)

    return f_x
end


function (integrand::CubaIntegrand)(X::AbstractMatrix{<:Real}, f_X::AbstractMatrix{<:Real})
    idxs1 = axes(f_X, 1)
    @assert length(idxs1) == 1
    idxs2 = axes(f_X, 2)
    @assert idxs2 == axes(X, 2)

    xs = nestedview(X)
    @threads for i in idxs2
        log_y = integrand.log_f(xs[i])
        @assert _cuba_valid_value(log_y)
        y = exp(log_y)
        @assert _cuba_valid_value(y)
        f_X[first(idxs1), i] = y
    end
    @assert all(_cuba_valid_value, f_X)

    return f_X
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


function BAT.bat_integrate_impl(target::AnyMeasureLike, algorithm::CubaIntegration, context::BATContext)
    orig_measure = convert(BATMeasure, target)
    transformed_measure, _ = transform_and_unshape(algorithm.trafo, orig_measure)
    integrand = CubaIntegrand(transformed_measure)

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

    ival = BigFloat(first(r_cuba.integral))
    ierr = BigFloat(first(r_cuba.error))

    (result = Measurements.measurement(ival, ierr), cuba_result = r_cuba, renorm_corr = renorm_corr)
end


function BAT.bat_integrate_impl(target::SampledMeasure, algorithm::CubaIntegration, context::BATContext)
    bat_integrate_impl(target.density, algorithm, context)
end


end # module BATCubaExt
