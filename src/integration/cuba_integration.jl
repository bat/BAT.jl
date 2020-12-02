# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct CubaIntegrand{D<:AbstractDensity} <: Function
    density::D
end


_cuba_valid_value(x) = !isnan(x) && x < typeof(x)(+Inf)


function (integrand::CubaIntegrand)(x::AbstractVector{<:Real}, f::AbstractVector{<:Real})
    idxs = axes(f, 1)
    @assert length(idxs) == 1

    density = integrand.density
    vol = spatialvolume(var_bounds(density))
    logv = log_volume(vol)
    @assert _cuba_valid_value(logv)

    x_trafo = fromuhc(x, vol)
    logd = eval_logval(density, x_trafo)
    @assert _cuba_valid_value(logd)

    f[first(idxs)] = exp(logd + logv)
    @assert all(_cuba_valid_value,f)

    f
end


function (integrand::CubaIntegrand)(X::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real})
    idxs1 = axes(f, 1)
    @assert length(idxs1) == 1
    idxs2 = axes(f, 2)
    @assert idxs2 == axes(X, 2)

    density = integrand.density
    vol = spatialvolume(var_bounds(density))
    logv = log_volume(vol)
    @assert _cuba_valid_value(logv)

    x_trafo = fromuhc(nestedview(X), vol)
    @threads for i in idxs2
        logd = eval_logval(density, x_trafo[i])
        @assert _cuba_valid_value(logd)
        f[first(idxs1), i] = exp(logd + logv)
    end

    f
end



"""
    VEGASIntegration

VEGASIntegration integration algorithm.

Only supports densities with finite rectangular bounds.

!!! note

    This functionality is only available then the
    [Cuba](https://github.com/giordano/Cuba.jl) package is loaded (e.g. via
    `import CUBA`).
"""
@with_kw struct VEGASIntegration <: IntegrationAlgorithm
    rtol::Float64 = Cuba.RTOL
    atol::Float64 = Cuba.ATOL
    minevals::Int = Cuba.MINEVALS
    maxevals::Int = Cuba.MAXEVALS
    nstart::Int = Cuba.NSTART
    nincrease::Int = Cuba.NINCREASE
    nbatch::Int = Cuba.NBATCH
    nthreads::Int = Base.Threads.nthreads()
end
export VEGASIntegration


function bat_integrate_impl(target::AbstractDensity, algorithm::VEGASIntegration)
    integrand = CubaIntegrand(target)
    
    r = Cuba.vegas(
        integrand, totalndof(target), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        nstart = algorithm.nstart, nincrease = algorithm.nincrease, nbatch = algorithm.nbatch
    )

    (result = Measurements.measurement(first(r.integral), first(r.error)), cuba = r)
end



"""
    SuaveIntegration

    SuaveIntegration integration algorithm.

Only supports densities with finite rectangular bounds.

!!! note

    This functionality is only available then the
    [Cuba](https://github.com/giordano/Cuba.jl) package is loaded (e.g. via
    `import CUBA`).
"""
@with_kw struct SuaveIntegration <: IntegrationAlgorithm
    rtol::Float64 = Cuba.RTOL
    atol::Float64 = Cuba.ATOL
    minevals::Int = Cuba.MINEVALS
    maxevals::Int = Cuba.MAXEVALS
    nnew::Int = Cuba.NNEW
    nmin::Int = Cuba.NMIN
    flatness::Float64 = Cuba.FLATNESS
    nthreads::Int = Base.Threads.nthreads()
end
export SuaveIntegration


function bat_integrate_impl(target::AbstractDensity, algorithm::SuaveIntegration)
    integrand = CubaIntegrand(target)
    
    r = Cuba.suave(
        integrand, totalndof(target), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        nnew = algorithm.nnew, nmin = algorithm.nmin, flatness = algorithm.flatness
    )

    (result = Measurements.measurement(first(r.integral), first(r.error)), cuba = r)
end



"""
    DivonneIntegration

    DivonneIntegration integration algorithm.

Only supports densities with finite rectangular bounds.

!!! note

    This functionality is only available then the
    [Cuba](https://github.com/giordano/Cuba.jl) package is loaded (e.g. via
    `import CUBA`).
"""
@with_kw struct DivonneIntegration <: IntegrationAlgorithm
    rtol::Float64 = Cuba.RTOL
    atol::Float64 = Cuba.ATOL
    minevals::Int = Cuba.MINEVALS
    maxevals::Int = Cuba.MAXEVALS
    key1::Int = Cuba.KEY1
    key2::Int = Cuba.KEY2
    key3::Int = Cuba.KEY3
    maxpass::Int = Cuba.MAXPASS
    border::Float64 = Cuba.BORDER
    maxchisq::Float64 = Cuba.MAXCHISQ
    mindeviation::Float64 = Cuba.MINDEVIATION
    ngiven::Int = Cuba.NGIVEN
    ldxgiven::Int = Cuba.LDXGIVEN
    nextra::Int = Cuba.NEXTRA
    nthreads::Int = Base.Threads.nthreads()
end
export DivonneIntegration


function bat_integrate_impl(target::AbstractDensity, algorithm::DivonneIntegration)
    integrand = CubaIntegrand(target)
    
    r = Cuba.divonne(
        integrand, totalndof(target), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        key1 = algorithm.key1, key2 = algorithm.key2, key3 = algorithm.key3,
        maxpass = algorithm.maxpass, border = algorithm.border, maxchisq = algorithm.maxchisq,
        mindeviation = algorithm.mindeviation, ngiven = algorithm.ngiven, ldxgiven = algorithm.ldxgiven,
        nextra = algorithm.nextra
    )

    (result = Measurements.measurement(first(r.integral), first(r.error)), cuba = r)
end



"""
    CuhreIntegration

    CuhreIntegration integration algorithm.

Only supports densities with finite rectangular bounds.

!!! note

    This functionality is only available then the
    [Cuba](https://github.com/giordano/Cuba.jl) package is loaded (e.g. via
    `import CUBA`).
"""
@with_kw struct CuhreIntegration <: IntegrationAlgorithm
    rtol::Float64 = Cuba.RTOL
    atol::Float64 = Cuba.ATOL
    minevals::Int = Cuba.MINEVALS
    maxevals::Int = Cuba.MAXEVALS
    key::Int = Cuba.KEY
    nthreads::Int = Base.Threads.nthreads()
end
export CuhreIntegration


function bat_integrate_impl(target::AnyDensityLike, algorithm::CuhreIntegration)
    density = convert(AbstractDensity, target)

    integrand = CubaIntegrand(density)
    
    r = Cuba.cuhre(
        integrand, totalndof(density), 1, nvec = algorithm.nthreads,
        rtol = algorithm.rtol, atol = algorithm.atol,
        minevals = algorithm.minevals, maxevals = algorithm.maxevals,
        key = algorithm.key
    )

    (result = Measurements.measurement(first(r.integral), first(r.error)), cuba = r)
end



const CubaIntegration = Union{VEGASIntegration, SuaveIntegration, DivonneIntegration, CuhreIntegration}

function bat_integrate_impl(target::AnyDensityLike, algorithm::CubaIntegration)
    density = convert(AbstractDensity, target)
    bat_integrate_impl(density, algorithm)
end

