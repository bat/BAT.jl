# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATMGVIExt

import MGVI
using MGVI: MGVIContext, MGVIConfig, MGVIResult, mgvi_step, mgvi_sample

import BAT

import BAT
BAT.pkgext(::Val{:MGVI}) = BAT.PackageExtension{:MGVI}()

using BAT: MeasureLike, BATMeasure, DensitySample, DensitySampleVector, BATContext, unevaluated
using BAT: transform_and_unshape, bat_initval, apply_trafo_to_init, exec_map!
using BAT: getlikelihood, getprior, StandardMvNormal
using BAT: checked_logdensityof
using BAT: get_adselector
using BAT: should_log_progress_now
using BAT: MGVISampling, FixedMGVISchedule, MGVISampleInfo, is_std_mvnormal

using HeterogeneousComputing: get_gencontext

using Random
using ArraysOfArrays
using DensityInterface, InverseFunctions
#using ValueShapes
using Statistics: mean
using Accessors: @reset
using Printf: @sprintf


function BAT.ext_default(::BAT.PackageExtension{:MGVI}, ::Val{:MGVI_CONFIG})
    return MGVIConfig(optimizer = MGVI.NewtonCG(steps = 4))
end


function _mgvi_schedule_first(schedule::FixedMGVISchedule)
    real_n, state = iterate(schedule.nsamples)
    return round(Int, real_n), state
end

function _mgvi_schedule_next(schedule::FixedMGVISchedule, result::MGVIResult, schedule_state)
    iter_result = iterate(schedule.nsamples, schedule_state)
    if isnothing(iter_result)
        return nothing
    else
        real_n, state = iter_result
        return round(Int, real_n), state
    end
end


function _append_mgvi_samples!(smpls::DensitySampleVector, m::BATMeasure, flat_samples::AbstractMatrix{<:Real}, info::MGVISampleInfo)
    new_smpls_v = nestedview(flat_samples)
    n = length(new_smpls_v)
    new_logd = similar(smpls.logd, n)
    exec_map!(checked_logdensityof(m), BAT.default_executor(), new_logd, new_smpls_v)
    new_weight = similar(smpls.weight, n)
    fill!(new_weight, one(eltype(new_weight)))
    mean_new_logd = mean(new_logd)
    fixed_mnlp = isnan(info.mnlp) ? -mean_new_logd : info.mnlp
    @assert fixed_mnlp ≈ -mean_new_logd
    @reset info.mnlp = fixed_mnlp
    new_info = similar(smpls.info, n)
    fill!(new_info, info)
    new_aux = similar(smpls.aux, n)
    fill!(new_aux, nothing)
    append!(smpls.v, new_smpls_v); append!(smpls.logd, new_logd); append!(smpls.weight, new_weight)
    append!(smpls.info, new_info); append!(smpls.aux, new_aux)
end

function BAT.evalmeasure_impl(m::BATMeasure, algorithm::MGVISampling, context::BATContext)
    start_time = time()
    log_time = start_time
    (; pretransform, init, nsamples, schedule, config, store_unconverged) = algorithm
    mgvi_context = MGVIContext(get_gencontext(context), BAT._get_checked_adselector(context, :MGVISampling))

    transformed_m, f_pretransform = transform_and_unshape(pretransform, m, context)
    transformed_m_uneval = unevaluated(transformed_m)

    likelihood, prior = getlikelihood(transformed_m_uneval), getprior(transformed_m_uneval)
    if !is_std_mvnormal(prior)
        throw(ArgumentError("$(nameof(typeof(algorithm))) can't be used for measures that do not have a standard multivariate normal prior after `pretransform`"))
    end

    f_model, obs = try
        BAT._get_model(likelihood), BAT._get_observation(likelihood)
    catch err
        if err isa MethodError
            throw(ArgumentError("$(nameof(typeof(algorithm))) requires a likelihood based on a forward model and observed data, but don't know how to extract them from a likelihood of type $(nameof(typeof(likelihood)))."))
        else
            rethrow()
        end
    end

    @debug "Initialzing MGVI initial center point."

    initalg = apply_trafo_to_init(f_pretransform, init)
    center = collect(bat_initval(transformed_m, initalg, context).result)

    step_nsamples, schedule_state = let sched_first = _mgvi_schedule_first(schedule)
        if !isnothing(sched_first)
            sched_first
        else
            throw(ArgumentError("MGVI schedule is empty."))
        end
    end

    result, center = mgvi_step(f_model, obs, step_nsamples, center, config, mgvi_context)
    nsteps::Int = 1

    should_log, log_time, elapsed_time = should_log_progress_now(start_time, log_time)
    if should_log
        @debug "MGVI step $nsteps with $step_nsamples independent samples, current MNLP $(result.mnlp) after $(@sprintf "%.1f s" elapsed_time)."
    end

    dummy_sample = DensitySample(center, zero(result.mnlp), one(BAT._IntWeightType), MGVISampleInfo(0, false, zero(result.mnlp)), nothing)
    transformed_smpls = DensitySampleVector(typeof(dummy_sample), length(center))
    if store_unconverged
        _append_mgvi_samples!(transformed_smpls, transformed_m_uneval, result.samples, MGVISampleInfo(nsteps, false, result.mnlp))
    end

    isdone::Bool = false
    while !isdone
        let sched_next = _mgvi_schedule_next(schedule, result, schedule_state)
            if !isnothing(sched_next)
                step_nsamples, schedule_state = sched_next
                result, center = mgvi_step(f_model, obs, step_nsamples, center, config, mgvi_context)
                nsteps += 1
                if store_unconverged
                    _append_mgvi_samples!(transformed_smpls, transformed_m_uneval, result.samples, MGVISampleInfo(nsteps, false, result.mnlp))
                end
            else
                isdone = true
            end
        end

        should_log, log_time, elapsed_time = should_log_progress_now(start_time, log_time)
        if should_log
            @debug "MGVI step $nsteps with $step_nsamples independent samples, current MNLP $(result.mnlp) after $(@sprintf "%.1f s" elapsed_time)."
        end
    end

    elapsed_time = time() - start_time
    @debug "Finished MGVI iterations, completed $nsteps steps."

    final_flat_smpls = mgvi_sample(f_model, obs, nsamples, center, config, mgvi_context)
    nsteps += 1
    n_samples_total = size(final_flat_smpls, 2)
    n_samples_indep = div(n_samples_total, 2)
    _append_mgvi_samples!(transformed_smpls, transformed_m_uneval, final_flat_smpls, MGVISampleInfo(nsteps, true, oftype(result.mnlp,NaN)))

    elapsed_time = time() - start_time
    @debug "Generated final MGVI samples in transformed space after $nsteps, produced $n_samples_indep independent samples after $(@sprintf "%.1f s" elapsed_time)."

    smpls = inverse(f_pretransform).(transformed_smpls)
    smpls_mnlp = last(smpls.info).mnlp

    elapsed_time = time() - start_time
    @debug "Completed MGVI sampling after $nsteps, produced $n_samples_indep independent samples after $(@sprintf "%.1f s" elapsed_time)."

    evalresult = (
        result_pretransform = transformed_smpls, f_pretransform = f_pretransform, 
        mnlp = smpls_mnlp
    )
    dsm = DensitySampleMeasure(smpls, dof = MeasureBase.getdof(em), ess = n_samples)

    return EvalMeasureImplReturn(;
        empirical = dsm,
        # ToDo:
        #approx = ...,
        #modes = ,,,,
        #samplegen = ,,,,
        evalresult = evalresult
    )
end


end # module BATMGVIExt
