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

include("mgvi/mgvi_sampling.jl")

end # module BATMGVIExt
