# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATMGVIExt

using MGVI

using BAT

BAT.pkgext(::Val{:MGVI}) = BAT.PackageExtension{:MGVI}()

using BAT: MeasureLike, BATMeasure
using BAT: transform_and_unshape, measure_support, all_active_names, exec_map!

using Random
using ArraysOfArrays
using DensityInterface, InverseFunctions, ValueShapes
import Measurements


function BAT.bat_sample_impl(m::BATMeasure, algorithm::ReactiveNestedSampling, context::BATContext)
    transformed_m, trafo = transform_and_unshape(algorithm.trafo, m, context)

    if !BAT.has_uhc_support(transformed_m)
        throw(ArgumentError("$algorithm doesn't measures that are not limited to the unit hypercube"))
    end

    # !!!!!!!!!!!!!

    return (
        result = smpls, result_trafo = transformed_smpls, trafo = trafo,
        uwresult = uwsmpls, uwresult_trafo = uwtransformed_smpls,
        logintegral = logintegral, ess = ess,
        info = r
    )
end


end # module BATMGVIExt
