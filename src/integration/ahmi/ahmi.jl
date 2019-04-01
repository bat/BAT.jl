
include("data_types.jl")
include("data_tree.jl")

include("util.jl")

include("point_cloud.jl")
include("integration_volume.jl")

include("hyper_rectangle.jl")
include("whitening_transformation.jl")
include("harmonic_mean_integration.jl")
include("hm_integration_rectangle.jl")

include("uncertainty.jl")


# ToDo: rename hm_... functions to ahmi_...

export hm_init
export hm_whiteningtransformation!
export hm_createpartitioningtree!
export hm_findstartingsamples!
export hm_determinetolerance!
export hm_hyperrectanglecreation!
export hm_integratehyperrectangles!

export hm_integrate!
export hm_integrate

export ahmi_integrate

export split_samples
export split_dataset

export DataSet
export WhiteningResult
export SpacePartitioningTree
export IntermediateResult

export HMIPrecisionSettings
export HMIFastSettings
export HMIStandardSettings
export HMIMultiThreadingSettings
export HMISettings

export data_whitening
export isinitialized

export HMIData
export HMIResult
export PointCloud
export IntegrationVolume
