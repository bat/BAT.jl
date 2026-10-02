var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#BAT-Documentation-1",
    "page": "Home",
    "title": "BAT Documentation",
    "category": "section",
    "text": "BAT.jl is the Julia version of the Bayesian Analysis Toolkit. It is designed to help solve statistical problems encountered in Bayesian inference. Typical examples are the extraction of the values of the free parameters of a model, the comparison of different models in the light of a given data set, and the test of the validity of a model to represent the data set at hand. BAT.jl aims to provide multiple algorithms that give access to the full Bayesian posterior distribution, to enable parameter estimation, limit setting and uncertainty propagation. BAT.jl also provides supporting functionality like plotting recipes and reporting functions.This package is a complete rewrite of the previous C++-BAT in Julia. BAT.jl provides several improvements over it\'s C++ predecessor, but has not yet reached feature parity in all areas. There is no backward compatibility, but the spirit is the same: providing a tool for Bayesian computations of complex models that require application-specific code.BAT.jl is implemented in pure Julia and allows for a flexible definition of mathematical models and applications while enabling the user to code for the performance required for computationally expensive numerical operations. BAT.jl provides implementations (internally of via other Julia packages) of algorithms for sampling, optimization and integration. While predefined models are (resp. will soon be) provided for standard cases, such as simple counting experiments, binomial problems or Gaussian models, BAT\'s main strength lies in the analysis of complex models. The package is designed to enable multi-threaded and distributed code execution at various levels, multi-threaded MCMC chains are provided out-of-the-box.In addition to likelihood functions implemented in Julia, BAT.jl provides a lightweight binary protocol to connect to functions written in other languages and running in separate processes (code for likelihoods written in C++ is included)."
},

{
    "location": "#Getting-started-1",
    "page": "Home",
    "title": "Getting started",
    "category": "section",
    "text": ""
},

{
    "location": "#Prerequisites-1",
    "page": "Home",
    "title": "Prerequisites",
    "category": "section",
    "text": "TODO: ..."
},

{
    "location": "#How-to-get-started-/-Tutorial-1",
    "page": "Home",
    "title": "How-to-get-started / Tutorial",
    "category": "section",
    "text": "TODO: ..."
},

{
    "location": "#Developer-Instructions-1",
    "page": "Home",
    "title": "Developer Instructions",
    "category": "section",
    "text": "To generate and view a local version of the documentation, runcd docs\njulia make.jl localthen open \"docs/build/index.html\" in your browser.When changing the code of BAT.jl and testing snippets and examples in the REPL, automatic code reloading comes in very handy. Try out Revise.jl."
},

{
    "location": "#Documentation-1",
    "page": "Home",
    "title": "Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "#User-guide-1",
    "page": "Home",
    "title": "User guide",
    "category": "section",
    "text": ""
},

{
    "location": "#FAQ-1",
    "page": "Home",
    "title": "FAQ",
    "category": "section",
    "text": ""
},

{
    "location": "#Publications-and-talks-1",
    "page": "Home",
    "title": "Publications and talks",
    "category": "section",
    "text": ""
},

{
    "location": "#How-to-cite-BAT.jl-1",
    "page": "Home",
    "title": "How to cite BAT.jl",
    "category": "section",
    "text": ""
},

{
    "location": "#LICENSE-1",
    "page": "Home",
    "title": "LICENSE",
    "category": "section",
    "text": ""
},

{
    "location": "#Algorithms-1",
    "page": "Home",
    "title": "Algorithms",
    "category": "section",
    "text": ""
},

{
    "location": "#Sampling-algorithms-(and-interfaces)-1",
    "page": "Home",
    "title": "Sampling algorithms (and interfaces)",
    "category": "section",
    "text": "TODO: List algorithms and short short descriptions."
},

{
    "location": "#Integration-algorithms-(and-interfaces)-1",
    "page": "Home",
    "title": "Integration algorithms (and interfaces)",
    "category": "section",
    "text": "TODO: List algorithms and short short descriptions."
},

{
    "location": "#Optimization-algorithms-(and-interfaces)-1",
    "page": "Home",
    "title": "Optimization algorithms (and interfaces)",
    "category": "section",
    "text": "TODO: List algorithms and short short descriptions."
},

{
    "location": "#Other-algorithms-(and-interfaces)-1",
    "page": "Home",
    "title": "Other algorithms (and interfaces)",
    "category": "section",
    "text": ""
},

{
    "location": "#Interfaces-1",
    "page": "Home",
    "title": "Interfaces",
    "category": "section",
    "text": "TODO: List interfaces"
},

{
    "location": "#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": ""
},

{
    "location": "#Common-models-and-problems-1",
    "page": "Home",
    "title": "Common models and problems",
    "category": "section",
    "text": ""
},

{
    "location": "#The-1-D-Gaussian-model-1",
    "page": "Home",
    "title": "The 1-D Gaussian model",
    "category": "section",
    "text": ""
},

{
    "location": "#The-Poisson-problem-(counting-experiments)-1",
    "page": "Home",
    "title": "The Poisson problem (counting experiments)",
    "category": "section",
    "text": ""
},

{
    "location": "#The-binomial-case-1",
    "page": "Home",
    "title": "The binomial case",
    "category": "section",
    "text": ""
},

{
    "location": "#Add-more-models-and-problems-1",
    "page": "Home",
    "title": "Add more models and problems",
    "category": "section",
    "text": ""
},

{
    "location": "#Published-scientific-examples-1",
    "page": "Home",
    "title": "Published scientific examples",
    "category": "section",
    "text": ""
},

{
    "location": "#A-multivariate-Gaussian-combination-model-(similar-to-BLUE)-1",
    "page": "Home",
    "title": "A multivariate Gaussian combination model (similar to BLUE)",
    "category": "section",
    "text": ""
},

{
    "location": "#The-EFTfitter-1",
    "page": "Home",
    "title": "The EFTfitter",
    "category": "section",
    "text": ""
},

{
    "location": "#Benchmarks-and-performance-tests-1",
    "page": "Home",
    "title": "Benchmarks and performance tests",
    "category": "section",
    "text": ""
},

{
    "location": "#Acknowledgements-1",
    "page": "Home",
    "title": "Acknowledgements",
    "category": "section",
    "text": "We acknowledge the contributions from all the BAT.jl users who help us make BAT.jl a better project. Your help is very welcome!Development of BAT.jl has been supported by funding fromDeutsche Forschungsgemeinschaft (DFG, German Research Foundation)"
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": "Coming soon ..."
},

{
    "location": "basics/#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "basics/#Mathematical-basics-1",
    "page": "Basics",
    "title": "Mathematical basics",
    "category": "section",
    "text": "... to be written ..."
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "DocTestSetup  = quote\n    using BAT\nend"
},

{
    "location": "api/#Types-1",
    "page": "API",
    "title": "Types",
    "category": "section",
    "text": "Order = [:type]"
},

{
    "location": "api/#Functions-1",
    "page": "API",
    "title": "Functions",
    "category": "section",
    "text": "Order = [:function]"
},

{
    "location": "api/#BAT.AbstractDensity",
    "page": "API",
    "title": "BAT.AbstractDensity",
    "category": "type",
    "text": "AbstractDensity\n\nThe following functions must be implemented for subtypes:\n\nBAT.nparams\nBAT.unsafe_density_logval\n\nIn some cases, it may be desirable to override the default implementations of the functions\n\nBAT.exec_capabilities\nBAT.unsafe_density_logval!\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.AbstractMCMCCallback",
    "page": "API",
    "title": "BAT.AbstractMCMCCallback",
    "category": "type",
    "text": "AbstractMCMCCallback <: Function\n\nSubtypes (e.g. X) must support\n\n(::X)(level::Integer, chain::MCMCIterator) => nothing\n(::X)(level::Integer, tuner::AbstractMCMCTuner) => nothing\n\nto be compabtible with mcmc_iterate!, mcmc_tune_burnin!, etc.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.AbstractProposalDist",
    "page": "API",
    "title": "BAT.AbstractProposalDist",
    "category": "type",
    "text": "AbstractProposalDist\n\nThe following functions must be implemented for subtypes:\n\nBAT.distribution_logpdf\nBAT.proposal_rand!\nBAT.nparams, returning the number of parameters (i.e. dimensionality).\nLinearAlgebra.issymmetric, indicating whether p(a -> b) == p(b -> a) holds true.\n\nIn some cases, it may be desirable to override the default implementation of BAT.distribution_logpdf!.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.DataSet",
    "page": "API",
    "title": "BAT.DataSet",
    "category": "type",
    "text": "DataSet{T<:AbstractFloat, I<:Integer}\n\nHolds the MCMC output. For construction use constructor: function DataSet{T<:Real}(data::Matrix{T}, logprob::Vector{T}, weights::Vector{T})\n\nVariables\n\n\'data\' : An P x N array with N data points with P parameters.\n\'logprob\' : The logarithmic probability for each samples stored in an array\n\'weights\' : How often each sample occurred. Set to an array of ones if working directly on MCMC output\n\'ids\' : Array which is used to assign each sample to a batch, required for the cov. weighed uncertainty estimation\n.sortids : an array of indices which stores the original ordering of the samples (the space partitioning tree reorders the samples), required to calculate an effective sample size.\n\'N\' : number of samples\n\'P\' : number of parameters\n\'nsubsets\' : the number of batches\n\'iswhitened\' : a boolean value which indicates whether the data set is iswhitened\n\'isnew\' : a boolean value which indicates whether the data set was swapped out with a different one (it is possible to redo the integration with a different sample set using previously generated hyper-rectangles)\n\'partitioningtree\' : The space partitioning tree, used to efficiently identify samples in a point cloud\n\'startingIDs\' : The Hyper-Rectangle Seed Samples are stored in this array\n\'tolerance\' : A threshold required for the hyper-rectangle creation process.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.ExecCapabilities",
    "page": "API",
    "title": "BAT.ExecCapabilities",
    "category": "type",
    "text": "struct ExecCapabilities\n    nthreads::Int\n    threadsafe::Bool\n    nprocs::Int\n    remotesafe::Bool\nend\n\nSpecifies the execution capabilities of functions that support an ExecContext argument.\n\nnthreads specifies the maximum number of threads the function can utilize efficiently, internally. If nthreads <= 1, the function implementation is single-threaded. nthreads == 0 indicates that the function is cheap and that when used in combination with other functions, their capabilities should dominate.\n\nthreadsafe specifies whether the function is thread-safe, and can be can be run on multiple threads in parallel by the caller.\n\nnprocs specifies the maximum number of worker processes the function can utilize efficiently, internally. If procs <= 1, the function cannot use worker processes. nthreads == 0 carries equivalent meaning to nthreads == 0.\n\nremotesafe specifies that the function can be run on a remote thread, it implies that the function arguments can be (de-)serialized safely.\n\nFunctions with an ExecContext argument should announce their capabilities via methods of exec_capabilities. Functions should, ideally, either support internal multithreading (nthreads > 1) or be thread-safe (threadsafe == true). Likewise, functions should either utilize worker processes (nprocs > 1) internally or support remote execution (remotesafe == true) by the caller.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.ExecContext",
    "page": "API",
    "title": "BAT.ExecContext",
    "category": "type",
    "text": "struct ExecContext\n    use_threads::Bool\n    onprocs::Vector{Int64}\nend\n\nFunctions that take an ExecContext argument must limit their use of threads and processes accordingly. Depending on use_threads, the function may use all (or only a single) thread(s) on each process in onprocs (in addition to the current thread on the current process).\n\nThe caller may choose to change the ExecContext from call to call, based on execution time and latency measurements, etc.\n\nFunctions can announce their BAT.ExecCapabilities via exec_capabilities.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.GRConvergence",
    "page": "API",
    "title": "BAT.GRConvergence",
    "category": "type",
    "text": "GRConvergence\n\nGelman-Rubin maximum(R^2) convergence test.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.GenericDensity",
    "page": "API",
    "title": "BAT.GenericDensity",
    "category": "type",
    "text": "GenericDensity{F} <: AbstractDensity\n\nConstructors:\n\nGenericDensity(log_f, nparams::Int)\n\nTurns the logarithmic density function log_f into a BAT-compatible AbstractDensity. log_f must support\n\n`log_f(params::AbstractVector{<:Real})::Real`\n\nwith length(params) == nparams.\n\nIt must be safe to execute log_f in parallel on multiple threads and processes.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.HMIData",
    "page": "API",
    "title": "BAT.HMIData",
    "category": "type",
    "text": "HMIData{T<:AbstractFloat, I<:Integer}\n\nIncludes all the informations of the integration process, including a list of hyper-rectangles, the results of the whitening transformation, the starting ids, and the average number of points and volume of the created hyper-rectangles.\n\nVariables\n\n\'dataset1\' : Data Set 1\n\'dataset2\' : Data Set 2\n\'whiteningresult\' : contains the whitening matrix and its determinant, required to scale the final integral estimate\n\'volumelist1\' : An array of integration volumes created using dataset1, but filled with samples from dataset2\n\'volumelist2\' : An array of integration volumes created using dataset2, but filled with samples from dataset1\n\'cubelist1\' : An array of small hyper-cubes created around seeding samples of dataset 1\n\'cubelist2\' : An array of small hyper-cubes created around seeding samples of dataset 2\n\'iterations1\' : The number of volume adapting iterations for the creating volumelist1\n\'iterations2\' : The number of volume adapting iterations for the creating volumelist2\n\'rejectedrects1\' : An array of ids, indicating which hyper-rectangles of volumelist1 were rejected due to trimming\n\'rejectedrects2\' : An array of ids, indicating which hyper-rectangles of volumelist2 were rejected due to trimming\n\'integralestimates\' : A dictionary containing the final integral estimates with uncertainty estimation using different uncertainty estimators. Also includes all intermediate results required for the integral estimate combination\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.HMISettings",
    "page": "API",
    "title": "BAT.HMISettings",
    "category": "type",
    "text": "HMISettings\n\nholds the settings for the hm_integrate function. There are several default constructors available: HMIFastSettings() HMIStandardSettings() HMIPrecisionSettings()\n\n#Variables\n\n\'whitening_method::Symbol\' : which whitening method to use\n\'max_startingIDs::Integer\' : influences how many starting ids are allowed to be generated\n\'maxstartingIDsfraction::AbstractFloat\' : how many points are considered as possible starting points as a fraction of total points available\n\'rect_increase::AbstractFloat\' : describes the procentual rectangle volume increase/decrease during hyperrectangle creation. Low values can increase the precision if enough points are available but can cause systematically wrong results if not enough points are available.\n\'useallrects::Bool\' : All rectangles are used for the integration process no matter how big their overlap is. If enabled the rectangles are weighted by their overlap.\n\'useMultiThreading\' : activate multithreading support.\n\'warning_minstartingids\' : the required minimum amount of starting samples\n\'dotrimming\' : determines whether the integral estimates are trimmed (1σ trim) before combining them into a final result (more robust)\n\'uncertaintyestimators\' : A dictionary of different uncertainty estimator functions. Currently three functions are available: hmcombineresultslegacy! (outdated, overestimates uncertainty significantly in higher dimensions), hmcombineresultscovweighted! (very fast) and hmcombineresults_analyticestimation! (recommended)\n\nend\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.IntegrationVolume",
    "page": "API",
    "title": "BAT.IntegrationVolume",
    "category": "type",
    "text": "IntegrationVolume{T<:AbstractFloat, I<:Integer}\n\nVariables\n\n\'pointcloud\' : holds the point cloud of the integration volume\n\'spatialvolume\' : the boundaries of the integration volume\n\'volume\' : the volume\n\nHold the point cloud and the spatial volume for integration.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.IntegrationVolume-Union{Tuple{I}, Tuple{T}, Tuple{DataSet{T,I},HyperRectVolume{T}}, Tuple{DataSet{T,I},HyperRectVolume{T},Bool}} where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.IntegrationVolume",
    "category": "method",
    "text": "IntegrationVolume(dataset::DataSet{T, I}, spvol::HyperRectVolume{T}, searchpts::Bool = true)::IntegrationVolume{T, I}\n\ncreates an integration region by calculating the point cloud an the volume of the spatial volume.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.OnlineMvMean",
    "page": "API",
    "title": "BAT.OnlineMvMean",
    "category": "type",
    "text": "OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}\n\nMulti-variate mean implemented via Kahan-Babuška-Neumaier summation.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.OnlineUvMean",
    "page": "API",
    "title": "BAT.OnlineUvMean",
    "category": "type",
    "text": "OnlineUvMean{T<:AbstractFloat}\n\nUnivariate mean implemented via Kahan-Babuška-Neumaier summation.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.PointCloud",
    "page": "API",
    "title": "BAT.PointCloud",
    "category": "type",
    "text": "PointCloud{T<:AbstractFloat, I<:Integer}\n\nStores the information of the points of an e.g. HyperRectVolume\n\nVariables\n\n\'maxLogProb\' : The maximum log. probability of one of the points inside the hyper-rectangle\n\'minLogProb\' : The minimum log. probability of one of the points inside the hyper-rectangle\n\'maxWeightProb\' : the weighted max. log. probability\n\'minWeightProb\' : the weighted min. log. probability\n\'probfactor\' : The probability factor of the hyper-rectangle\n\'probweightfactor\' : The weighted probability factor\n\'points\' : The number of points inside the hyper-rectangle\n\'pointIDs\' : the IDs of the points inside the hyper-rectangle, might be empty because it is optional and costs performance\n\'searchres\' : used to boost performance\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.PointCloud-Union{Tuple{I}, Tuple{T}, Tuple{DataSet{T,I},HyperRectVolume{T},Bool}} where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.PointCloud",
    "category": "method",
    "text": "PointCloud{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, hyperrect::HyperRectVolume{T}, searchpts::Bool = false)::PointCloud\n\ncreates a point cloud by searching the data tree for points which are inside the hyper-rectangle The parameter searchpts determines if an array of the point IDs is created as well\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.WhiteningResult",
    "page": "API",
    "title": "BAT.WhiteningResult",
    "category": "type",
    "text": "WhiteningResult{T<:AbstractFloat}\n\nStores the information obtained during the Whitening Process\n\nVariables\n\n\'determinant\' : The determinant of the whitening matrix\n\'targetprobfactor\' : The suggested target probability factor\n\'whiteningmatrix\' : The whitening matrix\n\'meanvalue\' : the mean vector of the input data\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.bat_sampler",
    "page": "API",
    "title": "BAT.bat_sampler",
    "category": "function",
    "text": "bat_sampler(d::Distribution)\n\nTries to return a BAT-compatible sampler for Distribution d. A sampler is BAT-compatible if it supports random number generation using an arbitrary AbstractRNG:\n\nrand(rng::AbstractRNG, s::SamplerType)\nrand!(rng::AbstractRNG, s::SamplerType, x::AbstractArray)\n\nIf no specific method of bat_sampler is defined for the type of d, it will default to sampler(d), which may or may not return a BAT-compatible sampler.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.density_logval",
    "page": "API",
    "title": "BAT.density_logval",
    "category": "function",
    "text": "density_logval(\n    density::AbstractDensity,\n    params::AbstractVector{<:Real},\n    exec_context::ExecContext = ExecContext()\n)\n\nVersion of density_logval for a single parameter vector.\n\nDo not implement density_logval directly for subtypes of AbstractDensity, implement BAT.unsafe_density_logval instead.\n\nSee ExecContext for thread-safety requirements.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.density_logval!",
    "page": "API",
    "title": "BAT.density_logval!",
    "category": "function",
    "text": "density_logval!(\n    r::AbstractVector{<:Real},\n    density::AbstractDensity,\n    params::VectorOfSimilarVectors{<:Real},\n    exec_context::ExecContext = ExecContext()\n)\n\nCompute log of values of a density function for multiple parameter value vectors.\n\nInput:\n\ndensity: density function\nparams: parameter values\nexec_context: Execution context\n\nOutput is stored in\n\nr: Vector of log-result values\n\nArray size requirements:\n\naxes(params, 1) == axes(r, 1)\n\nNote: density_logval! must not be called with out-of-bounds parameter vectors (see param_bounds). The result of density_logval! for parameter vectors that are out of bounds is implicitly -Inf, but for performance reasons the output is left undefined: density_logval! may fail or store arbitrary values in r.\n\nDo not implement density_logval! directly for subtypes of AbstractDensity, implement BAT.unsafe_density_logval! instead.\n\nSee ExecContext for thread-safety requirements.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.distribution_logpdf",
    "page": "API",
    "title": "BAT.distribution_logpdf",
    "category": "function",
    "text": "distribution_logpdf(\n    pdist::AbstractProposalDist,\n    params_new::AbstractVector,\n    params_old:::AbstractVector\n)\n\nAnalog to distribution_logpdf!, but for a single parameter vector.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.distribution_logpdf!",
    "page": "API",
    "title": "BAT.distribution_logpdf!",
    "category": "function",
    "text": "distribution_logpdf!(\n    p::AbstractArray,\n    pdist::AbstractProposalDist,\n    params_new::Union{AbstractVector,VectorOfSimilarVectors},\n    params_old:::Union{AbstractVector,VectorOfSimilarVectors}\n)\n\nReturns log(PDF) value of pdist for transitioning from old to new parameter values for multiple parameter sets.\n\nend\n\nInput:\n\nparams_new: New parameter values (column vectors)\nparams_old: Old parameter values (column vectors)\n\nOutput is stored in\n\np: Array of PDF values, length must match, shape is ignored\n\nArray size requirements:\n\nsize(params_old, 1) == size(params_new, 1) == length(pdist)\nsize(params_old, 2) == size(params_new, 2) or size(params_old, 2) == 1\nsize(params_new, 2) == length(p)\n\nImplementations of distribution_logpdf! must be thread-safe.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.fromuhc!",
    "page": "API",
    "title": "BAT.fromuhc!",
    "category": "function",
    "text": "fromuhc!(Y::AbstractVector, X::AbstractVector, vol::SpatialVolume)\nfromuhc!(Y::VectorOfSimilarVectors, X::VectorOfSimilarVectors, vol::SpatialVolume)\n\nBijective transformation of coordinates X within the unit hypercube to coordinates Y in vol. If X and Y are matrices, the transformation is applied to the column vectors. Use Y === X to transform in-place.\n\nUse inv(fromuhc!) to get the the inverse transformation.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.fromuhc-Tuple{AbstractArray{T,1} where T,SpatialVolume}",
    "page": "API",
    "title": "BAT.fromuhc",
    "category": "method",
    "text": "fromuhc(X::AbstractVector, vol::SpatialVolume)\nfromuhc(X::VectorOfSimilarVectors, vol::SpatialVolume)\n\nBijective transformation from unit hypercube to vol. See fromuhc!.\n\nUse inv(fromuhc) to get the the inverse transformation.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.fromui",
    "page": "API",
    "title": "BAT.fromui",
    "category": "function",
    "text": "y = fromui(x::Real, lo::Real, hi::Real)\ny = fromui(x::Real, lo_hi::ClosedInterval{<:Real})\n\nLinear bijective transformation from the unit inverval (i.e. x ∈ 0..1) to y ∈ lo..hi.\n\nUse inv(fromui) to get the the inverse transformation.\n\nUse @inbounds to disable range checking on the input value.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.hm_init-Union{Tuple{V}, Tuple{I}, Tuple{T}, Tuple{HMIData{T,I,V},HMISettings}} where V<:SpatialVolume where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.hm_init",
    "category": "method",
    "text": "function hm_init!(result, settings)\n\nSets the global multithreading setting and ensures that a minimum number of samples, dependent on the number of dimensions, are provided.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.hm_integrate!-Union{Tuple{HMIData{T,I,V}}, Tuple{V}, Tuple{I}, Tuple{T}, Tuple{HMIData{T,I,V},Symbol}} where V<:SpatialVolume where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.hm_integrate!",
    "category": "method",
    "text": "function hm_integrate!(result, settings = HMIPrecisionSettings())\n\nThis function starts the adaptive harmonic mean integration. See arXiv:1808.08051 for more details. It needs a HMIData struct as input, which holds the samples, in form of a dataset, the integration volumes and other properties, required for the integration, and the final result.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.hm_whiteningtransformation!-Union{Tuple{V}, Tuple{I}, Tuple{T}, Tuple{HMIData{T,I,V},HMISettings}} where V<:SpatialVolume where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.hm_whiteningtransformation!",
    "category": "method",
    "text": "function hm_whiteningtransformation!(result, settings)\n\nApplies a whitening transformation to the samples. A custom whitening method can be used by overriding settings.whitening_function!\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.issymmetric_around_origin",
    "page": "API",
    "title": "BAT.issymmetric_around_origin",
    "category": "function",
    "text": "issymmetric_around_origin(d::Distribution)\n\nReturns true (resp. false) if the Distribution is symmetric (resp. non-symmetric) around the origin.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.log_volume",
    "page": "API",
    "title": "BAT.log_volume",
    "category": "function",
    "text": "log_volume(vol::SpatialVolume)\n\nGet the logarithm of the volume of the space in vol.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.nparams",
    "page": "API",
    "title": "BAT.nparams",
    "category": "function",
    "text": "nparams(X::Union{AbstractParamBounds,MCMCIterator,...})\n\nGet the number of parameters of X.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.param_bounds-Tuple{AbstractDensity}",
    "page": "API",
    "title": "BAT.param_bounds",
    "category": "method",
    "text": "param_bounds(density::AbstractDensity)::AbstractParamBounds\n\nGet the parameter bounds of density. See density_logval! for the implications and handling of the bounds.\n\nUse\n\nnew_density = density[bounds::ParamVolumeBounds]\n\nto create a new density function with additional bounds.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.proposal_rand!",
    "page": "API",
    "title": "BAT.proposal_rand!",
    "category": "function",
    "text": "function proposal_rand!(\n    rng::AbstractRNG,\n    pdist::GenericProposalDist,\n    params_new::Union{AbstractVector,VectorOfSimilarVectors},\n    params_old::Union{AbstractVector,VectorOfSimilarVectors}\n)\n\nGenerate one or multiple proposed parameter vectors, based on one or multiple previous parameter vectors.\n\nInput:\n\nrng: Random number generator to use\npdist: Proposal distribution to use\nparams_old: Old parameter values (vector or column vectors, if a matrix)\n\nOutput is stored in\n\nparams_new: New parameter values (vector or column vectors, if a matrix)\n\nThe caller must guarantee:\n\nsize(params_old, 1) == size(params_new, 1)\nsize(params_old, 2) == size(params_new, 2) or size(params_old, 2) == 1\nparams_new !== params_old (no aliasing)\n\nImplementations of proposal_rand! must be thread-safe.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.spatialvolume",
    "page": "API",
    "title": "BAT.spatialvolume",
    "category": "function",
    "text": "spatialvolume(b::ParamVolumeBounds)::SpatialVolume\n\nReturns the spatial volume that defines the parameter bounds.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.MCMCCallbackWrapper",
    "page": "API",
    "title": "BAT.MCMCCallbackWrapper",
    "category": "type",
    "text": "MCMCCallbackWrapper{F} <: AbstractMCMCCallback\n\nWraps a callable object to turn it into an AbstractMCMCCallback.\n\nConstructor:\n\nMCMCCallbackWrapper(f::Any)\n\nf needs to support the call syntax of an AbstractMCMCCallback.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.SearchResult",
    "page": "API",
    "title": "BAT.SearchResult",
    "category": "type",
    "text": "SearchResult{T<:AbstractFloat, I<:Integer}\n\nStores the results of the space partitioning tree\'s search function\n\nVariables\n\n\'pointIDs\' : the IDs of samples found, might be empty because it is optional\n\'points\' : The number of points found.\n\'maxLogProb\' : the maximum log. probability of the points found.\n\'minLogProb\' : the minimum log. probability of the points found.\n\'maxWeightProb\' : the weighted minimum log. probability found.\n\'minWeightProb\' : the weighted maximum log. probfactor found.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.apply_bounds",
    "page": "API",
    "title": "BAT.apply_bounds",
    "category": "function",
    "text": "apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType)\n\nSpecify lower and upper bound via interval.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.apply_bounds!",
    "page": "API",
    "title": "BAT.apply_bounds!",
    "category": "function",
    "text": "apply_bounds!(params::AbstractVector, bounds::AbstractParamBounds)\n\nApply bounds to parameters params.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.apply_bounds-Union{Tuple{H}, Tuple{L}, Tuple{X}, Tuple{X,L,H,BoundsType}, Tuple{X,L,H,BoundsType,Any}} where H<:Real where L<:Real where X<:Real",
    "page": "API",
    "title": "BAT.apply_bounds",
    "category": "method",
    "text": "apply_bounds(x::<:Real, lo::<:Real, hi::<:Real, boundary_type::BoundsType)\n\nApply lower/upper bound lo/hi to value x. boundary_type may be hard_bounds, cyclic_bounds or reflective_bounds.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.autocrl-Union{Tuple{AbstractArray{T,1}}, Tuple{T}, Tuple{AbstractArray{T,1},AbstractArray{Int64,1}}} where T<:Real",
    "page": "API",
    "title": "BAT.autocrl",
    "category": "method",
    "text": "autocrl(xv::AbstractVector{T}, kv::AbstractVector{Int} = Vector{Int}())\n\nautocorrelation := Σ Cov[xi,x(i+k)]/Var[x]\n\nComputes the autocorrelations at various leg k of the input vector (time series) xv. The vector kv is the collections of lags to take into account\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.calculate_localmode-Tuple{Any}",
    "page": "API",
    "title": "BAT.calculate_localmode",
    "category": "method",
    "text": "calculate_localmode(hist)\n\nCalculates the modes of a 1d statsbase histogram. A vector of the bin-center of the heighest bin(s) is(are) returned.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.create_hypercube-Union{Tuple{T}, Tuple{Array{T,1},T}} where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.create_hypercube",
    "category": "method",
    "text": "create_hypercube{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume\n\ncreates a hypercube shaped spatial volume\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.create_hyperrectangle-Union{Tuple{I}, Tuple{T}, Tuple{I,DataSet{T,I},T,HMISettings}} where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.create_hyperrectangle",
    "category": "method",
    "text": "This function creates a hyper-rectangle around each starting sample. It starts by building a hyper-cube  and subsequently adapts each face individually, thus turning the hyper-cube into a hyper-rectangle. The faces are adjusted in a way to match the shape of the distribution as best as possible.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.effective_sample_size-Tuple{AbstractArray,AbstractArray{T,1} where T}",
    "page": "API",
    "title": "BAT.effective_sample_size",
    "category": "method",
    "text": "effective_sample_size(params::AbstractArray, weights::AbstractVector; with_weights=true)\n\nEffective size estimation for a (multidimensional) ElasticArray. By default applies the Kish approximation with the weigths available, but can be turned off (with_weights=false).\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.effective_sample_size-Tuple{DensitySampleVector}",
    "page": "API",
    "title": "BAT.effective_sample_size",
    "category": "method",
    "text": "effective_sample_size(samples::DensitySampleVector; with_weights=true)\n\nEffective size estimation for a (multidimensional) DensitySampleVector. By default applies the Kish approximation with the weigths available, but can be turned off (with_weights=false).\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.effective_sample_size-Union{Tuple{AbstractArray{T1,1}}, Tuple{T1}, Tuple{T2}, Tuple{AbstractArray{T1,1},AbstractArray{T2,1}}, Tuple{AbstractArray{T1,1},AbstractArray{T2,1},AbstractArray{Int64,1}}} where T1<:Real where T2<:Number",
    "page": "API",
    "title": "BAT.effective_sample_size",
    "category": "method",
    "text": "Effective size estimation for a vector of samples xv. If a weight vector w is provided, the Kish approximation is applied.\n\nBy default computes the autocorrelation up to the square root of the number of entries in the vector, unless an explicit list of lags is provided (kv).\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.eval_density_logval!",
    "page": "API",
    "title": "BAT.eval_density_logval!",
    "category": "function",
    "text": "eval_density_logval!(...)\n\nInternal function to first apply bounds and then evaluate density.\n\nGuarantees that for out-of-bounds parameters:\n\ndensity_logval is not called\nlog value of density is set to (resp. returned as) -Inf\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.eval_prior_posterior_logval!",
    "page": "API",
    "title": "BAT.eval_prior_posterior_logval!",
    "category": "function",
    "text": "BAT.eval_prior_posterior_logval!(...)\n\nInternal function to first apply bounds to the parameters and then compute prior and posterior log valued.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.exec_capabilities",
    "page": "API",
    "title": "BAT.exec_capabilities",
    "category": "function",
    "text": "exec_capabilities(f, args...)::ExecCapabilities\n\nDetermines the execution capabilities of a function f that supports an ExecContext argument, when called with arguments args.... The ExecContext argument itself is excluded from args... for exec_capabilities.\n\nBefore calling f, the caller must use\n\nexec_capabilities(f, args...)\n\nto determine the execution capabilities of f with the intended arguments, and take the resulting ExecCapabilities into account. If f is not thread-safe (but remote-safe), and the caller needs to run it on multiple threads, the caller may deep-copy the function arguments.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.find_hypercube_centers-Union{Tuple{I}, Tuple{T}, Tuple{DataSet{T,I},WhiteningResult{T},HMISettings}} where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.find_hypercube_centers",
    "category": "method",
    "text": "find_hypercube_centers(dataset::DataSet{T, I}, whiteningresult::WhiteningResult, settings::HMISettings)::Vector{I}\n\nfinds possible starting points for the hyperrectangle creation\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.gr_Rsqr-Tuple{AbstractArray{#s123,1} where #s123<:MCMCBasicStats}",
    "page": "API",
    "title": "BAT.gr_Rsqr",
    "category": "method",
    "text": "gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})\n\nGelman-Rubin R^2 for all parameters.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.hyperrectangle_creationproccess!-Union{Tuple{I}, Tuple{T}, Tuple{DataSet{T,I},T,HMISettings,Array{IntegrationVolume{T,I,HyperRectVolume{T}},1},Array{HyperRectVolume{T},1},Array{I,1},Atomic{I},Progress}} where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.hyperrectangle_creationproccess!",
    "category": "method",
    "text": "This function assigns each thread its own hyper-rectangle to build, if in multithreading-mode.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.initial_params!",
    "page": "API",
    "title": "BAT.initial_params!",
    "category": "function",
    "text": "BAT.initial_params!(\n    params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},\n    rng::AbstractRNG,\n    model::AbstractBayesianModel,\n    algorithm::MCMCAlgorithm\n)::typeof(params)\n\nFill params with random initial parameters suitable for model and algorithm. The default implementation will try to draw the initial parameters from the prior of the model.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.modify_hypercube!-Union{Tuple{T}, Tuple{HyperRectVolume{T},Array{T,1},T}} where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.modify_hypercube!",
    "category": "method",
    "text": "create_hypercube!{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume\n\nresizes a hypercube shaped spatial volume\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.modify_integrationvolume!-Union{Tuple{I}, Tuple{T}, Tuple{IntegrationVolume{T,I,V} where V<:SpatialVolume,DataSet{T,I},HyperRectVolume{T}}, Tuple{IntegrationVolume{T,I,V} where V<:SpatialVolume,DataSet{T,I},HyperRectVolume{T},Bool}} where I<:Integer where T<:AbstractFloat",
    "page": "API",
    "title": "BAT.modify_integrationvolume!",
    "category": "method",
    "text": "modify_integrationvolume!(intvol::IntegrationVolume{T, I}, dataset::DataSet{T, I}, spvol::HyperRectVolume{T}, searchpts::Bool = true)\n\nupdates an integration volume with new boundaries. Recalculates the pointcloud and volume.\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.unsafe_density_logval",
    "page": "API",
    "title": "BAT.unsafe_density_logval",
    "category": "function",
    "text": "BAT.unsafe_density_logval(\n    density::AbstractDensity,\n    params::AbstractVector{<:Real},\n    exec_context::ExecContext = ExecContext()\n)\n\nUnsafe variant of density_logval, implementations may rely on\n\nsize(params, 1) == nparams(density)\n\nThe caller must ensure that these conditions are met!\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.unsafe_density_logval!-Tuple{AbstractArray{#s102,1} where #s102<:Real,AbstractDensity,ArraysOfArrays.ArrayOfSimilarArrays{#s101,1,1,2,P} where P<:AbstractArray{#s101,2} where #s101<:Real,ExecContext}",
    "page": "API",
    "title": "BAT.unsafe_density_logval!",
    "category": "method",
    "text": "BAT.unsafe_density_logval!(\n    r::AbstractVector{<:Real},\n    density::AbstractDensity,\n    params::VectorOfSimilarVectors{<:Real},\n    exec_context::ExecContext\n)\n\nUnsafe variant of density_logval!, implementations may rely on\n\nsize(params, 1) == nparams(density)\nsize(params, 2) == length(r)\n\nThe caller must ensure that these conditions are met!\n\n\n\n\n\n"
},

{
    "location": "api/#BAT.wgt_effective_sample_size-Union{Tuple{AbstractArray{T,1}}, Tuple{T}} where T<:Real",
    "page": "API",
    "title": "BAT.wgt_effective_sample_size",
    "category": "method",
    "text": "wgt_effective_sample_size(w::AbstractVector{T})\n\nKish\'s approximation for weighted samples effectivesamplesize estimation. Computes the weighting factor for weigthed samples, where w is the vector of weigths.\n\n\n\n\n\n"
},

{
    "location": "api/#Base.intersect-Tuple{ExecCapabilities,ExecCapabilities}",
    "page": "API",
    "title": "Base.intersect",
    "category": "method",
    "text": "intersect(a:ExecCapabilities, b:ExecCapabilities)\n\nGet the intersection of execution capabilities of a and b, i.e. the ExecCapabilities that should be used when to functions are used in combination (e.g. in sequence).\n\n\n\n\n\n"
},

{
    "location": "api/#Documentation-1",
    "page": "API",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [BAT]\nOrder = [:type, :function]"
},

{
    "location": "LICENSE/#",
    "page": "LICENSE",
    "title": "LICENSE",
    "category": "page",
    "text": ""
},

{
    "location": "LICENSE/#LICENSE-1",
    "page": "LICENSE",
    "title": "LICENSE",
    "category": "section",
    "text": "using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))"
},

]}
