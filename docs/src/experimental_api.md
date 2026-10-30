# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

```@docs
ARPWeighting
BAT.batalgorithm
bat_compare
bat_integrated_autocorr_len
bat_marginalmode
BAT.auto_renormalize
BAT.BinnedModeEstimator
BAT.convert_for
BAT.DistributionTransform
BAT.LowRankAffineTransform
BAT.PathfinderTransformInit
BAT.enable_error_log
BAT.error_log
BAT.EvalException
BAT.evalmeasure_impl
BAT.ext_default
BAT.get_adselector
BAT.get_valid_adselector
BAT.PackageExtension
BAT.pkgext
BAT.set_rng
batmeasure
BridgeSampling
EllipsoidalNestedSampling
EllipticalSliceMCMCSampling
GridSampler
HierarchicalDistribution
PriorImportanceSampler
ReactiveNestedSampling
SliceMCMCSampling
SobolSampler
truncate_batmeasure
ValueAndThreshold

BAT.validate_evalmeasure
BAT.MCMCChainState
BAT.MCMCChainStateInfo
BAT.MCMCIterator
BAT.MCMCProposal
BAT.MCMCProposalState
BAT.MCMCProposalTunerState
BAT.MCMCState
BAT.MCMCTempering
BAT.MCMCTransformTunerState
BAT.MeasureEvalInfo
BAT.PolarShellDistribution
BAT.SimpleMCMCProposalState
BAT.TemperingState
```

## Histogram weights and statistical overlays

Histogram `:probability` and `:pdf` normalization use temporary relative weights.
Common finite weight scaling therefore preserves their results within the
supported numerical range. Stored samples remain unchanged. Inputs must have
finite nonnegative weights and finite coordinates wherever weight is positive. Positive mass that would be lost
in the working precision raises an error. A `MarginalDist` requires positive
in-bin mass.

Histogram `:none` retains counts; `:density` divides counts by bin volume without
turning them into probabilities. These modes and report counts require a
representable absolute total. Overflow raises an error even if each bin could
individually hold its count. Legacy Plots mean/spread overlays
also retain absolute frequency-weight arithmetic and its correction; a corrected
spread requires total weight greater than one. Only requested overlays and their
selected coordinates are evaluated.

Density division avoids intermediate products of widths and total mass. The
result still must fit the numerical types actually used. Legacy Plots mean/spread
overlays reject nonfinite or collapsed positive spread and interval geometry.
Constant coordinates remain supported.

`MarginalDist` checks its dependency's actual PDF and reconstructed bin masses.
Its dependency's cached coordinate moments have separate extreme-range limits.
Histogram regions rank cells by density and accumulate integrated cell mass.
For count/probability histograms the stored heights are masses; for density/PDF
histograms the mass is height times actual bin volume. Regions preserve the
original displayed heights and include the entire cutoff-density plateau. This
intentionally replaces first-bin tie breaking. They are whole-bin highest-density
regions: achieved probability can exceed the request, sometimes substantially,
and the result is not a globally minimum-volume selection of whole bins.

The internal central-region helper uses actual left/right tail masses and retains
a boundary bin when the removed tail would exactly equal its target. Zero requests
are empty, full-mass requests retain every positive cell, and zero-mass histograms
remain empty displays. Legacy Plots region labels report achieved probability.
Custom labels and colors stay associated with their requests when requests are
reordered.

Region arithmetic factors volumes and densities before common power-of-two mass
scaling. Each positive cell must retain positive relative mass; otherwise it raises
an error. Region helpers accept ordinary integer/floating-point histogram storage,
with endpoints exactly representable in the promoted Float64 or BigFloat working
type. Arithmetic must use round-to-nearest and preserve subnormals; BigFloat inputs
at its minimum exponent boundary are rejected. These checks do not change the
caller's arithmetic mode. Finite arithmetic can change a whole boundary plateau
near a probability crossing or nearly tied
computed density; it does not guarantee identical masks under every floating-point
rescaling. Whole-bin regions use original edges, including horizontal fills and
unequal 2D cells.

Legacy Plots contours are approximate isolines of an interpolated density field,
not exact boundaries whose enclosed probability equals the selected cell mass.
Their cutoffs use the same relative-density field that is displayed. Smoothing
uses bin-index units and zero padding/cropping, then recomputes cell probabilities
within that displayed domain; it is not uniform physical-space smoothing on unequal
bins. Contours require at least two cells on each axis and a representable positive
relative-density field, with distinct Float64 centers inside their cells, even when
exact cell regions are representable. Coincident
cutoffs retain the tighter request's color. Numeric contour labels cannot be
interpreted as original density units.

Contour `smoothing` accepts a finite, nonnegative real scalar or a two-tuple
`(sigma_x, sigma_y)`. A scalar applies to both axes; zero disables smoothing on
that axis. Each axis uses Gaussian support radius `2*ceil(sigma)` after conversion
to the Float64 working type. Active smoothing is admitted before constructing its
kernel or padded input, with per-contour limits of `2^20` kernel cells, `2^23`
additional Float64 numeric-buffer slots (64 MiB payload), and `2^26` direct
convolution contributions. Gaussian offset squares must also fit the native
integer type; this further restricts wide one-axis kernels on 32-bit hosts.
The buffer limit includes the kernel, padded input,
coefficient vectors and numerical temporaries; it excludes existing histogram
storage, preceding region preparation, object/runtime overhead and rendering.
It is not a total allocation or process-memory bound. For example, a 200×200
grid admits scalar sigma 9 but rejects sigma 10 on work cost. Excessive requests
raise `ArgumentError` rather than changing the requested smoothing. Reduce sigma
or the number of bins, or select whole-bin regions. Empty/all-zero probability
requests and zero-mass histograms require no smoothing and accept any finite,
nonnegative sigma; sign and finiteness are still validated.
