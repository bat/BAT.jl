# Optional Plots checks

From the repository root:

```sh
julia --project=test/optional/plots -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test/optional/plots test/optional/plots/runtests.jl
```

These tests exercise the legacy recipes, selected statistical overlays, retained
frequency-weight correction, rejected-sample scatter encoding and actual Plots
rendering. CI runs this optional environment with a job deadline on Julia 1.10
and current stable Julia. The core test environment does not require Plots.

Histogram-region checks cover unequal-cell geometry, full 1D endpoints and
horizontal fills, request/label/color association, and separately colored contour
layers derived from their displayed field. Saved GR output complements numerical
and final-series geometry assertions.
