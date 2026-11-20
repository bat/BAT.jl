# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATEnsembleMCMCPureRNGsExt

import BAT
import EnsembleMCMC
using PureRNGs: Philox4x64, Threefry4x64, subrng
using Random123: Philox4x, Threefry4x

if Base.pkgversion(EnsembleMCMC) >= v"0.0.6"
    # BAT stores the partition address in ctr4:ctr2 and the draw position in ctr1.
    BAT._ensemble_key(rng::Philox4x{UInt64,R}) where {R} =
        subrng(Philox4x64(BAT.rngpart_getseed(rng)), R, rng.ctr4, rng.ctr3, rng.ctr2)
    BAT._ensemble_key(rng::Threefry4x{UInt64,R}) where {R} =
        subrng(Threefry4x64(BAT.rngpart_getseed(rng)), R, rng.ctr4, rng.ctr3, rng.ctr2)
end

end
