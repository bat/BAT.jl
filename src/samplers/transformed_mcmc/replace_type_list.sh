# find . -name \*.jl -execdir sh -c
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCInitAlgorithm\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCTuningAlgorithm\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCBurninAlgorithm\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCIterator\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(AbstractMCMCTunerInstance\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCProposal\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(SampleID\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(AbstractMCMCStats\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(AbstractMCMCWeightingScheme\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(AbstractProposalDist\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(ProposalDistSpec\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCTempering\)/ Transformed\1/g" {} \;
find . -name \*.jl -execdir sed -i -e "s/ \(MCMCTemperingInstance\)/ Transformed\1/g" {} \;
