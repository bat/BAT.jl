# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieNumerics
using BAT, Test, Makie, CairoMakie, LinearAlgebra
const EXT=Base.get_extension(BAT,:BATMakieExt)
const DIAG=BAT._default_makie_diagonal_config()
const PAIR=BAT._default_makie_triagonal_config()
primitives(x,w,r,c=DIAG)=EXT.compute_plotting_primitives(view(x,:,:),view(w,:),r,EXT.LiveRecipe(),EXT.LiveCell(),c)
cell(fig,i,j)=only(filter(x->x isa Axis,contents(fig[1,1][i,j])))

@testset "Makie numerical boundary regressions" begin
    @testset "Geometry retains scientific coordinates" begin
        d=[1e8-1 1e8 1e8+1;1e8-1 1e8+1 1e8]; w=ones(3)
        h=primitives(d[1:1,:],w,QuantileHist1D())
        @test length(unique(first.(h.stairs_data)))>3
        @test all(p->p[1] isa Float64,h.stairs_data)
        @test all(p->p[1] isa Float64,h.stairs_data)
        # The narrower legacy fixture no longer meets unbounded KDE's physical
        # lattice admission. Retain its rejection and test Float64 geometry on
        # a resolvable lattice at the same large coordinate offset.
        kde_data=reshape(1e8 .+ 16 .* [-1.,0.,1.],1,:)
        @test_throws ArgumentError primitives(d[1:1,:],w,KDE1D())
        @test_throws ArgumentError primitives(d[1:1,:],w,QuantileKDE1D())
        k=primitives(kde_data,w,KDE1D())
        @test first.(k.poly_points)==vcat(k.x,reverse(k.x))
        @test length(unique(first.(k.poly_points)))>3
        @test length(unique(Float32.(k.x))) < length(unique(k.x))
        q=primitives(kde_data,w,QuantileKDE1D())
        @test length(unique(first.(q.full_line)))>3
        @test all(poly->all(p->p[1] isa Float64,poly),q.polys)
        p=primitives(d[1:1,:],w,PDF1D())
        @test first.(p.poly_points)==vcat(p.x,reverse(p.x))
        c=primitives(d,w,Cov2D(),PAIR)
        @test length(unique(c.ellipse_points))>100
        # Independent eigenvalue oracle for covariance [2/3 1/3;1/3 2/3].
        lengths=sort(norm.(c.axes_segments .- Ref([1e8,1e8])))
        @test lengths≈[sqrt(1/3),sqrt(1/3),1.,1.] atol=3e-8
        transposed=EXT.compose_plotspecs(c,Cov2D(),PAIR;transposed=true)
        @test only(first(transposed).args)==[Point{2,Float64}(p[2],p[1]) for p in c.ellipse_points]
        tiny=primitives(reshape(1e-40.*[-2.,-1.,0.,1.,2.],1,:),ones(5),PDF1D(),merge(DIAG,(npoints_pdf=3,)))
        @test maximum(tiny.y)>floatmax(Float32)
        @test all(isfinite,tiny.y)
        @test all(p->all(isfinite,p),tiny.poly_points)
        trace=(x=vec(d[1,:]),y=vec(d[2,:]),recency=[0.,.5,1.],group_ranges=[(Int32(1),Int32(0),1:3)],all_chainids=Int32[1])
        for transpose in (false,true)
            specs=EXT.compose_plotspecs(trace,Trace2D(),PAIR;transposed=transpose)
            expected=transpose ? Point{2,Float64}.(trace.y,trace.x) : Point{2,Float64}.(trace.x,trace.y)
            @test first(first(specs).args)==expected
            @test length(unique(first(first(specs).args)))==3
        end
        # Empty/live output geometry types remain stable for ComputeGraph.
        for r in (QuantileHist1D(),KDE1D(),QuantileKDE1D(),PDF1D())
            @test typeof(primitives(zeros(1,0),Float64[],r))==typeof(primitives(kde_data,w,r))
        end
        @test typeof(primitives(zeros(2,0),Float64[],Cov2D(),PAIR))==typeof(c)
    end
    @testset "Zero-mass KDE has no density" begin
        d=[-1. 0. 1.;1. 0. -1.]
        for (r,x,c) in ((KDE1D(),d[1:1,:],DIAG),(QuantileKDE1D(),d[1:1,:],DIAG),(KDE2D(),d,PAIR),(QuantileKDE2D(),d,PAIR))
            zero=primitives(x,zeros(3),r,c)
            empty=primitives(zeros(size(x,1),0),Float64[],r,c)
            @test isequal(zero,empty)
            @test isempty(EXT.compose_plotspecs(zero,r,c))
            @test !isempty(EXT.compose_plotspecs(primitives(x,ones(3),r,c),r,c))
        end
    end
    @testset "Marker area is invariant to finite weight scaling" begin
        d=[1. 2. 3.;3. 2. 1.]
        for scale in (1.,5e307,1e-308)
            w=scale.*[1.,2.,3.]
            @test all(isfinite,w)
            p=primitives(d,w,Scatter2D(),PAIR)
            @test only(EXT.compose_plotspecs(p,Scatter2D(),PAIR)).kwargs[:markersize]≈sqrt.([.5,1.,1.5]).*PAIR.markersize
            chain=EXT.compute_plotting_primitives(view(d,:,:),view(w,:),Int32[1,1,1],ChainScatter2D(),EXT.LiveRecipe(),EXT.LiveCell(),PAIR)
            @test only(EXT.compose_plotspecs(chain,ChainScatter2D(),PAIR)).kwargs[:markersize]≈sqrt.([.5,1.,1.5]).*PAIR.markersize
        end
        @test EXT._scatter_markersizes([nextfloat(0.),0.,0.],PAIR.markersize)≈[sqrt(3)*PAIR.markersize,0.,0.]
        @test EXT._scatter_markersizes([-3.,-2.,1.],PAIR.markersize)==PAIR.markersize
        @test EXT._scatter_markersizes([-1.,0.,1.],PAIR.markersize)==PAIR.markersize
        @test_throws DomainError EXT._scatter_markersizes([-1.,2.,3.],PAIR.markersize)
        @test iszero(EXT._scatter_markersizes(zeros(3),PAIR.markersize))
        for w in (Float64[],ones(3))
            @test EXT._scatter_markersizes(w,PAIR.markersize)==PAIR.markersize
        end
    end
    @testset "Public PDF and covariance rendering" begin
        samples=DensitySampleVector([[x] for x in 1e-40.*[-2.,-1.,0.,1.,2.]],zeros(5))
        fig=bat_makie_plot(samples,(diagonal=PDF1D,upper=Hist2D,lower=Hist2D),[1],1;display=false)
        try
            axis=cell(fig,1,1)
            poly=only(filter(p->p isa Makie.Poly,axis.scene.plots))
            # The actual public plot retains the full-precision converted input.
            @test all(p->all(isfinite,p),poly[1][])
            @test maximum(p->p[2],poly[1][])>floatmax(Float32)
            mktempdir() do dir
                file=joinpath(dir,"tiny-pdf.png")
                save(file,fig)
                @test filesize(file)>0
            end
        finally
            empty!(fig)
        end
        d=[1e8-1 1e8 1e8+1;1e8-1 1e8+1 1e8]
        samples=DensitySampleVector([collect(v) for v in eachcol(d)],zeros(3))
        fig=bat_makie_plot(samples,(diagonal=QuantileHist1D,upper=Scatter2D,lower=Scatter2D),[1,2],2;display=false)
        try
            first(filter(x->x isa Toggle,fig.content)).active[]=true
            ellipse=only(filter(p->p isa Makie.Lines,cell(fig,1,2).scene.plots))[1][]
            @test length(ellipse)==200
            @test length(unique(ellipse))>100
            @test maximum(p->p[1],ellipse)-minimum(p->p[1],ellipse)>1
            mktempdir() do dir
                file=joinpath(dir,"offset-covariance.png")
                save(file,fig)
                @test filesize(file)>0
            end
        finally
            empty!(fig)
        end
    end
end
end
