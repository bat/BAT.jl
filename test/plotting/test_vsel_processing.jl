

@testset "vsel processing" begin 

    vs = NamedTupleShape(a = ArrayShape{Real}(3,3,3), 
                         b = ArrayShape{Real}(7), 
                         c = ArrayShape{Real}(2), 
                         d = ScalarShape{Real}(), 
                         e = ScalarShape{Real}(), 
                         f = ScalarShape{Real}(), 
                         g = ScalarShape{Real}(), 
                         h = ArrayShape{Real}(2,2)
                         )

    logd = fill(NaN, 2)

    flat_v = nestedview(rand(44,2))

    unshaped_samples = DensitySampleVector(flat_v, logd)

    shaped_samples = vs.(unshaped_samples)

    testinputs_s =    [ [1,2,3], 
                        :([1,2,3]),
                        [2:5], 
                        :([2:5]),
                        [1:3, 5:6, 8], 
                        :([1:3, 5:6, 8]),         
                        2,  
                        :c,             
                        :(a[1,2]),      
                        :(b[2:4]),      
                        :([a[1,2, :], :d]), 
                        [:(a[1,2, :]), :b] 
                    ]

    testinputs_u =    [ [1,2,3],
                        [2:5],          
                        2 
                    ]              

    norm_correct_s  = [  [:a, :b, :c],
                         [:a, :b, :c],
                         [:b, :c, :d, :e],
                         [:b, :c, :d, :e],
                         [:a, :b, :c, :e, :f, :h],
                         [:a, :b, :c, :e, :f, :h],
                         [:b],
                         [:c],
                         [:(a[1,2])],
                         [:(b[2:4])],
                         [:(a[1, 2, :]), :d],
                         [:(a[1, 2, :]), :b]
                    ]


    norm_correct_u  = [  [1,2,3],
                         [2:5],          
                         [2] 
                    ] 

    
                        
    test_vsel_s     = [ [:(a[1]), :b, :c],
                        [:(a[1]), :(a[2:7])],
                        [:(a[1,2,3]), :(a[1,1,:]), :(a[1,:,:])] 
                    ]
    
    test_vsel_u   =   [ [1,2,3],
                        [2:5] 
                    ]



    shaped_marg_idxs_correct = [   [1, 28, 29, 30, 31, 32, 33, 34, 35, 36],
                                   [1, 2, 3, 4, 5, 6, 7],
                                   [22, 1, 10, 19, 1, 4, 7, 10, 13, 16, 19, 22, 25]
    ]
    #=
    shaped_mapped_correct =   [ OrderedDict(  :a⌞1⌟ => [1],
                                              :b    => [28, 29, 30, 31, 32, 33, 34],
                                              :c    => [35, 36]   ),

                                OrderedDict(  :a⌞1⌟   => [1],
                                              :a⌞2ː7⌟ => [2, 3, 4, 5, 6, 7]  ), 

                                OrderedDict(  :a⌞1ˌ2ˌ3⌟ => [22],
                                              :a⌞1ˌ1ˌː⌟ => [1, 10, 19],
                                              :a⌞1ˌːˌː⌟ => [1, 4, 7, 10, 13, 16, 19, 22, 25])
                            ]     
                              
                              

    unshaped_mapped_correct = [ OrderedDict(  :v⌞1⌟ => [1],
                                              :v⌞2⌟ => [2],
                                              :v⌞3⌟ => [3]   ),

                                OrderedDict(  :v⌞2ː5⌟ => [2, 3, 4, 5] ) 
                            ]                           
    =#                     
    @testset "normalize" begin
        
        for (i, el) in enumerate(testinputs_s)
        
            @test _normalize_vsel_shaped(vs, el) == norm_correct_s[i]
    
        end
        
        for (i, el) in enumerate(testinputs_u)
        
            @test _normalize_vsel_unshaped(el) == norm_correct_u[i]
    
        end

    end

    @testset "marg_idxs" begin
        
        for (i, el) in enumerate(test_vsel_s)
            
            @test marg_idxs_shaped(shaped_samples, el)[1] == shaped_marg_idxs_correct[i]

        end
        
    end
end
