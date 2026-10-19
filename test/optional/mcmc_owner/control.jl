# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run only as an isolated child of runtests.py. This control does not load BAT.
using Test
Base.exit_on_sigint(false) # This test process owns its application signal policy.

function control()
    sleep(0.01)
    try
        println("BAT_OWNER READY control")
        flush(stdout)
        sleep(30)
        error("the external driver did not interrupt the control")
    catch err
        @test err isa InterruptException
        println("BAT_OWNER RESULT control")
        flush(stdout)
    end
end

println("Julia ", VERSION, "; kernel=", Sys.KERNEL, "; machine=", Sys.MACHINE,
    "; threads=", Threads.nthreads(), "; main_pool=", Threads.threadpool())
control()
