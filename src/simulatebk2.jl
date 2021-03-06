export
    Simulation,
    simulate


"""
    Simulation

Describe a PDSampling: information about the initial point, the time of the
simulation, the function to sample from an IPP, etc.
"""
struct Simulation
    x0::Vector{Real}       # Starting point
    y0::Vector{Real}
    v0::Vector{Real}       # Starting velocity
    T::Real                # Simulation time
    nextevent::Function    # Appropriate simulation for first arrival time
    p::Function            # Target function for MH acceptance/rejection
    gll::Function          # Gradient of Log Lik (potentially CV)
    nextboundary::Function # Where/When is the next boundary hit
    lambdaref::Real        # Refreshment rate
    lambdamh::Real         # MH update rate
    algname::String        # BPS, ZZ, GBPS, BOOMERANG
    epsilon::Real           # For Metropolis Hastings
    nmh::Int                # Number of iterations for each MH step
    # derived
    dim_x::Int               # dimensionality
    dim_y::Int
    # optional named arguments
    mass::Matrix{Real}     # mass matrix (preconditioner)
    blocksize::Int         # increment the storage by blocks
    maxsimtime::Real       # max. simulation time (s)
    maxsegments::Int       # max. num. of segments
    maxgradeval::Int       # max. num. grad. evals
    refresh!::Function     # refreshment function (TODO: examples)

    # constructor
    function Simulation(x0, y0, v0, T, nextevent, p, gradloglik, nextboundary,
                lambdaref=1.0, lambdamh = 1.0, algname="BOOMERANG", epsilon, nmh;
                mass=diagm(0=>ones(0)), blocksize=1000, maxsimtime=4e3,
                maxsegments=1_000_000, maxgradeval=100_000_000,
                refresh! = refresh_global! )
        # check none of the default named arguments went through
        @assert !(x0 == :undefined || y0 == :undefined || v0 == :undefined || T == :undefined ||
                  nextevent == :undefined || gradloglik == :undefined || nextboundary == :undefined) "Essential arguments undefined"
#        @assert !(algname=="BOOMERANG-MH" & y0 == :undefined) "Essential MH arguments undefined"
        # basic check to see that things are reasonable
        @assert length(x0) == length(v0) > 0 "Inconsistent arguments"
        @assert T > 0.0 "Simulation time must be positive"
        @assert lambdaref >= 0.0 "Refreshment rate must be >= 0"
        @assert lambdamh >= 0.0 "MH rate must be >= 0"
        ALGNAME = uppercase(algname)
        @assert (ALGNAME ∈ ["BPS", "ZZ","GBPS","BOOMERANG"]) "Unknown algorithm <$algname>"
        new( x0, y0, v0, T,
             nextevent, p, gradloglik, nextboundary,
             lambdaref, lambdamh, ALGNAME, epsilon, nmh, length(x0), length(y0),
             mass, blocksize, maxsimtime,
             maxsegments, maxgradeval, refresh! )
    end
end
# Constructor with named arguments
function Simulation(;
            x0 = :undefined,
            y0 = :undefined,
            v0 = :undefined,
            T = :undefined,
            nextevent = :undefined,
            target = :undefined,
            gradloglik = :undefined,
            nextboundary = :undefined,
            lambdaref = 1.0,
            lambdamh = 1.0,
            algname = "BOOMERANG",
            epsilon = 0.01,
            nmh = 1,
            mass = diagm(0=>ones(0)),
            blocksize = 1000,
            maxsimtime = 4e3,
            maxsegments = Int(1e6),
            maxgradeval = Int(1e8),
            refresh! = refresh_global! )
    # calling the unnamed constructor
    Simulation( x0, y0, v0, T,
                nextevent,
                p,
                gradloglik,
                nextboundary,
                lambdaref,
                lambdamh,
                algname,
                epsilon,
                nmh;
                mass = mass,
                blocksize = blocksize,
                maxsimtime = maxsimtime,
                maxsegments = maxsegments,
                maxgradeval = maxgradeval,
                refresh! = refresh! )
end

"""
    simulate(sim)

Launch a PD simulation defined in the sim variable. Return the corresponding
Path and a dictionary of indicators (clocktime, ...).
"""
function simulate(sim::Simulation)
    # keep track of how much time we've been going for
    start = time()
    # dimensionality
    d1 = sim.dim_x
    d2 = sim.dim_y
    # initial states
    x, y, v = copy(sim.x0), copy(sim.y0), copy(sim.v0)



    # time counter, and segment counter
    t, i = 0.0, 1
    # counters for the number of effective loops and
    # for the number of evaluations of the gradient
    # this will be higher than the number of segments i
    lcnt, gradeval = 0, 0

    # storing by blocks of blocksize nodes at the time
    blocksize = sim.blocksize

    # storing xs as a single column for efficient resizing
    xs, ys, ts, is_jump = zeros(d1*blocksize), zeros(d2*blocksize), zeros(blocksize), falses(blocksize)
    # store initial point
    xs[1:d1] = x
    ys[1:d2] = y
    # mass matrix?
    mass = copy(sim.mass) # store it here as we may want to adapt it

    # check if nextevent takes 2 or 3 parameters
    nevtakes2 = (length(methods(sim.nextevent).ms[1].sig.parameters)-1) == 2

    # compute current reference metropolis hasting time
    lambdaref = sim.lambdamh
    taumh = (lambdamh>0.0) ? Random.randexp() / lambdamh : Inf

    # compute current reference bounce time
    lambdaref = sim.lambdaref
    tauref = (lambdaref>0.0) ? Random.randexp() / lambdaref : Inf

    # Compute time to next boundary + normal
    (taubd, normalbd) = sim.nextboundary(x, v)


    # keep track of how many refresh events
    nrefresh = 0
    # keep track of how many boundary hits
    nboundary = 0
    # keep track of how many standard bounces
    nbounce = 0

    while (t < sim.T) && (gradeval < sim.maxgradeval)
        # increment the counter to keep track of the number of effective loops
        lcnt += 1

        # Default for trajectories. False only when there's no bounce and the point registered is in the
        # halmitonian trajectory

        #jumping_event = true
        # simulate first arrival from IPP
    #    bounce = nevtakes2 ? sim.nextevent(x, v) : sim.nextevent(x, v, tauref)
        bounce = nevtakes2 ? sim.nextevent(x, y, v) : sim.nextevent(x, y, v, tauref)

    # find next event (unconstrained case taubd=NaN (ignored))
        tau = min(bounce.tau, taubd, tauref, taumh)
        # standard bounce
        if tau == bounce.tau
            # there will be an evaluation of the gradient
            gradeval += 1
            # updating time
            t += tau

            # updating position
            if sim.algname == "BOOMERANG"
                xbk = copy(x)
                x = sin(tau)*v+cos(tau)*x
                v = -sin(tau)*xbk+cos(tau)*v
            else
                x += tau*v
            end
            # exploiting the memoryless property
            tauref -= tau
            # ---- BOUNCE ----
            g = sim.gll(x,y)

            if bounce.dobounce(g, v) # e.g.: thinning, acc/rej
                # if accept
                nbounce += 1
                # updating position
                if sim.algname == "BOOMERANG"
                    # if a mass matrix is provided
                    if length(mass) > 0
                        v = reflect_boomerang!(g, v, mass)
                        # standard Boomerang bounce
                    else
                        v = reflect_boomerang!(g, v)
                    end
                elseif sim.algname == "BPS"
                    # if a mass matrix is provided
                    if length(mass) > 0
                        v = reflect_bps!(g, v, mass)
                        # standard BPS bounce
                    else
                    v = reflect_bps!(g,v)
                    end
                elseif sim.algname == "GBPS"
                    v = reflect_gbps(g, v)
                elseif sim.algname == "ZZ"
                    v = reflect_zz!(bounce.flipindex, v)
                end
            else
                #jumping_event = false
                # move closer to the boundary/refreshment time
                taubd -= tau
                # we don't need to record when rejecting when trajectoris are linear
                #if sim.algname != "BOOMERANG"
                    continue
                #end
            end
        # hard bounce against boundary
        elseif tau == taubd
            # CHANGE LATER!!!
            nboundary += 1
            # Record point epsilon from boundary for numerical stability

            if sim.algname == "BOOMERANG"
                x = sin(tau - 1e-10)*v+cos(tau)*x
            else
                x +=  (tau - 1e-10) * v
            end


            t += tau
            # exploiting the memoryless property
            tauref -= tau
            # ---- BOUNCE (boundary) ----
            if sim.algname ∈ ["BPS", "GBPS"]
                # Specular reflection (possibly with mass matrix)
                if length(mass) > 0
                    v = reflect_bps!(normalbd, v, mass)
                else
                    v = reflect_bps!(normalbd,v)
                end
            elseif sim.algname == "ZZ"
                v = reflect_zz!(findall((v.*normalbd).<0.0), v)
            else
                # Specular reflection (possibly with mass matrix)
                if length(mass) > 0
                    v = reflect_boomerang!(normalbd, v, mass)
                else
                    v = reflect_boomerang!(normalbd, v)
                end
            end
        # metropolis hastings
        elseif tau == taumh
            t += tau
            for 1:sim.nmh
                ydash = y+randn(d2) * sim.epsilon
                if rand() < min(1.0, sim.p(ydash) / sim.p(y))
                    y = copy(y_dash)
                end
            end
        #    v = sim.refresh!(v)
        # random refreshment
        else
            #= to be in this part, lambdaref should be greater than 0.0
            because if lambdaref=0 then tauref=Inf. There may be a weird
            corner case in which an infinity filters through which we would
            skip =#
            if !isinf(tau)
                #
                nrefresh += 1
                #
                if sim.algname ∈ ["BPS", "GBPS","ZZ"]
                    x += tau*v
                else
                    x = sin(tau)*v+cos(tau)*x
                end
                # ---- REFRESH ----
                if sim.algname=="ZZ"
                    v  .= rand([-1,1], length(v))
                    v  /= norm(v)
                else
                    v = sim.refresh!(v)
                end
                t += tau
                # update tauref
                tauref = Random.randexp()/lambdaref
            end
        end
        # check when the next boundary hit will occur
        (taubd, normalbd) = sim.nextboundary(x, y, v)
        # increment the counter for the number of segments
        i += 1

        # Increase storage on a per-need basis.
        if mod(i,blocksize)==0
            resize!(xs, length(xs) + d1 * blocksize)
            resize!(ys, length(ys) + d2 * blocksize)
            resize!(ts, length(ts) + blocksize)
            resize!(is_jump, length(is_jump) + blocksize)
        end

        # Storing path times
        ts[i] = t
         # storing columns vertically, a vector is simpler/cheaper to resize
        xs[((i-1) * d1 + 1):(i * d1)] = x
        ys[((i-1) * d2 + 1):(i * d2)] = y
        # storing if it's a corner or just part of the hamiltonian path
        #is_jump[i] = jumping_event

        # Safety checks to break long loops every 100 iterations
        if mod(lcnt, 100) == 0
            if  (time() - start) > sim.maxsimtime
                println("Max simulation time reached. Stopping")
                break
            end
            if  i > sim.maxsegments
                println("Too many segments generated. Stopping")
                break
            end
        end
    end # End of while loop

    details = Dict(
        "clocktime" => time()-start,
        "ngradeval" => gradeval,
        "nloops"    => lcnt,
        "nsegments" => i,
        "nbounce"   => nbounce,
        "nboundary" => nboundary,
        "nrefresh"  => nrefresh
    )

#    (Path(reshape(xs[1:(i*d)], (d,i)), ts[1:i], is_jump[1:i]), details)
    (Path(reshape(xs[1:(i*d1)], (d1,i)), reshape(y[1:(i*d2)], (d2,i)), ts[1:i]), details)

end
