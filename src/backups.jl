@time begin
    lcnt += 1
    println(t)
    #println("x: ",x, "v: ",v)
    # Default for trajectories. False only when there's no bounce and the point registered is in the
    # halmitonian trajectory

    #jumping_event = true
    # simulate first arrival from IPP
#    bounce = nevtakes2 ? sim.nextevent(x, v) : sim.nextevent(x, v, tauref)
    @time begin
        if sim.MHsampler
            if sim.algname ∈ ["BPS","ZZ","CS"]
                bounce = nevtakes2 ? sim.nextevent(x, y, v,invmass) : sim.nextevent(x, y, v, invmass, tauref) #add y later
            else
                bounce = nevtakes2 ? sim.nextevent(x, y, v) : sim.nextevent(x, y, v, tauref) #add y later
            end
            tau = min(bounce.tau, taubd, tauref, taumh)
        else
            bounce = nevtakes2 ? sim.nextevent(x, v) : sim.nextevent(x, v, tauref) # remove y later
            tau = min(bounce.tau, taubd, tauref)
        end
    end
# find next event (unconstrained case taubd=NaN (ignored))
    #tau = min(bounce.tau, taubd, tauref, taumh)
    # standard bounce
@time begin
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
        if sim.MHsampler
            # exploiting the memoryless property
            taumh -= tau
            if sim.algname ∈ ["BPS","ZZ","CS"]
                g = sim.gll(x,invmass)#bounce = nevtakes2 ? sim.nextevent(x, y, v,invmass) : sim.nextevent(x, y, v, invmass, tauref) #add y later
            else
                g = sim.gll(x,y)
            end
        else
            g = sim.gll(x)
        end
#            println("gradll: ", max.(0,(-g.*v)[bounce.flipindex]))
        if bounce.dobounce(g, v) # e.g.: thinning, acc/rej
            # if accept

            nbounce += 1
            # updating position
            if sim.algname == "BOOMERANG"
                # if a mass matrix is provided
                if length(mass) > 0
#                        vaux = reflect_boo!(g, v, mass)
                #    println("gradd: ",g)
                #    println("pos: ",x)
                #    println("vel: ",v)
                    vdelta=2.0dot(g, v) * (mass * g) / dot(mass * g, g)
                #    println("Bouncing occurs: ",vdelta)
                    v.-=vdelta
                    # standard Boomerang bounce
                else
                    v = reflect_boo!(g, v)
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
            elseif sim.algname == "CS"
                v = reflect_cs!(g, v)
            end
            # There is no bouncing for CS, only refreshment
        else
            #jumping_event = false
            # move closer to the boundary/refreshment time
            taubd -= tau
            # we don't need to record when rejecting when trajectoris are linear
            #if sim.algname != "BOOMERANG"
        #        continue
            #end
        end
    # hard bounce against boundary
end
@time begin
    if tau == taubd
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
        if sim.MHsampler
            taumh -= tau
        end

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
                v = reflect_boo!(normalbd, v, mass)
            else
                v = reflect_boo!(normalbd, v)
            end
        end
        # Implement something for CS later

    # metropolis hastings
    elseif tau == taumh
        t += tau
        nmetropolis  += 1

        for iter in 1:sim.nmh
            #x_atjump = sin(tau)*v+cos(tau)*x
            if sim.algname == "BOOMERANG"
                xbk = copy(x)
                x = sin(tau)*v+cos(tau)*x
                v = -sin(tau)*xbk+cos(tau)*v
            else
                x += tau*v
            end
            ydash = y+randn(d2) * sim.epsilon

            if rand() < min(1.0, (sim.ytarget(x,ydash) ./ sim.ytarget(x,y))[1])
                y = copy(ydash)
                naccepted += 1
                # Storing the accepted values during L steps for backup
                ysfull[((j-1) * d2 + 1):(j * d2)] = y
                ys_acc_rate[j] = naccepted/(naccepted+nrejected)
                j += 1

            else
                nrejected += 1
            end
            iter += 1
        end
        mass = sim.Sigmay(y) #add y later
        if sim.algname ∈ ["BPS","ZZ"]
                invmass=inv(mass)
        end

        # update taumh
        taumh = Random.randexp()/lambdamh
        # exploiting the memoryless property
        tauref -= tau
        #v = sim.refresh!(v)
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
            if sim.algname ∈ ["BPS", "GBPS","ZZ","CS"]
                x += tau*v
            else
                x = sin(tau)*v+cos(tau)*x
            end
            # ---- REFRESH ----
            if sim.algname=="ZZ"
                v  .= rand([-1,1], length(v)).*multaux
                #v  /= norm(v)
                #temp
            elseif sim.algname=="CS"
                w = zeros(d)
                w[rand(1:d,1)]=rand([-1,1],1)
                v  .= 7*w#.*multaux
            elseif sim.algname ∈ ["BPS","BOOMERANG"]
                #v = refresh_boo!(v,mass)
                if length(mass) > 0
                #    initial = Distributions.MvNormal([0.545,.028,.803,0.024],mass)
                @time begin
                    initial = Distributions.MvNormal(zeros(d),mass)
                    v = Distributions.rand(initial,1)[:,1]
                end
                else
                    initial = Distributions.MvNormal(zeros(d),Matrix{Float64}(I, d, d))
                #    initial = Distributions.MvNormal([1,.05,1,.05],[0.5, 0.05, 0.5, 0.05].*Matrix{Float64}(I, d, d))
                #    initial = Distributions.MvNormal(zeros(d),mass)
                    v = Distributions.rand(initial,1)[:,1]
                end
            else
                v = sim.refresh!(v)
            end
            t += tau
            # update tauref
            tauref = Random.randexp()/lambdaref
        end
    end
    @time begin
        # check when the next boundary hit will occur
        #(taubd, normalbd) = sim.nextboundary(x, v)
        taubd = Inf
        # increment the counter for the number of segments
    end
    i += 1

    # Increase storage on a per-need basis.
    if mod(i,blocksize)==0
        resize!(xs, length(xs) + d * blocksize)
        if sim.MHsampler
            resize!(ys, length(ys) + d2 * blocksize)
        end
        resize!(ts, length(ts) + blocksize)
        #resize!(is_jump, length(is_jump) + blocksize)
    end


    # Storing path times
    ts[i] = t
     # storing columns vertically, a vector is simpler/cheaper to resize
    xs[((i-1) * d + 1):(i * d)] = x
    if sim.MHsampler
        ys[((i-1) * d2 + 1):(i * d2)] = y
        if mod(j,blocksize)==0
            resize!(ysfull, length(ysfull) + d2 * blocksize)
            resize!(ys_acc_rate, length(ys_acc_rate) + blocksize)
        end
    end
    # storing if it's a corner or just part of the hamiltonian path
    #is_jump[i] = jumping_event

    # Safety checks to break long loops every 100 iterations
    if mod(lcnt, 20) == 0
        if  (time() - start) > sim.maxsimtime
            println("Max simulation time reached. Stopping")
        #    break
        end
        if  i > sim.maxsegments
            println("Too many segments generated. Stopping")
        #    break
        end
    end
end #End Time count
