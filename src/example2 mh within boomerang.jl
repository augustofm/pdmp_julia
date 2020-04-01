# Metropolis within PDMP

# Boo - MH
target(y) = exp(dot(y-mvg.mu,mvg.prec*(y-mvg.mu)))
gradll(x) = gradloglik(mvg, x)+massmatrix*x#inv(massmatrix)*x
nextev_boo(x, v) = nextevent_boomerang(gradll, mvg, x, v)
T    = 1000  # length of path generated
lref = 0.5#0.5#0.15#0.1    # rate of refreshment
lmh　= 1
epsilon=0.01
nmh=20
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0.2,0.2]
y0 = [0.2,0.2]
v0   = randn(p) # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG-MH"

# Define a simulation
sim_boomh = Simulation(x0, y0, v0, T, nextev_boo, target, gradll,
                  nextbd, lref, lmh, algname, epsilon, nmh; maxgradeval = 2000000)
(path_boo, details_boo) = simulate(sim_boo)

samples = samplepath_boo(path_boo,[0.0:0.1:round(path_boo.ts[end]-1,digits=0);])
plot(samples[1,:],samples[2,:])
    annotate!(2, -2, text(string("nsegments: ", details_boo["nsegments"],"\n",
    "nbounce: ", details_boo["nbounce"],"\n",
    "nrefresh: ", details_boo["nrefresh"]), :red, :right, 10))
        annotate!(-1.7, 2, text(string("path mean: ", round.(pathmean_boo(path_boo),digits=3)), :black, :right,7))
        #annotate!(-1.7, 2, text(string("correlation_total: ", round.(cor(path_boo.xs[1,:],path_boo.xs[2,:]),digits=3)), :black, :right,7))
