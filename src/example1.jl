### My first Julia script. So exciting!
using PDSampler
using LinearAlgebra
import Random
using Plots
using DelimitedFiles
using Distributions

p=2
#normal to faces and intercepts
ns, a = Matrix{Float64}(I, p, p), [-10,-10]#zeros(p)
#ns, a = [[100 0];[100 0]], zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

Random.seed!(12)
#P1  = randn(p,p)
P1 = [[1.06 -0.26];[-0.26 1.06]]
#1 = [[1.0 0];[0 1.0]]

#P1bk = copy(P1)
P1 *= P1'
#P1 = P1bk+Matrix{Float64}(I, p, p)
P1 += norm(P1)/100*Matrix{Float64}(I,p,p)
C1  = inv(P1); C1 += C1'; C1/=2;
L1  = cholesky(C1)
# figure out how to add 1 to the array
mu = zeros(p)#+[3.,3.]
#mu  = P1*P1bk*(zeros(p)+[3.,3.])
mvg = MvGaussianCanon(mu, P1)

#a = sqrt(P1)


# Building a BPS Simulation

gradll(x) = gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU
nextev_bps(x, v) = nextevent_bps(mvg, x, v)

T    = 1000.0   # length of path generated
lref = 1     # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0,0]
v0   = randn(p) # starting velocity
v0  /= norm(v0) # put it on the sphere (not necessary)
# Define a simulation
sim_bps = Simulation( x0, v0, T, nextev_bps, gradll,
                  nextbd, lref; maxgradeval = 2000)
(path_bps, details_bps) = simulate(sim_bps)

plot(path_bps.xs[1,:],path_bps.xs[2,:])
annotate!(4, 4, text(string("nsegments: ", details_bps["nsegments"],"\n",
    "nbounce: ", details_bps["nbounce"],"\n",
    "nrefresh: ", details_bps["nrefresh"]), :red, :right, 10))
    annotate!(4, 7.5, text(string("path mean: ", round.(pathmean(path_bps),digits=3)), :black, :right,7))
cor(path_bps.xs[2,:], path_bps.xs[1,:])
details_bps
pathmean(path_bps)





# Building a Zig Zag Simulation

gradll(x) = gradloglik(mvg, x)
nextev_zz(x, v) = nextevent_zz(mvg, x, v)

T    = 1000.0   # length of path generated
lref = 1#1.0     # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0,0]
v0   = rand([-1,1], p) # starting velocity
v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "ZZ"
# Define a simulation
sim_zz = Simulation( x0, v0, T, nextev_zz, gradll,
                  nextbd, lref, algname; maxgradeval = 200000)
(path_zz, details_zz) = simulate(sim_zz)

plot(path_zz.xs[1,:],path_zz.xs[2,:])
annotate!(4, 9, text(string("nsegments: ", details_zz["nsegments"],"\n",
    "nbounce: ", details_zz["nbounce"],"\n",
    "nrefresh: ", details_zz["nrefresh"]), :red, :right, 10))
    annotate!(4, 7.5, text(string("path mean: ", round.(pathmean(path_zz),digits=3)), :black, :right,7))
cor(path_bps.xs[2,:], path_bps.xs[1,:])

pathmean(path_zz)
details_zz


# Building a Boomerang Simulation with bounces and refreshment

#massmatrix = [[1 0];[0 1]]
massmatrix = P1
gradll(x) = gradloglik(mvg, x)+massmatrix*x#inv(massmatrix)*x
nextev_boo(x, v) = nextevent_boomerang(gradll, mvg, x, v)
T    = 1000.0   # length of path generated
lref = 0.5#0.5#0.15#0.1    # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0,0]
v0   = randn(p) # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG"

# Define a simulation
sim_boo = Simulation(x0, v0, T, nextev_boo, gradll,
                  nextbd, lref, algname, mass=massmatrix; maxgradeval = 2000000)
(path_boo, details_boo) = simulate(sim_boo)

plot(path_boo.xs[1,:],path_boo.xs[2,:])
    annotate!(2, -2, text(string("nsegments: ", details_boo["nsegments"],"\n",
    "nbounce: ", details_boo["nbounce"],"\n",
    "nrefresh: ", details_boo["nrefresh"]), :red, :right, 10))
        annotate!(-1.7, 2.3, text(string("path mean: ", round.(pathmean(path_boo),digits=3)), :black, :right,7))
        annotate!(-1.7, 2, text(string("correlation_total: ", round.(cor(path_boo.xs[1,:],path_boo.xs[2,:]),digits=3)), :black, :right,7))
        annotate!(-1.7, 1.7, text(string("correlation_corners: ", round.(cor(path_boo.xs[1,findall(path_boo.is_jump)],path_boo.xs[2,findall(path_boo.is_jump)]),digits=3)), :black, :right,7))


pathmean(path_boo)
details_boo


# Building a Boomerang Simulation with NO bounces, but with  refreshment

gradll(x) = gradloglik(mvg, x)+x#inv(massmatrix)*x
nextev_boo(x, v) = nextevent_boomerang(gradll, mvg, x, v)
T    = 1000.0   # length of path generated
lref = 0.5#0.15#0.1    # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0,0]
v0   = randn(p) # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG"

# Define a simulation
sim_boo2 = Simulation(x0, v0, T, nextev_boo, gradll,
                  nextbd, lref, algname; maxgradeval = 2000000)
(path_boo2, details_boo2) = simulate(sim_boo2)

plot(path_boo2.xs[1,:],path_boo2.xs[2,:])
    annotate!(2, -2, text(string("nsegments: ", details_boo2["nsegments"],"\n",
        "nbounce: ", details_boo2["nbounce"],"\n",
        "nrefresh: ", details_boo2["nrefresh"]), :red, :right, 10))
        annotate!(-1.7, 2.3, text(string("path mean: ", round.(pathmean(path_boo2),digits=3)), :black, :right,7))
        annotate!(-1.7, 2, text(string("correlation_total: ", round.(cor(path_boo2.xs[1,:],path_boo2.xs[2,:]),digits=3)), :black, :right,7))
        annotate!(-1.7, 1.7, text(string("correlation_corners: ", round.(cor(path_boo2.xs[1,findall(path_boo2.is_jump)],path_boo2.xs[2,findall(path_boo2.is_jump)]),digits=3)), :black, :right,7))



gradll(x) = gradloglik(mvg, x)

# Building a basic MC estimator
# (taking samples from 2D MVG that are in positive orthan)
sN = 1000
s  = broadcast(+, mu, L1.L*randn(p,sN))
mt = zeros(2)
np = 0
# Sum for all samples in the positive orthan
ss = [s; ones(sN)']
mt = sum(ss[:,i] for i in 1:sN if !any(e->e<0, ss[1:p,i]))
mt = mt[1:p]/mt[end]



bounce = nevtakes2 ? sim.nextevent(x, v) : sim.nextevent(x, v, tauref)

r = sin(tau_boo)^2
x .+=sqrt(r)*v+sqrt(1-r)*x
v .+=-sqrt(r)*x+sqrt(1-r)*v

xₜ(x) = x+sin(t)*v+cos(t)*x
yₜ(x) = v-sin(t)*x+cos(t)*v

plot(xₜ, yₜ, x0 , x)
plot(x0,x)
