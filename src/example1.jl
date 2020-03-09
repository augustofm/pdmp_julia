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
#P1bk = copy(P1)
P1 *= P1'
#P1 = P1bk+Matrix{Float64}(I, p, p)
P1 += norm(P1)/100*Matrix{Float64}(I,p,p)
C1  = inv(P1); C1 += C1'; C1/=2;
L1  = cholesky(C1)
# figure out how to add 1 to the array
mu = zeros(p)+[3.,3.]
#mu  = P1*P1bk*(zeros(p)+[3.,3.])
mvg = MvGaussianCanon(mu, P1)



# Building a BPS Simulation

gradll(x) = gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU

nextev_bps(x, v) = nextevent_bps(mvg, x, v)

T    = 1000.0   # length of path generated
lref = 1.0      # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [-4,-4]
v0   = randn(p) # starting velocity
v0  /= norm(v0) # put it on the sphere (not necessary)
# Define a simulation
sim_bps = Simulation( x0, v0, T, nextev_bps, gradll,
                  nextbd, lref; maxgradeval = 10000)
(path_bps, details_bps) = simulate(sim_bps)

plot(path_bps.xs[1,:],path_bps.xs[2,:])
details_bps
pathmean(path_bps)







# Building a Zig Zag Simulation

gradll(x) = gradloglik(mvg, x)
nextev_zz(x, v) = nextevent_zz(mvg, x, v)

T    = 1000.0   # length of path generated
lref = 0#1.0     # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [-4,-4]
v0   = rand([-1,1], p) # starting velocity
v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "ZZ"
# Define a simulation
sim_zz = Simulation( x0, v0, T, nextev_zz, gradll,
                  nextbd, lref, algname; maxgradeval = 10000)
(path_zz, details_zz) = simulate(sim_zz)

plot(path_zz.xs[1,:],path_zz.xs[2,:])
pathmean(path_zz)


# Building a Boomerang

gradll(x) = gradloglik(mvg, x)+x
nextev_boo(x, v) = nextevent_boomerang(mvg, x, v)
T    = 1000.0   # length of path generated
lref = 0.1    # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [10,　10]
v0   = randn(p) # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG"
# Define a simulation
sim_boo = Simulation( x0, v0, T, nextev_boo, gradll,
                  nextbd, lref, algname; maxgradeval = 10000)
(path_boo, details_boo) = simulate(sim_boo)

plot(path_boo.xs[1,:],path_boo.xs[2,:])
pathmean(path_boo)
details_boo


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
