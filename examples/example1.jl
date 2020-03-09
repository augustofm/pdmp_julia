### My first Julia script. So exciting!
using PDSampler
using LinearAlgebra
import Random
using Plots
using path.jl

include("PDSampler.jl")

p=2
#normal to faces and intercepts
ns, a = Matrix{Float64}(I, p, p), zeros(p)
geom = Polygonal(ns,a)

nextbd(x,v) = nextboundary(geom, x, v)

Random.seed!(12)
P1  = randn(p,p)
P1 *= P1'
P1 += norm(P1)/100*Matrix{Float64}(I,p,p)
C1  = inv(P1); C1 += C1'; C1/=2;
L1  = cholesky(C1)

# figure out how to add 1 to the array
mu  = zeros(p)+[1.,1.]
mvg = MvGaussianCanon(mu, P1)

gradll(x) = gradloglik(mvg, x)

nextev(x, v) = nextevent_bps(mvg, x, v)

T    = 1000.0   # length of path generated
lref = 2.0      # rate of refreshment
x0   = mu+L1.L*randn(p) # sensible starting point
v0   = randn(p) # starting velocity
v0  /= norm(v0) # put it on the sphere (not necessary)
# Define a simulation
sim = Simulation( x0, v0, T, nextev, gradll,
                  nextbd, lref ; maxgradeval = 10000)

(path, details) = simulate(sim)

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

using Plots
plot(path)
