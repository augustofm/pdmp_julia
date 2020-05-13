### My first Julia script. So exciting!
using PDSampler             #Main library
using LinearAlgebra         #For Cholesky
import Random               #For generating random vectors
using Plots                 #For plotting
using DelimitedFiles
using Distributions         #For mvg, Poisson...
#using LatexStrings          #For legends and labels
#using PGFPlotsX
#Plots.PGFPlotsBackend()

p=1
#normal to faces and intercepts
ns, a = Matrix{Float64}(I, p, p), zeros(p)
#ns, a = [[100 0];[100 0]], zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

Random.seed!(12)
#P1  = randn(p,p)
l=10
d1=Truncated(Normal(-3,1), -l, l)
d2=Truncated(Normal(3,1), -l, l)
x1=rand(d1, 10)
x2=rand(d2, 10)

x=vcat(x1,x2)#rand(Normal(-6,0.8),20), rand(Normal(2,0.8),10))

p=length(x)
# Boundaries
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

# Dist
kx(a,b,theta) = 0.5*exp(-theta*abs(a-b))
function Kernelmatrix(theta)
    K = Matrix{Float64}(I, p, p)
    for i in 1:p
        for j in 1:p
            K[i,j]=kx(x[i],x[j],theta)
        end
    end
    K += norm(K)/100000*Matrix{Float64}(I,p,p)
    K
end
function Kernelvector(theta,xstar)
    k = Matrix{Float64}(I, 1, p)
    for j in 1:p
        k[1,j]=kx(x[j],xstar,theta)
    end
    k'
end

theta0 = 1
massmatrix = Kernelmatrix(theta0)

mu = zeros(p)#+[3.,3.]
#mu  = P1*P1bk*(zeros(p)+[3.,3.])
#mvg = MvGaussianCanon(mu, P1)
#mvg = MvGaussianStandard(mu,C1)
#massmatrix = Matrix{Float64}(I, p, p)
mvg = MvGaussianStandard(mu,massmatrix)

#exp(-x^2/2)-> x^2/2 -> x*I()

# Building a BPS Simulation

gradll(x) = log.(exp.(-((x.+3).^2)/2)+exp.(-((x.-3).^2)/2)) + massmatrix*x## = - P1*(x-mu) = - nablaU
nextev_boo(x, v) = nextevent_boo(gradll, mvg, x, v)

T    = 10.0   # length of path generated
lref = 0.1     # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
prior = MvNormal(zeros(p),massmatrix)
x0 = zeros(p).+randn(p)#Distributions.rand(prior,1)[:,1]
v0 = Distributions.rand(prior,1)[:,1]

# Define a simulation
algname = "BOOMERANG"
sim_boo = Simulation( x0, v0, T, nextev_boo, gradll,
                  nextbd, lref,  mass=massmatrix; maxsegments = 1000)
(path_boo, details_boo) = simulate(sim_boo)

samples=samplepath_boo(path_boo,[0.0:0.01:round(path_boo.ts[end]-1,digits=0);])

plot(samples[1,:],samples[18,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Cte Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    xlims = (-25,25),
    ylims = (-25,25),
    #fmt = :png,
    color=:black)
    scatter!(path_boo.xs[1,:],path_boo.xs[18,:],legend=false)

fbar = mean(samples,dims=2)
