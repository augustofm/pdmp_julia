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

p=2
#normal to faces and intercepts
ns, a = Matrix{Float64}(I, p, p), [-Inf,Inf]#zeros(p)
#ns, a = [[100 0];[100 0]], zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

Random.seed!(12)
#P1  = randn(p,p)
#P1 = [[1.06 -0.26];[-0.26 1.06]]
#P1 = Matrix{Float64}(I, p, p)
P1 = [[1.0 -0.2];[-0.2 1.0]]

#P1bk = copy(P1)
P1 *= P1'
#P1 = P1bk+Matrix{Float64}(I, p, p)
P1 += norm(P1)/100*Matrix{Float64}(I,p,p)
C1  = inv(P1); C1 += C1'; C1/=2;
L1  = cholesky(C1)
# figure out how to add 1 to the array
mu = zeros(p)#+[3.,3.]
#mu  = P1*P1bk*(zeros(p)+[3.,3.])
#mvg = MvGaussianCanon(mu, P1)
mvg = MvGaussianStandard(mu,C1)
massmatrix = Matrix{Float64}(I, p, p)

#a = sqrt(P1)


# Building a BPS Simulation

gradll(x) = gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU
nextev_bps(x, v) = nextevent_bps(mvg, x, v)

T    = 900.0   # length of path generated
lref = 0.05     # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0.1,0.1]#,0.1,0.1,0.1]
v0 = [0.1,0.1]#,0.1,0.1,0.1]
#v0   = randn(p) # starting velocity
v0  /= norm(v0) # put it on the sphere (not necessary)
# Define a simulation
algname = "BPS"
sim_bps = Simulation( x0, v0, T, nextev_bps, gradll,
                  nextbd, lref,  mass=massmatrix; maxsegments = 100)
(path_bps, details_bps) = simulate(sim_bps)

#using LaTeXStrings
#Plots.scalefontsizes(0.8)
plot(path_bps.xs[1,:],path_bps.xs[2,:],
    legend=false,
    size=(600,600),
    #title="Bouncy Particle Sampler (2-dimensional Gaussian target)",
    xlabel = "X1(t)",
    ylabel = "X2(t)",
    ylims = (-4,4),
    xlims = (-4,4),
    color=:black)
    scatter!(path_bps.xs[1,:],path_bps.xs[2,:],legend=false)
    annotate!(4, -3, text(string("nsegments: ", details_bps["nsegments"],"\n",
    "nbounce: ", details_bps["nbounce"],"\n",
    "nrefresh: ", details_bps["nrefresh"]), :black, :right, 16))
    #annotate!(3, 4, text(string("path mean: ", round.(pathmean(path_bps),digits=3)), :black, :right,10))
savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/figure2_bps2")
#cor(path_bps.xs[2,:], path_bps.xs[1,:])
details_bps


# Building a Zig Zag Simulation

gradll(x) = gradloglik(mvg, x)
nextev_zz(x, v) = nextevent_zz(mvg, x, v)

T    = 900.0   # length of path generated
lref = 0.05#1.0     # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
#x0 = [0.1,0.1]#,0.1,0.1,0.1]
v0   = rand([-1,1], p) # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "ZZ"
# Define a simulation
sim_zz = Simulation( x0, v0, T, nextev_zz, gradll,
                  nextbd, lref, algname; maxgradeval = 500000, maxsegments=100)
(path_zz, details_zz) = simulate(sim_zz)

plot(path_zz.xs[1,:],path_zz.xs[2,:],
    legend=false,
    size=(600,600),
    #title="Zig-Zag Sampler (2-dimensional Gaussian target)",
    xlabel = "X1(t)",
    ylabel = "X2(t)",
    ylims = (-4,4),
    xlims = (-4,4),
    color=:black)
    scatter!(path_zz.xs[1,:],path_zz.xs[2,:],legend=false)
    annotate!(4, -3, text(string("nsegments: ", details_zz["nsegments"],"\n",
    "nbounce: ", details_zz["nbounce"],"\n",
    "nrefresh: ", details_zz["nrefresh"]), :black, :right, 16))
    #annotate!(3, 4, text(string("path mean: ", round.(pathmean(path_zz),digits=3)), :black, :right,10))
#cor(path_bps.xs[2,:], path_bps.xs[1,:])
savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/figure2_zz2")
#pathmean(path_zz)
details_zz


# Building a Boomerang Simulation with bounces and refreshment

massmatrix = Matrix{Float64}(I, p, p)
#massmatrix = C1
#gradll(x) = gradloglik(mvg, x)-massmatrix*x#inv(massmatrix)*x
gradll(x) = gradloglik(mvg,x)+x#(inv(massmatrix))*x
nextev_boo(x, v) = nextevent_boo(gradll, mvg, x, v)
T    = 90000  # length of path generated
lref = 0.03#0.5#0.15#0.1    # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [2.,2.]
v0   = rand(MvNormal(zeros(p),massmatrix),1)[:,1] # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG"

# Define a simulation
sim_boo = Simulation(x0, v0, T, nextev_boo, gradll,
                  nextbd, lref, algname,mass=massmatrix; maxgradeval = 1000000, maxsegments=100)
(path_boo, details_boo) = simulate(sim_boo)

samples = samplepath_boo(path_boo,[0.0:0.1:round(path_boo.ts[end]-1,digits=0);])

plot(samples[1,:],samples[2,:],
    legend=false,
    size=(600,600),
    #title="Boomerang Sampler (2-dimensional Gaussian target)",
    xlabel = "X1(t)",
    ylabel = "X2(t)",
    ylims = (-4,4),
    xlims = (-4,4),
    #fmt = :png,
    color=:black)
    scatter!(path_boo.xs[1,:],path_boo.xs[2,:],legend=false)
    annotate!(4, -3, text(string("nsegments: ", details_boo["nsegments"],"\n",
    "nbounce: ", details_boo["nbounce"],"\n",
    "nrefresh: ", details_boo["nrefresh"]), :black, :right, 16))
    #annotate!(2, 4, text(string("path mean: ", round.(pathmean_boo(path_boo),digits=3)), :black, :right,10))
savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/figure2_boo2")


# Building a Boomerang Simulation with NO bounces, but with  refreshment
massmatrix = Matrix{Float64}(I, p, p)
#massmatrix = Matrix{Float64}(I, p, p)
gradll(x) = gradloglik(mvg, x)+inv(massmatrix)*x
#gradll(x0)
nextev_boo(x, v) = nextevent_boo(gradll, mvg, x, v)
T    = 15000.0   # length of path generated
lref = 0.5#0.15#0.1    # rate of refreshment
#x0   = mu+L1.L*randn(p) # sensible starting point
x0 = [0, 0]
v0   = randn(p) # starting velocity
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG"

# Define a simulation
sim_boo2 = Simulation(x0, v0, T, nextev_boo, gradll,
                  nextbd, lref, algname,mass=massmatrix; maxgradeval = 45000)
(path_boo2, details_boo2) = simulate(sim_boo2)

samples2 = samplepath_boo(path_boo2,[0.0:0.1:round(path_boo2.ts[end]-1,digits=0);])
cov(samples2[1,:],samples2[2,:])
plot(samples2[1,:],samples2[2,:],
    legend=false,
    size=(600,600),
    title="Boomerang Sampler (2-dimensional Gaussian target)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    #scatter!(path_boo2.xs[1,:],path_boo2.xs[2,:],legend=false)
    annotate!(2, 3.5, text(string("nsegments: ", details_boo2["nsegments"],"\n",
    "nbounce: ", details_boo2["nbounce"],"\n",
    "nrefresh: ", details_boo2["nrefresh"]), :black, :right, 10))
    annotate!(2, 4, text(string("path mean: ", round.(pathmean_boo(path_boo2),digits=3)), :black, :right,10))

plot(new[1,:],new[2,:],
        legend=false,
        size=(600,600),
        title="Boomerang Sampler (2-dimensional Gaussian target)",
        xlabel = "X_1(t)",
        ylabel = "X_2(t)",
        ylims = (-3.5,4),
        #fmt = :png,
        color=:black)
        #scatter!(path_boo2.xs[1,:],path_boo2.xs[2,:],legend=false)
        annotate!(2, 3.5, text(string("nsegments: ", details_boo2["nsegments"],"\n",
        "nbounce: ", details_boo2["nbounce"],"\n",
        "nrefresh: ", details_boo2["nrefresh"]), :black, :right, 10))
        annotate!(2, 4, text(string("path mean: ", round.(pathmean_boo(path_boo2),digits=3)), :black, :right,10))

## Covariance matrix

In the end, shall we always reshift the trajectory points? L1.L'*x+xstar
new = L1.L'*samples2[:,:]





## Animation trial
axis1(i)=samples2[1,i]
axis2(i)=samples2[2,i]

anim = Animation()
p = plot(samples2[1,:],samples2[2,:],lab="Phi(t)")
#scatter!(1, [axis1,axis2])
scatter!(1, lab="")
for i in 1:length(samples2)
    p[2] = [axis1(i)], [axis2(i)]
    frame(anim)
end
gif(anim)



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

scatter(s[1,:],s[2,:],
    legend=false,
    size=(600,600),
    title="MC Sampler",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    #scatter!(path_boo.xs[1,:],path_boo.xs[2,:],legend=false)
    #annotate!(2, 3.5, text(string("nsegments: ", details_boo["nsegments"],"\n",
    #"nbounce: ", details_boo["nbounce"],"\n",
    #"nrefresh: ", details_boo["nrefresh"]), :black, :right, 10))
    #annotate!(2, 4, text(string("path mean: ", round.(pathmean_boo(path_boo),digits=3)), :black, :right,10))
    savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/figure_mc")


plot(s[1,:],s[2,:])
    #annotate!(2, -2, text(string("nsegments: ", details_boo2["nsegments"],"\n",
    #    "nbounce: ", details_boo2["nbounce"],"\n",
    #    "nrefresh: ", details_boo2["nrefresh"]), :red, :right, 10))
    #    annotate!(-1.7, 2.3, text(string("path mean: ", round.(pathmean(path_boo2),digits=3)), :black, :right,7))
    #    annotate!(-1.7, 2, text(string("correlation_total: ", round.(cor(path_boo2.xs[1,:],path_boo2.xs[2,:]),digits=3)), :black, :right,7))
    #    annotate!(-1.7, 1.7, text(string("correlation_corners: ", round.(cor(path_boo2.xs[1,findall(path_boo2.is_jump)],path_boo2.xs[2,findall(path_boo2.is_jump)]),digits=3)), :black, :right,7))



bounce = nevtakes2 ? sim.nextevent(x, v) : sim.nextevent(x, v, tauref)

r = sin(tau_boo)^2
x .+=sqrt(r)*v+sqrt(1-r)*x
v .+=-sqrt(r)*x+sqrt(1-r)*v

xₜ(x) = x+sin(t)*v+cos(t)*x
yₜ(x) = v-sin(t)*x+cos(t)*v

plot(xₜ, yₜ, x0 , x)
plot(x0,x)



r = 1
k = 3
n = 100

X = samples[1,1:20]
Y = samples[2,1:200]
