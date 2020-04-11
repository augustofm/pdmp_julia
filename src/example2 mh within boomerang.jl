using PDSampler
using LinearAlgebra         #For Cholesky
import Random               #For generating random vectors               #For plotting
using Plots
using DelimitedFiles
using Distributions


# Metropolis within PDMP
x0=rand(Normal(0,0.8), 30)
x1=vcat(rand(Normal(-6,0.8),20), rand(Normal(2,0.8),10))
x=vcat(x0,x1)
Y=vcat(repeat([-1];outer=[30]),repeat([1];outer=[30]))
#(exp.(-f).+1).^(-1)-(Y.+1)./2
#scatter(x,Y)
p=length(x)
# Boundaries
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)
#ns, a = [[100 0];[100 0]], zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

# Dist
#mu = zeros(p)
kx(a,b,theta) = theta[1]^2*exp((-1/(2*theta[2]^2))*(a-b)'*(a-b))
function Kernelmatrix(theta)
    #theta = theta[1]
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
    #theta = theta[1]
    k = Matrix{Float64}(I, 1, p)
    for j in 1:p
        k[1,j]=kx(x[j],xstar,theta)
    end
    k'
end



#massmatrix = Matrix{Float64}(I, p, p)
#gradll(f,theta) = gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
gradll(f) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x

ytarget(y) = pdf(MvNormal([2.,1.],Matrix{Float64}(I, 2, 2)), y)

T    = 10000  # length of path generated
lref = 0.1#0.5#0.15#0.1    # rate of refreshment
lmh　= 1.0

epsilon=0.1
nmh=20
#x0   = mu+L1.L*randn(p) # sensible starting point
#f0 = randn(p)
#f0 = repeat([1];outer=[p])
#f0 = repeat([1];outer=[p])

theta0 = [1, 1]

#maximum(eigvals(Kernelmatrix(theta0)))
massmatrix = Kernelmatrix(theta0)
initial = MvNormal(zeros(p),massmatrix)
f0 = Distributions.rand(initial,1)[:,1]
v0 = Distributions.rand(initial,1)[:,1]

#v0= vcat(repeat([0];outer=[30]),repeat([1];outer=[30]))
#v0   = randn(p) # starting velocity
nextev_mhwithinpdmp(f, v) = nextevent_mhwithinpdmp(gradll, f, v)
#v0  /= norm(v0) # put it on the sphere (not necessary)
algname = "BOOMERANG"
MH = false
# Define a simulation
#MHsampler=false, y0=f0, ytarget=ytarget,
#epsilon=epsilon, nmh=nmh,lambdamh=lmh,
sim_mhwithinpdmp = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 100000)
(path_mhwithinpdmp, details_mhwithinpdmp) = simulate(sim_mhwithinpdmp)

boo_part=Path(path_mhwithinpdmp.xs, path_mhwithinpdmp.ts)
samples=samplepath_boo(boo_part,[0.0:0.01:round(path_mhwithinpdmp.ts[end]-1,digits=0);])
#marginaldist = Vector{Float64}()
#theta_vector = unique(path_mhwithinpdmp.ys)

plot(path_mhwithinpdmp.ts,path_mhwithinpdmp.ys[1,:])
scatter(path_mhwithinpdmp.ys[1,:],path_mhwithinpdmp.ys[2,:])

plot(samples[40,:],samples[41,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (61-dimensional GPC)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part.xs[40,:],boo_part.xs[41,:],legend=false)

cov(samples[11,:],samples[13,:])

scatter(vcat(x0,x1),(exp.(-Y.*samples[:,2340]).+1).^(-1))
scatter(vcat(x0,x1),(exp.(-Y.*fbar).+1).^(-1))



for yaux in theta_vector
    idx = findall(x->x==yaux, path_mhwithinpdmp.ys)
    if length(idx)==1
        subset_sample=path_mhwithinpdmp.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp.xs[:,idx], path_mhwithinpdmp.ts[idx])
        subset_sample = samplepath_boo(subset,[subset.ts[1]:0.02:subset.ts[end];])
    end
    N = size(subset_sample)[2]
    marginals= logistic.(subset_sample.*y)
    prob=0
    for aux in 1:N
        prob+=prod(marginals[:,aux])
    end
    push!(marginaldist, prob/N)
end

plot(theta_vector,marginaldist)
