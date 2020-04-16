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

p=length(x)
# Boundaries
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

# Dist
kx(a,b,theta) = theta[1]^2*exp((-1/(2*theta[2]^2))*(a-b)'*(a-b))
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



gradll(f) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x

ytarget(y) = pdf(MvNormal([4.,2.],Matrix{Float64}(I, 2, 2)), y)

T    = 500  # length of path generated
lref = .1#0.5#0.15#0.1    # rate of refreshment

theta0 = [1., 1.]
massmatrix = Kernelmatrix(theta0)

#x0   = mu+L1.L*randn(p) # sensible starting point
prior = MvNormal(zeros(p),massmatrix)
f0 = Distributions.rand(prior,1)[:,1]
v0 = Distributions.rand(prior,1)[:,1]

algname = "BOOMERANG"
MH = false

## EXAMPLE 1, BOOMERANG GPC CTE BOUND

nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_cte(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp1 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 200000)
(path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)

boo_part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
samples1=samplepath_boo(boo_part1,[0.0:0.01:round(path_mhwithinpdmp1.ts[end]-1,digits=0);])

plot(samples1[56,:],samples1[60,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Cte Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part1.xs[56,:],boo_part1.xs[60,:],legend=false)

## EXAMPLE 2, BOOMERANG GPC AFFINE BOUND

nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_affine(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp2 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 20000)
(path_mhwithinpdmp2, details_mhwithinpdmp2) = simulate(sim_mhwithinpdmp2)

boo_part2=Path(path_mhwithinpdmp2.xs, path_mhwithinpdmp2.ts)
samples2=samplepath_boo(boo_part2,[0.0:0.01:round(path_mhwithinpdmp2.ts[end]-1,digits=0);])

plot(samples2[56,:],samples2[60,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part2.xs[56,:],boo_part2.xs[60,:],legend=false)


# Example 3 MH-within-PDMP
MH = true
T    = 50000
lmh　= 0.05
epsilon=0.05
nmh=20
theta0 = [5., 3.]
ytarget(y) = pdf(MvNormal([5.,3.],Matrix{Float64}(I, 2, 2)), y)
gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_affine(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp3 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxgradeval = 20000000)
(path_mhwithinpdmp3, details_mhwithinpdmp3) = simulate(sim_mhwithinpdmp3)

boo_part3=Path(path_mhwithinpdmp3.xs, path_mhwithinpdmp3.ts)
samples3=samplepath_boo(boo_part3,[0.0:0.01:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])

plot(samples3[56,:],samples3[60,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part3.xs[56,:],boo_part3.xs[60,:],legend=false)

plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])

fbar = mean(samples3,dims=2)
scatter(vcat(x0,x1),fbar)
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))


# Marginal distribution
marginaldist = Vector{Float64}()
theta_vector = unique(path_mhwithinpdmp3.ys)

theta_vector = reshape(theta_vector, (2,2553))

for yaux in theta_vector[2,:]
    #yaux = reshape(yaux[:,1],(2,1))
    idx = findall(x->x==yaux, path_mhwithinpdmp3.ys[2,:])
    if length(idx)==1
        subset_sample=path_mhwithinpdmp3.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp3.xs[:,idx], path_mhwithinpdmp3.ts[idx])
        subset_sample = samplepath_boo(subset,[subset.ts[1]:0.02:subset.ts[end];])
    end
    N = size(subset_sample)[2]
    marginals= logistic.(subset_sample.*Y)
    prob=0
    for aux in 1:N
        prob+=prod(marginals[:,aux])
    end
    push!(marginaldist, prob/N)
end

using Plots; plot()
#plot(size=(600,600))
scatter(theta_vector[1,:],theta_vector[2,:],size=(2000,2000))
scatter(theta_vector[1,:],marginaldist,size=(2000,2000))

using3D()
n = Int(length(theta_vector)/2)
x = theta_vector[1,:]
y = theta_vector[2,:]

xgrid = repeat(x',n,1)
ygrid = repeat(y,1,n)

z = zeros(n,n)

for i in 1:n
    for j in 1:n
        if i==j
            z[i:i,j:j] .= marginaldist[i]
        else
            z[i:i,j:j] .= NaN
        end
    end
end

fig = figure("pyplot_surfaceplot",figsize=(10,10))
ax = fig.add_subplot(2,1,1,projection="3d")
plot_surface(x,y,z, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)

xlabel("sigma")
ylabel("l")
PyPlot.title("Surface Plot")

py, bestindex = findmax(marginaldist)
theta_vector[:,bestindex]

using Plots; pyplot()
plot_surface(theta_vector[1,:],theta_vector[2,:],marginaldist,rstride=2,edgecolors="k", cstride=2,
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)

# Example 3 Factorized Boomerang Sampler
nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_factor(gradll, f, v)
sim_mhwithinpdmp = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 20000)
(path_mhwithinpdmp, details_mhwithinpdmp) = simulate(sim_mhwithinpdmp)

boo_part=Path(path_mhwithinpdmp.xs, path_mhwithinpdmp.ts)
samples=samplepath_boo(boo_part,[0.0:0.01:round(path_mhwithinpdmp.ts[end]-1,digits=0);])

plot(samples[40,:],samples[57,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (61-dimensional GPC)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part.xs[40,:],boo_part.xs[57,:],legend=false)



scatter(x,f0)

scatter(x,samples[:,340])
    scatter(x,samples[:,3140])
#    scatter(x,samples[:,13340])
    #scatter(vcat(x0,x1),samples[:,18040])

fbar = mean(samples,dims=2)
scatter(vcat(x0,x1),fbar)
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))


## 2-step Bayesian Inference
#marginaldist = Vector{Float64}()
#theta_vector = unique(path_mhwithinpdmp.ys)

for yaux in theta_vector
    idx = findall(x->x==yaux, path_mhwithinpdmp3.ys)
    if length(idx)==1
        subset_sample=path_mhwithinpdmp3.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp3.xs[:,idx], path_mhwithinpdmp3.ts[idx])
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


#MHsampler=false, y0=f0, ytarget=ytarget,
#epsilon=epsilon, nmh=nmh,lambdamh=lmh,
