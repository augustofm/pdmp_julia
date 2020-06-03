using Distributions
using DelimitedFiles
using AugmentedGaussianProcesses
using KernelFunctions;


X = readdlm("/Users/gusfmagalhaes/.julia/packages/AugmentedGaussianProcesses/FfXUz/examples/data/banana_X_train")
Y = readdlm("/Users/gusfmagalhaes/.julia/packages/AugmentedGaussianProcesses/FfXUz/examples/data/banana_Y_train")

Ms = [4, 8, 16, 32]
models = Vector{AbstractGP}(undef,length(Ms)+1)
kernel = KernelFunctions.SqExponentialKernel(1.0)

for (index, num_inducing) in enumerate(Ms)
    @info "Training with $(num_inducing) points"
    m = SVGP(X, Y, kernel,LogisticLikelihood(),GibbsSampling(),num_inducing)
    @time train!(m,20)
    models[index]=m;
end
