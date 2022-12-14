using Random

using MLJ
using MLFlowClient

using MLJLogger

Random.seed!(777)

# Create MLFlow instance
mlf = MLFlow("http://localhost:5000")

# Initiate new experiment
experiment = getorcreateexperiment(mlf, "linear-regression")

# Create a run in the new experiment
exprun = createrun(mlf, experiment)

# Get type for model
LinearRegressor = @load LinearRegressor pkg="MLJLinearModels"

# Make data for regression
n = 50
X = MLJ.table(reshape(collect(Float64, 1:n), n, 1))
y = 2 .* collect(1:n) .+ 5 + randn(n)

model = LinearRegressor()
mach = machine(model, X, y)
mach = fit!(mach)
# predict(mach, X)
# fitted_params(mach)

function MLFlowClient.logparam(mlf::MLFlow, run::Union{String,MLFlowRun,MLFlowRunInfo}, mach::MLJBase.Machine)
    for (key, value) in pairs(params(mach))
        value = getfield(mach.model, key)
        MLFlowClient.logparam(mlf, run, key, value)
    end
end

logparam(mlf, exprun, mach)


function MLFlowClient.logmetric(mlf::MLFlow, run::Union{String,MLFlowRun,MLFlowRunInfo}, mach::MLJBase.Machine, X; timestamp=missing, step=missing, measures=nothing)
    _measures = MLJBase._actual_measures(measures, mach.model)
    ŷ = predict(mach, X)
    y = mach.args[2]()
    for m in _measures
        @info m(ŷ, y)
        MLFlowClient.logmetric(mlf, run, string(typeof(m)), m(ŷ, y); timestamp=timestamp, step=step)
    end
end

logmetric(mlf, exprun, mach, X)
