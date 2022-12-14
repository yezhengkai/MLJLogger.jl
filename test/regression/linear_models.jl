Random.seed!(777)

# Get type for model
Regressor = @load LinearRegressor pkg=MLJLinearModels

# Make data for regression
n = 50
X = MLJ.table(reshape(collect(Float64, 1:n), n, 1))
y = 2 .* collect(1:n) .+ 5 + randn(n)

expected_params = Dict(
    "fit_intercept" => true,
    "solver" => nothing
)
model = Regressor(values(expected_params)...)
mach = machine(model, X, y)

# Create MLFlow instance
mlf = MLFlow("http://localhost:5000")

# Initiate new "linear-models" experiment or get it
exp = getorcreateexperiment(mlf, "regression")

# Create a run in experiment
exprun = createrun(
    mlf,
    exp;
    tags=[
        Dict(
            "key" => "mlflow.runName",
            "value" => "LinearRegressor-pkg=MLJLinearModels"
        ),
    ]
)

mach = fit!(mach)

# complete the experiment
updaterun(mlf, exprun, "FINISHED")

logparam(mlf, exprun, mach)
logmetric(mlf, exprun, mach, X)

# Get new runs from server
exprun = getrun(mlf, get_run_id(exprun))
@test sort(collect(keys(get_params(exprun)))) == sort(string.(keys(expected_params)))
@test sort(collect(values(get_params(exprun)))) == sort(string.(values(expected_params)))
