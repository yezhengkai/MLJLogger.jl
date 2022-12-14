Random.seed!(777)

# Get type for model
Regressor = @load RandomForestRegressor pkg=DecisionTree

# Make data for regression
n = 50
X = MLJ.table(reshape(collect(Float64, 1:n), n, 1))
y = 2 .* collect(1:n) .+ 5 + randn(n)

expected_params = Dict(
    :max_depth => -1,
    :min_samples_leaf => 1,
    :min_samples_split => 2,
    :min_purity_increase => 0.0,
    :n_subfeatures => -1,
    :n_trees => 10,
    :sampling_fraction => 0.7,
    :feature_importance => :impurity,
    :rng => Random.GLOBAL_RNG,
)
model = Regressor(;expected_params...)
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
            "value" => "RandomForestRegressor-pkg=DecisionTree"
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
