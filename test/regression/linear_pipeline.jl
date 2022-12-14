# https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/
X = (age    = [23, 45, 34, 25, 67],
     gender = categorical(['m', 'm', 'f', 'm', 'f']));
y = [67.0, 81.5, 55.6, 90.0, 61.1]

KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels
# pipe = (X -> coerce(X, :age=>Continuous)) |> ContinuousEncoder() |> KNNRegressor(K=2)

pipe_transformer = (X -> coerce(X, :age=>Continuous)) |> ContinuousEncoder()
pipe = pipe_transformer |> KNNRegressor(K=2)

# evaluate(pipe, X, y, resampling=CV(nfolds=3), measure=mae)

mach = machine(pipe, X, y)


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
            "value" => "LinearPipeline"
        ),
    ]
)

mach = fit!(mach)

# complete the experiment
updaterun(mlf, exprun, "FINISHED")

logparam(mlf, exprun, mach)
