using Random
using Test

using MLFlowClient
using MLJ
using MLJLogger

function mlflow_server_is_running(mlf::MLFlow)
    try
        response = MLFlowClient.mlfget(mlf, "experiments/list")
        return isa(response, Dict)
    catch e
        return false
    end
end

# creates an instance of mlf
# skips test if mlflow is not available on default location, http://localhost:5000
macro ensuremlf()
    e = quote
        mlf = MLFlow()
        mlflow_server_is_running(mlf) || return nothing
    end
    eval(e)
end

@testset "MLJLogger" begin
    @ensuremlf
    @testset "regression" begin
        # include("regression/decision_tree.jl")
        include("regression/linear_pipeline.jl")
        # include("regression/linear_models.jl")
        # include("regression/random_forest.jl")
    end
end
