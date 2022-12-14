module MLJLogger

using MLFlowClient
using MLJBase
using MLJIteration
using MLJModels


# ===== For MLJBase.Machine (start) =====
function MLFlowClient.logparam(
    mlf::MLFlowClient.MLFlow,
    run::Union{String,MLFlowClient.MLFlowRun,MLFlowClient.MLFlowRunInfo},
    mach::MLJBase.Machine,
)
    for (key, value) in pairs(params(mach))
        value = getfield(mach.model, key)
        MLFlowClient.logparam(mlf, run, key, value)
    end

    return nothing
end

function MLFlowClient.logmetric(
    mlf::MLFlowClient.MLFlow,
    run::Union{String,MLFlowClient.MLFlowRun,MLFlowClient.MLFlowRunInfo},
    mach::MLJBase.Machine,
    X,
    y=nothing;
    timestamp=missing,
    step=missing,
    measures=nothing,
)
    if isnothing(y)
        y = mach.args[2]()
    end

    if nrows(X) != nrows(y)
        error("The number of rows of x and y do not match.")
    end

    _measures = MLJBase._actual_measures(measures, mach.model)
    ŷ = predict(mach, X)
    for m in _measures
        MLFlowClient.logmetric(
            mlf,
            run,
            string(typeof(m)), m(ŷ, y);
            timestamp=timestamp,
            step=step
        )
    end

    return nothing
end
# ===== For MLJBase.Machine (end) =====

# ===== For MLJBase.Machine{<:MLJBase.{}Pipeline, C} (start) =====
function MLFlowClient.logparam(
    mlf::MLFlowClient.MLFlow,
    run::Union{String,MLFlowClient.MLFlowRun,MLFlowClient.MLFlowRunInfo},
    mach::MLJBase.Machine{
        <:Union{
            MLJBase.DeterministicPipeline,
            MLJBase.ProbabilisticPipeline,
            MLJBase.IntervalPipeline,
            MLJBase.UnsupervisedPipeline,
            MLJBase.StaticPipeline,
        }
    },
)
    components = MLJBase.components(mach.model)
    for component in components
        if isempty(MLJModels.models(MLJBase.name(component)))
            continue
        end
        comp_name = MLJBase.name(component)
        for (key, value) in pairs(params(component))
            MLFlowClient.logparam(mlf, run, "$(comp_name)-$(key)", value)
        end
    end

    return nothing
end

# TODO: MLFlowClient.logmetric

# ===== For MLJBase.Machine{<:MLJBase.{}Pipeline, C} (end) =====

# ===== For MLJBase.Machine{<:MLJIteration.EitherIteratedModel, C} (end) =====
# https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/
function MLFlowClient.logparam(
    mlf::MLFlowClient.MLFlow,
    run::Union{String,MLFlowClient.MLFlowRun,MLFlowClient.MLFlowRunInfo},
    mach::MLJBase.Machine{<:MLJIteration.EitherIteratedModel},
)

end

end
