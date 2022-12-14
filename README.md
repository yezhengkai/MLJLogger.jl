Tracking and sharing MLJ workflows using MLFlow

## Run mlflow server for testing in local machine
```bash
$ julia --project=mlflow-server --startup-file=no
```
```julia-repl
julia> include("mlflow-server/run_mlflow_ui.jl")  # This can be stopped by pressing Ctrl-C.
```


## Reference
- [MLJ.jl Projects - Summer of Code: Tracking and sharing MLJ workflows using MLFlow](https://julialang.org/jsoc/gsoc/MLJ/#tracking_and_sharing_mlj_workflows_using_mlflow)
- [Discourse: MLFlow Integration](https://discourse.julialang.org/t/mlflow-integration/88320)
- [GitHub: CondaPkg.jl](https://github.com/cjdoris/CondaPkg.jl)