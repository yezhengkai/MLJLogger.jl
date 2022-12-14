using CondaPkg

CondaPkg.resolve()
CondaPkg.withenv() do
    mlflow = CondaPkg.which("mlflow")
    run(`$mlflow ui`)
end
