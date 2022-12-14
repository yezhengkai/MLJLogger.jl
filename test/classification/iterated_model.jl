X, y = make_moons(1000, rng=12)
EvoTreeClassifier = @load EvoTreeClassifier verbosity=0

iterated_model = IteratedModel(model=EvoTreeClassifier(rng=123, η=0.005),
                               resampling=Holdout(rng=123),
                               measures=log_loss,
                               controls=[Step(5),
                                         Patience(10),
                                         NumberLimit(100)],
                               retrain=true)

mach = machine(iterated_model, X, y)

fit!(mach)
