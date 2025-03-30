function run_training_for_dataset(key::String, splits::Dict{String,Tuple})
    @info "Training on dataset: $key"
    # Unpack splits: each is a tuple with (X_train, y_train, X_cal, y_cal, X_test, y_test)
    X_train, y_train, X_cal, y_cal, X_test, y_test = splits[key]
    
    # Prepare training data (ensure correct shapes)
    x_train = X_train  # already in the right shape (after luxify)
    y_train = y_train  # vector of labels
    
    # Model setup
    model = Model.LSTMAnomalyClassifier(1, 32, 1)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Optimizer and train state
    opt = Optimisers.Adam()
    train_state = Lux.Training.TrainState(model, ps, st, opt)
    
    # Loss function and metric
    loss_fn = BinaryCrossEntropyLoss()
    accuracy(ŷ, y) = mean((ŷ .> Float32(0.5)) .== y)
    
    # Loss wrapper
    function compute_loss(model, ps, st, (x, y))
        ŷ, st_ = model(x, ps, st)
        loss = loss_fn(ŷ, y)
        return loss, st_, (; y_pred = ŷ)
    end
    
    # Training loop (for example, 30 epochs)
    for epoch in 1:10
        train_state
        grads, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(),
            (model, ps, st, data) -> compute_loss(model, ps, st, data),
            (x_train, y_train),
            train_state
        )
    
        acc = accuracy(stats.y_pred, y_train)
        @info "Epoch $epoch | Loss = $(round(loss, digits=5)) | Accuracy = $(round(acc * 100, digits=2))%"
    end
    
    # Return the trained model and train state (or evaluation metrics)
    return train_state
end