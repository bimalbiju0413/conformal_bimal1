module Evaluate

using Statistics

export run_evaluation, compute_precision, compute_recall, compute_f1_score

# --- Conformal Prediction Functions ---
function compute_nonconformity_scores(ŷ::Vector{<:Real}, y::Vector{<:Real})
    return abs.(ŷ .- y)
end

function compute_p_values(cal_scores::Vector{<:Real}, test_scores::Vector{<:Real})
    return [mean(cal_scores .>= s) for s in test_scores]
end

function predict_with_conformal(p_values::Vector{<:Real}, α::Real)
    return p_values .<= α
end

# --- Metric Functions ---
function compute_precision(y_pred::Vector{Bool}, y_true::Vector{<:Real})
    tp = sum(y_pred .& (y_true .== 1))
    fp = sum(y_pred .& (y_true .== 0))
    return tp / (tp + fp + 1e-8)
end

function compute_recall(y_pred::Vector{Bool}, y_true::Vector{<:Real})
    tp = sum(y_pred .& (y_true .== 1))
    fn = sum(.!y_pred .& (y_true .== 1))
    return tp / (tp + fn + 1e-8)
end

function compute_f1_score(p, r)
    return 2 * (p * r) / (p + r + 1e-8)
end

# --- Evaluation Function ---
function run_evaluation(model, train_state, X_cal, y_cal, X_test, y_test, α::Real)
    # Generate predictions on calibration and test sets
    ŷ_cal, _ = model(X_cal, train_state.parameters, train_state.states)
    ŷ_cal = vec(ŷ_cal)
    
    ŷ_test, _ = model(X_test, train_state.parameters, train_state.states)
    ŷ_test = vec(ŷ_test)
    
    # Compute nonconformity scores and p-values using our functions
    cal_scores = compute_nonconformity_scores(ŷ_cal, y_cal)
    test_scores = compute_nonconformity_scores(ŷ_test, y_test)
    p_values = compute_p_values(cal_scores, test_scores)
    
    # Flag anomalies using the conformal threshold
    predictions = collect(predict_with_conformal(p_values, α))
    
    # Compute metrics using our custom functions explicitly
    p_metric = compute_precision(predictions, y_test)
    r_metric = compute_recall(predictions, y_test)
    f1 = compute_f1_score(p_metric, r_metric)
    
    return Dict("precision" => p_metric, "recall" => r_metric, "f1" => f1)
end

end # module Evaluate