module Conformal
using Statistics

export compute_nonconformity_scores, compute_p_values, predict_with_conformal

# Nonconformity score = |ŷ - y|
function compute_nonconformity_scores(ŷ::Vector{<:Real}, y::Vector{<:Real})
    return abs.(ŷ .- y)
end

# Compute p-values for test scores
function compute_p_values(cal_scores::Vector{<:Real}, test_scores::Vector{<:Real})
    return [mean(cal_scores .>= s) for s in test_scores]
end

# Predict using p-values and a threshold (significance level)
function predict_with_conformal(p_values::Vector{<:Real}, α::Real)
    return p_values .<= α
end

end # modulei