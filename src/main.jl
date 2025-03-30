using DataFrames, CSV, Dates, Statistics, PrettyTables, Random

# Include all necessary files
include("load_data.jl")        # loads `datasets` and `labels`
include("preprocess.jl")       # produces `all_splits`
include("model.jl")            # defines LSTMAnomalyClassifier
include("train_all.jl")        # defines run_training_for_dataset(...)
include("Conformal.jl")        # defines conformal prediction functions
include("evaluation.jl")       # defines the Evaluate module

using .Model
using .Conformal
using .Evaluate
using Random

# Set the window size (must match preprocessing)
const WINDOW = 50

# Define the results directory
results_dir = "results"
if !isdir(results_dir)
    mkpath(results_dir)
end

# Helper: reconstruct test DataFrame from the original dataset.
# We assume that the test split of the original DataFrame is from index t2+1 to end,
# where t2 = floor(0.60 * N). For sliding windows, we take the timestamp, value, and label
# from the last row in each window.
function reconstruct_test_df(df::DataFrame)
    N = nrow(df)
    t2 = Int(floor(0.60 * N))
    test_df = df[(t2+1):end, :]
    n_windows = nrow(test_df) - WINDOW + 1
    if n_windows <= 0
        error("Not enough test data to form sliding windows.")
    end
    timestamps = test_df.timestamp[WINDOW:end]
    values = test_df.value[WINDOW:end]
    # Assume the original df was marked with an anomaly flag in column "is_anomaly"
    labels = hasproperty(test_df, :is_anomaly) ? test_df.is_anomaly[WINDOW:end] : zeros(Bool, n_windows)
    return DataFrame(timestamp = timestamps, value = values, label = labels)
end

# Pipeline function: train, evaluate, and produce a CSV result DataFrame for one dataset.
function run_pipeline_for_dataset(key::String, splits::Dict{String, Tuple})
    println("Processing dataset: ", key)
    
    # Unpack splits: (X_train, y_train, X_cal, y_cal, X_test, y_test)
    X_train, y_train, X_cal, y_cal, X_test, y_test = splits[key]
    
    # Train the model on this dataset
    train_state = run_training_for_dataset(key, splits)
    
    # Set the conformal significance threshold
    α = 0.20
    
    # Evaluate the model using the Evaluate.run_evaluation function
    metrics = Evaluate.run_evaluation(train_state.model, train_state, X_cal, y_cal, X_test, y_test, α)
    println("Dataset: $key, Evaluation Metrics: ", metrics)
    
    # For CSV reporting: compute anomaly scores on test set using absolute error
    ŷ_test, _ = train_state.model(X_test, train_state.parameters, train_state.states)
    ŷ_test = vec(ŷ_test)
    anomaly_scores = abs.(ŷ_test .- y_test)
    
    # Reconstruct the test DataFrame from the original dataset
    original_df = datasets[key]
    test_df_reconstructed = reconstruct_test_df(original_df)
    
    # Check if the number of sliding windows matches the number of anomaly scores
    n_windows = nrow(test_df_reconstructed)
    if n_windows != length(anomaly_scores)
        println("Warning: For dataset $key, expected $n_windows windows but got $(length(anomaly_scores)) anomaly scores.")
    end
    
    # Add the computed anomaly scores to the reconstructed test DataFrame
    test_df_reconstructed[!, :anomaly_score] = anomaly_scores
    
    # Keep only the desired columns: timestamp, value, anomaly_score, label
    result_df = test_df_reconstructed[:, [:timestamp, :value, :anomaly_score, :label]]
    
    # Add a column for the dataset key (so you know the source)
    result_df[!, :dataset] = fill(key, nrow(result_df))
    
    return result_df, metrics
end

# --- Main Loop: Process All Datasets ---
# We'll also group results by subfolder (e.g., "realAWSCloudwatch", "realKnownCause", etc.)
results_by_subfolder = Dict{String, Vector{DataFrame}}()
metrics_by_dataset = Dict{String, Dict}()

for (key, _) in all_splits
    if !haskey(labels, key)
        println("Skipping dataset $key as no labels available.")
        continue
    end
    
    result_df, metrics = run_pipeline_for_dataset(key, all_splits)
    metrics_by_dataset[key] = metrics
    
    # Determine subfolder from dataset key. If key is "subfolder/filename.csv", subfolder is first part.
    subfolder = occursin("/", key) ? split(key, "/")[1] : "default"
    
    # Save an individual CSV for this dataset
    safe_key = replace(key, "/" => "_")
    output_subdir = joinpath(results_dir, subfolder)
    if !isdir(output_subdir)
        mkpath(output_subdir)
    end
    output_file = joinpath(output_subdir, safe_key * "_results.csv")
    CSV.write(output_file, result_df)
    println("Saved CSV for dataset $key at: ", output_file)
    
    # Also store in our subfolder dictionary for aggregated reports
    if haskey(results_by_subfolder, subfolder)
        push!(results_by_subfolder[subfolder], result_df)
    else
        results_by_subfolder[subfolder] = [result_df]
    end
end

# --- Save Aggregated CSVs for Each Subfolder ---
for (subfolder, dfs) in results_by_subfolder
    combined_df = vcat(dfs...)
    output_subdir = joinpath(results_dir, subfolder)
    output_file = joinpath(output_subdir, "aggregated_results.csv")
    CSV.write(output_file, combined_df)
    println("Saved aggregated CSV for subfolder '$subfolder' at: ", output_file)
end

# --- Print Aggregated Evaluation Metrics ---
println("\nAggregated Evaluation Metrics:")
for (key, metrics) in metrics_by_dataset
    println("Dataset: $key => ", metrics)
end

# --- Create a Pretty Table Report for Evaluation Metrics ---
report = DataFrame(Dataset = String[], Precision = Float64[], Recall = Float64[], F1 = Float64[])
for (key, metrics) in metrics_by_dataset
    push!(report, (key, metrics["precision"], metrics["recall"], metrics["f1"]))
end

pretty_table(report, header=["Dataset", "Precision", "Recall", "F1 Score"], crop=:none)