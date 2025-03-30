include("load_data.jl")   # defines `datasets::Dict{String,DataFrame}` and `labels::Dict`

using Dates, Statistics, DataFrames, MLDataPattern

const WINDOW = 50

# ─── Timestamp Parser ──────────────────────────────────────────────────────
function parse_dt(s::String)
    for fmt in (dateformat"yyyy-mm-ddTHH:MM:SS.sssss", dateformat"yyyy-mm-dd HH:MM:SS.sssss")
        try return DateTime(s, fmt) catch end
    end
    error("Unable to parse timestamp: $s")
end

# ─── Sliding‑Window Generator ──────────────────────────────────────────────
function make_windows(vals::Vector{Float32}, flags::Vector{Bool}, w::Int)
    # Collect each sliding window into a plain Vector{Float32}
    X_raw = [collect(window) for window in slidingwindow(vals, w, 1)]
    # Collect and collapse boolean flags into a single label per window
    y = Float32.(map(any, [collect(window) for window in slidingwindow(flags, w, 1)]))
    return prepare_flux_input(X_raw), y
end

function prepare_flux_input(X::Vector{Vector{Float32}})
    N, w = length(X), length(X[1])
    arr = Array{Float32}(undef, N, w, 1)
    for i in 1:N
        arr[i, :, 1] .= X[i]
    end
    return arr
end

# ─── Helper: Convert to Lux format (input_dim, seq_len, batch) ─────────────
luxify(X::Array{Float32,3}) = permutedims(X, (3, 2, 1))

# ─── Preprocess All NAB Series ─────────────────────────────────────────────
all_splits = Dict{String, Tuple}()

for (path, df) in datasets
    ints = labels[path]
    df.is_anomaly = falses(nrow(df))
    for iv in ints
        start, stop = parse_dt(iv[1]), parse_dt(iv[2])
        mask = (df.timestamp .>= start) .& (df.timestamp .<= stop)
        df.is_anomaly[mask] .= true
    end

    N = nrow(df)
    t1, t2 = Int(floor(0.40*N)), Int(floor(0.60*N))
    train_df = df[1:t1, :]; calib_df = df[(t1+1):t2, :]; test_df = df[(t2+1):end, :]

    μ, σ = mean(train_df.value), std(train_df.value)
    for sub in (train_df, calib_df, test_df)
        sub.value = Float32.((sub.value .- μ) ./ σ)
    end

     local X_train, train_y = make_windows(train_df.value, collect(train_df.is_anomaly), WINDOW)
     local X_calib, calib_y = make_windows(calib_df.value, collect(calib_df.is_anomaly), WINDOW)
     local X_test, test_y   = make_windows(test_df.value, collect(test_df.is_anomaly), WINDOW)

    # 🔄 Convert to Lux format
    all_splits[path] = (
        luxify(X_train),
        train_y,
        luxify(X_calib),
        calib_y,
        luxify(X_test),
        test_y
    )

    println("Processed $(basename(path)): Train=$(size(X_train)), Calib=$(size(X_calib)), Test=$(size(X_test)) → Lux format = ", size(luxify(X_train)))
end