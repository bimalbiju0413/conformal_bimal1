using CSV, DataFrames, JSON3, Dates

data_dir = "data"
labels_path = "labels/combined_windows.json"

datasets = Dict{String,DataFrame}()
for (root, _, files) in walkdir(data_dir)
    for file in files
        if endswith(file, ".csv")
            fullpath = joinpath(root, file)
            # Build the relative key exactly as in the JSON
            relpath = replace(fullpath, data_dir * "/" => "")
            df = CSV.read(fullpath, DataFrame; dateformat="yyyy-mm-dd HH:MM:SS")
            df.timestamp = DateTime.(df.timestamp)
            sort!(df, :timestamp)
            datasets[relpath] = df
            println("Loaded $relpath → $(nrow(df)) rows")
        end
    end
end

labels = JSON3.read(open(labels_path))
println("Loaded labels for $(length(keys(labels))) series.")
