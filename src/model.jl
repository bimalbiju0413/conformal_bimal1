module Model

using Lux

export LSTMAnomalyClassifier

"""
    LSTMAnomalyClassifier(input_dim, hidden_dim, output_dim)

Creates an LSTM-based sequence-to-one classifier.
- Input shape: (batch, seq_len, input_dim)
- Output: scalar per batch (probability of anomaly)
"""
function LSTMAnomalyClassifier(input_dim::Int, hidden_dim::Int, output_dim::Int)
    lstm_cell = Lux.LSTMCell(input_dim => hidden_dim)
    classifier = Lux.Dense(hidden_dim => output_dim, Lux.sigmoid)

    return Lux.@compact(; lstm_cell, classifier) do x::AbstractArray{T, 3} where {T}
        x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
        y, carry = lstm_cell(x_init)
        for x in x_rest
            y, carry = lstm_cell((x, carry))
        end
        @return vec(classifier(y))
    end
end

end # module