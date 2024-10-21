struct GridStructure
    m₁::Integer
    m₂::Integer
    condIndSubsets::Vector{Vector{Int64}}
    W::SparseMatrixCSC
    W̄::SparseMatrixCSC
end