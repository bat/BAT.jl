#=
# Run with:
include("test_mcmc_rand.jl"); show_scatter()
=#

using Distributions
using PDMats
using StatsBase
using IntervalSets
using Base.Test
using BAT
using JLD

using BAT.Logging


#α_vec = collect(1:1:20)
#m_vec = collect(1:5:101)
#df_vec = collect(1.0:10.0:51.0)
#iter = collect(1:1:50)

α_vec = collect(0.6:0.5:0.6)
m_vec = collect(10:5:10)
df_vec = collect(1.0:1.0:1.0)
iter = collect(1:1:2)

n = size(iter, 1)

nsq = sqrt(n)

d_mean = load("data_mean.jld")["data_mean"]
d_cov = load("data_cov.jld")["data_cov"]
d_tuned = load("data_tuned.jld")["data_tuned"]

m_mean = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2)
m_cov = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2, 2)

se_mean = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2)
se_cov = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2, 2)

for i in indices(α_vec, 1)
    for m in indices(m_vec, 1)
        for k in indices(df_vec, 1)
            m_mean[i, m, k, 1] = mean(d_mean[i, m, k, :, 1])
            m_mean[i, m, k, 2] = mean(d_mean[i, m, k, :, 2])

            se_mean[i, m, k, 1] = std(d_mean[i, m, k, :, 1])
            se_mean[i, m, k, 2] = std(d_mean[i, m, k, :, 2])

            m_cov[i, m, k, 1, 1] = mean(d_cov[i, m, k, :, 1, 1])
            m_cov[i, m, k, 1, 2] = mean(d_cov[i, m, k, :, 1, 2])
            m_cov[i, m, k, 2, 1] = mean(d_cov[i, m, k, :, 2, 1])
            m_cov[i, m, k, 2, 2] = mean(d_cov[i, m, k, :, 2, 2])

            se_cov[i, m, k, 1, 1] = std(d_cov[i, m, k, :, 1, 1])
            se_cov[i, m, k, 1, 2] = std(d_cov[i, m, k, :, 1, 2])
            se_cov[i, m, k, 2, 1] = std(d_cov[i, m, k, :, 2, 1])
            se_cov[i, m, k, 2, 2] = std(d_cov[i, m, k, :, 2, 2])

        end
    end
end

best_mean_dim1 = minimum(se_mean[:, :, :, 1])
best_mean_dim2 = minimum(se_mean[:, :, :, 2])

ind_best_mean_dim1 = ind2sub(se_mean[:, :, :, 1], indmin(se_mean[:, :, :, 1]))
ind_best_mean_dim2 = ind2sub(se_mean[:, :, :, 2], indmin(se_mean[:, :, :, 2]))

best_cov_dim11 = minimum(se_cov[:, :, :, 1, 1])
best_cov_dim12 = minimum(se_cov[:, :, :, 1, 2])
best_cov_dim21 = minimum(se_cov[:, :, :, 2, 1])
best_cov_dim22 = minimum(se_cov[:, :, :, 2, 2])

ind_best_cov_dim11 = ind2sub(se_cov[:, :, :, 1, 1], indmin(se_cov[:, :, :, 1, 1]))
ind_best_cov_dim12 = ind2sub(se_cov[:, :, :, 1, 2], indmin(se_cov[:, :, :, 1, 2]))
ind_best_cov_dim21 = ind2sub(se_cov[:, :, :, 2, 1], indmin(se_cov[:, :, :, 2, 1]))
ind_best_cov_dim22 = ind2sub(se_cov[:, :, :, 2, 2], indmin(se_cov[:, :, :, 2, 2]))
