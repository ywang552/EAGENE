using CSV
using DataFrames
using SparseArrays, StatsBase
using Plots
a, b = "d4", 1
N = 10
nnz_value = 50
fp = joinpath("data", "small")

fn = "$(a)_$(b)_wt.tsv"
data = CSV.read(joinpath(fp, fn), DataFrame; delim='\t')
wt = Matrix(data)

fn = "$(a)_$(b)_ko.tsv"
df = CSV.read(joinpath(fp, fn), DataFrame; delim='\t')
ko = Matrix(df)

fn = "$(a)_$(b)_ts.tsv"
df = CSV.read(joinpath(fp, fn), DataFrame; delim='\t')
ts = Matrix(df)

fn = "$(a)_1_gs.tsv"
df = CSV.read(joinpath(fp, fn), DataFrame; delim='\t', header = false)
gst = Matrix(df)
gs = spzeros(Bool,size(ko,1), size(ko,1))
N = size(ko,1)
for row in eachrow(gst)
    src, dst, connect = row
    gs[parse(Int, replace(src, "G" => "")), parse(Int, replace(dst, "G" => ""))] = connect        
end 

fn = "$(a)_$(b)_pt.tsv"
df = CSV.read(joinpath(fp, fn), DataFrame; delim='\t')
pt = Matrix(df)
println(df)

ts = ts[:,2:end]

re = vcat([ts[11+i*21:21+i*21,:] for i in range(0, 4)]...)

ss = vcat(wt, ts[1:21:end,:], ko)

avgwt = vec(median(ss, dims = 1))

z = 1
re = reshape(re, (11, 5, N))
p = plot(re[:,1,z], ylims = [0, 1], linestyle = :dash, lw = 4, label = "time series data ")
hline!(avgwt[z:z], lw = 4, label = "average steady_states")
xlabel!("time")
ylabel!("gene strength")
display(p)

# savefig("figs/time_series_g1_perturbed.png")

heatmap(Int.(gs), color = [:black, :green])
xlabel!("gene counter")
ylabel!("gene counter")
# savefig("figs/heatmap_gs_d4_t1.png")