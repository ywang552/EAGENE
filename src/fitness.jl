using CSV, DataFrames
using SparseArrays, StatsBase
using Plots

N = 10
nnz_value = 50


path = raw"E:\bioinformatics\DREAM4 in silico challenge\DREAM4 in-silico challenge\Size 10\DREAM4 gold standards"
gst = CSV.read(joinpath(path, "insilico_size10_1_goldstandard.tsv"), DataFrame; delim='\t', header = false)
gs = spzeros(Bool, 10, 10)
for row in eachrow(gst)
    source, sink, binary = row
    if binary == 1
        gs[parse(Int,source[2:end]),parse(Int,sink[2:end]) ] = 1       
    end
end

path = raw"E:\bioinformatics\DREAM4 in silico challenge\DREAM4 in-silico challenge\Size 10\DREAM4 training data\insilico_size10_1"
ts = CSV.read(joinpath(path, "insilico_size10_1_timeseries.tsv"), DataFrame; delim='\t')
ts = Matrix(ts[:,2:end])
ts
re = vcat([ts[11+i*21:21+i*21,:] for i in 0:4]...)


path = raw"E:\bioinformatics\DREAM4 in silico challenge\DREAM4 in-silico challenge\Size 10\Supplementary information\insilico_size10_1\Perturbations"
pt = Matrix(CSV.read(joinpath(path, "insilico_size10_1_timeseries_perturbations.tsv"), DataFrame; delim='\t'))
pt

path = raw"E:\bioinformatics\DREAM4 in silico challenge\DREAM4 in-silico challenge\Size 10\DREAM4 training data\insilico_size10_1"
wt = Matrix(CSV.read(joinpath(path, "insilico_size10_1_wildtype.tsv"), DataFrame; delim='\t'))

path = raw"E:\bioinformatics\DREAM4 in silico challenge\DREAM4 in-silico challenge\Size 10\DREAM4 training data\insilico_size10_1"
ko = Matrix(CSV.read(joinpath(path, "insilico_size10_1_knockouts.tsv"), DataFrame; delim='\t'))


ss = ts[1:21:end, :]
ss = vcat(wt, ss, ko)

avgwt = median(ss, dims = 1)
z = 10
p = plot(re[:,z], ylims = [0, 1], linestyle = :dash)
scatter!(1:11:55, [re[1:11:end,z]])
hline!(avgwt[z:z])
display(p)
