using Serialization
using AxisKeys
using StaticArrays
using SplitApplyCombine
Dictionary = SplitApplyCombine.Dictionary
import Base.Iterators: product
using Statistics
using StatsBase
using DataFrames, DataFramesMeta, CSV
using ProgressMeter
using DataStructures: DefaultDict
using StatsFuns: logistic
import StatsFuns

# %% ==================== Project-specific ====================

function relative_score(problem, path; normalize=true)
    actual = value(problem, path)
    best = value(problem)
    if normalize
        random = random_value(problem)
        (actual - random) / (best - random)
    else
        actual - best
    end
end

random_value(problem) = mean(p->value(problem, p), paths(problem))

make_pid(i) = "P" * lpad(i, 2, '0')

# %% ==================== General Purpose ====================

function sliding_window(xs, k)
    map(1:length(xs) - (k-1)) do i
        @view xs[i:i+(k-1)]
    end
end

function partition(xs, k)
    map(1:k:length(xs) - (k-1)) do i
        @view xs[i:i+(k-1)]
    end
end

basetype(f::Type) = f.name.wrapper
basetype(f) = basetype(typeof(f))

call(f) = f()

named_tuple(x::NamedTuple) = x
function named_tuple(x)
    n = Tuple(propertynames(x))
    NamedTuple{n}(getproperty.(Ref(x), n))
end

ensure_finite(x; default) = isfinite(x) ? x : default
ensure_nonmissing(x; default) = !ismissing(x) ? x : default

macro require(ex)
    quote
        $ex || return missing
    end |> esc
end

function filtermap(f::Function, args...)
    map(f, args...) |> skipmissing |> collect
end

Base.write(filename::AbstractString) = x -> write(filename, x)

function chunk(arr, sz)
    @assert length(arr) % sz == 0
    [arr[i:i+sz-1] for i in 1:sz:length(arr)]
end

softmax(x::Vector) = StatsFuns.softmax(x)

function softmax(f::Function, xs, i::Int)
    num = NaN; denom = 0
    for (j, x) in enumerate(xs)
        efx = exp(f(x))
        if i == j
            num = efx
        end
        denom += efx
    end
    num / denom
end

function lapse(p::Real, ε, N)
    (1 - ε) * p + ε * (1/N)
end

function lapse(p::Vector, ε)
    N = length(p)
    lapse.(p, ε, N)
end

function lapse_softmax(f::Function, xs, i::Int; β, ε)
    lapse(softmax(x -> β * f(x), xs, i), ε, length(xs))
end
lapse_softmax(xs, i::Int; β, ε) = lapse_softmax(identity, xs, i; β, ε)


function sample_lapse_softmax(f::Function, xs; β, ε)
    if rand() < ε
        rand(xs)
    else
        argmax(xs) do x
            f(x) + rand(Gumbel()) / β
        end
    end
end


Base.get(name::Symbol) = Base.Fix2(getproperty, name)
Base.get(i::Int) = Base.Fix2(getindex, i)
Base.get(x, name::Symbol) = getproperty(x, name)
Base.get(x, i::Int) = getindex(x, i)

getprop(x, name::Symbol, default) = hasproperty(x, name) ? getproperty(x, name) : default
getfrom(x) = Base.Fix1(get, x)

isin(xs) = x -> x in xs


function integer_labeler(T::DataType)
    N = 0
    DefaultDict{T,Int}() do
        N += 1
        N
    end
end


function print_header(txt; color=:magenta)
    display_width = displaysize(stdout)[2]
    n_fill = fld(display_width - length(txt) - 2, 2)
    n_space = 2
    n_dash = n_fill - n_space
    println()
    print(' '^n_space)
    printstyled('-'^n_dash; color, bold=true)
    print(' ', txt, ' ')
    printstyled('-'^n_dash; color, bold=true)
    print(' '^n_space)
    println()
end

macro catch_missing(expr)
    esc(quote
        try
            $expr
        catch
            missing
        end
    end)
end

function pooled_mean_std(ns::AbstractVector{<:Integer},
                        μs::AbstractVector{<:Number},
                        σs::AbstractVector{<:Number})
    nsum = sum(ns)
    meanc = ns' * μs / nsum
    vs = replace!(σs .^ 2, NaN=>0)
    varc = sum((ns .- 1) .* vs + ns .* abs2.(μs .- meanc)) / (nsum - 1)
    return meanc, .√(varc)
end

function cache(f, file; disable=false, read_only=false, overwrite=false)
    disable && return f()
    !overwrite && isfile(file) && return deserialize(file)
    read_only && error("No cached result $file")
    mkpath(dirname(file))
    result = f()
    serialize(file, result)
    result
end

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return basetype(T)([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function grid(;kws...)
    X = map(Iterators.product(values(kws)...)) do x
        (; zip(keys(kws), x)...)
    end
    KeyedArray(X; kws...)
end

function initialize_keyed(val; keys...)
    KeyedArray(fill(val, (length(v) for (k, v) in keys)...); keys...)
end

function wrap_counts(df::DataFrame; dims...)
    @chain df begin
        groupby(collect(keys(dims)))
        combine(nrow => :n)
        AxisKeys.populate!(initialize_keyed(0.; dims...), _, :n)
    end
end

function wrap_pivot(df::DataFrame, val, f; dims...) 
    @chain df begin
        groupby(collect(keys(dims)))
        combine(val => f => :_val)
        AxisKeys.populate!(initialize_keyed(0.; dims...), _, :_val)
    end
end

macro bywrap(x, what, val, default=missing)
    arg = :(:_val = $val)
    esc(quote
        b = $(DataFramesMeta.by_helper(x, what, arg))
        what_ = $what isa Symbol ? ($what,) : $what
        wrapdims(b, :_val, what_..., sort=true; default=$default)
    end)
end

function keyed(name, xs)
    KeyedArray(xs; Dict(name => xs)...)
end

keymax(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmax(X).I))...)
keymax(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmax(x)]
keymin(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmin(X).I))...)
keymin(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmin(x)]

round1(x) = round(x; digits=1)
round2(x) = round(x; digits=2)
round3(x) = round(x; digits=3)
round4(x) = round(x; digits=4)

fmt(digits, x) = Printf.format(Printf.Format("%.$(digits)f"), x)

function Base.diff(K::KeyedArray; dims, removefirst::Bool=true)
    range = removefirst ? (2:size(K, dims)) : (1:size(K,dims)-1)
    out = similar(selectdim(K, dims, range) )
    out[:] = Base.diff(parent(parent(K)); dims=AxisKeys.dim(parent(K),dims))
    return out
end

Base.dropdims(idx::Union{Symbol,Int}...) = X -> dropdims(X, dims=idx)
squeezify(f) = (X, dims...) -> dropdims(f(X; dims); dims)
smaximum = squeezify(maximum)
sminimum = squeezify(minimum)
smean = squeezify(mean)
ssum = squeezify(sum)

safe_maximum(x; default=missing) = isempty(x) ? default : maximum(x)
safe_maximum(f::Function, x; default=missing) = isempty(x) ? default : maximum(f, x)
safe_minimum(x; default=missing) = isempty(x) ? default : minimum(x)
safe_minimum(f::Function, x; default=missing) = isempty(x) ? default : minimum(f, x)
safe_sum(x; default=missing) = isempty(x) ? default : sum(x)
safe_sum(f::Function, x; default=missing) = isempty(x) ? default : sum(f, x)
safe_mean(x; default=missing) = isempty(x) ? default : mean(x)
safe_mean(f::Function, x; default=missing) = isempty(x) ? default : mean(f, x)
safe_only(xs; default=missing) = isempty(xs) ? default : only(xs)


imap(f, xs...) = map(f, Iterators.countfrom(1), xs...)
flatmap(f, xs...) = mapreduce(f, vcat, xs...)

repeatedly(f, N) = map(i->f(), 1:N)
reduce_repeatedly(f, op, N; kws...) = mapreduce(i->f(), op, 1:N; kws...)
monte_carlo(f, N=10000) = N \ reduce_repeatedly(f, (+), N)

Base.map(f::Function) = xs -> map(f, xs)
Base.reduce(op) = xs -> reduce(op, xs)

linscale(x, low, high) = low + x * (high-low)
logscale(x, low, high) = exp(log(low) + x * (log(high) - log(low)))
unlinscale(x, low, high) = (x - low) / (high-low)
unlogscale(x, low, high) = (log(x) - log(low)) / (log(high) - log(low))

juxt(fs...) = x -> Tuple(f(x) for f in fs)
clip(x, lo, hi) = min(hi, max(lo, x))

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)
normalize(x) = x ./ sum(x)
normalize!(x) = x ./= sum(x)

