using JSON
using NamedTupleTools
using Memoize
using Glob

struct Fixation
    time::Float64
    end_time::Float64
    x::Float64
    y::Float64
end
duration(f::Fixation) = f.end_time - f.time

struct Gaze
    time::Float64
    x::Float64
    y::Float64
    fixation::Bool
end

struct Mouse
    time::Float64
    x::Float64
    y::Float64
end

abstract type Trial end
abstract type Action end
abstract type Move <: Action end
abstract type Look <: Action end

function Base.show(io::IO, t::T) where T <: Action
    print(io, "$T($(t.state))")
end

function Base.show(io::IO, t::T) where T <: Trial
    print(io, "$T($(t.pid), $(t.trial_index))")
end
# (H)uman and (S)imulated versions

struct HMove <: Move
    state::Int
    time::Float64
end
struct SMove <: Move
    state::Int
end

struct HLook <: Look
    state::Int
    time::Float64
    end_time::Float64
    move::Bool
    fixations::Vector{Fixation}
end
struct SLook <: Look
    state::Int
    move::Bool
end
SLook(s) = SLook(s, false)

duration(l::HLook) = l.end_time - l.time
is_move(l::Look) = l.move

struct HTrial <: Trial
    pid::String
    uid::String
    start_time::Float64
    trial_index::Int
    problem::Problem
    actions::Vector{Action}
    gaze_contingent::Bool
    time_limit::Int
    layout::Vector{Tuple{Float64, Float64}}
end

struct STrial <: Trial
    pid::String
    trial_index::Int
    problem::Problem
    actions::Vector{Action}
end

trial_id(t) = "$(t.pid)-$(t.trial_index)"

actions(t::Trial) = t.actions
moves(t::Trial)::Vector{Move} = filter(a -> a isa Move, t.actions)
path(t::Trial) = moves(t) .|> get(:state)
looks(t::Trial)::Vector{Look} = filter(a -> a isa Look, t.actions)
looked(t::Trial)::Vector{Int} = looks(t) .|> get(:state)
seen(t::Trial) = Set(looked(t))

value(t::Trial) = value(t.problem, path(t))
relative_score(t::Trial; normalize=true) = relative_score(t.problem, path(t))

fixated(t::HTrial) = filter(a -> a isa Look && !isempty(a.fixations), t.actions) .|> get(:state)
get_fixations(t::HTrial) = get_fixations(t.uid, t.start_time, end_time(t))
duration(t::HTrial) = moves(t)[end].time - t.start_time
first_move_time(t::HTrial) = moves(t)[1].time - t.start_time
timeout(t::HTrial) = moves(t)[end].time > t.start_time + t.time_limit
timeout(t::STrial) = false
end_time(t::Trial) = moves(t)[end].time

function premove_looks(t::Trial; keep_move=false)::Vector{Look}
    Iterators.takewhile(actions(t)) do a
        a isa Look && (keep_move || !is_move(a))
    end |> collect
end
premove_looked(t::Trial; keep_move=false)::Vector{Int} = premove_looks(t; keep_move) .|> get(:state)

nonmove_looked(t::Trial)::Vector{Int} = filter(!is_move, looks(t)) .|> get(:state)

function premove_fixated(t::HTrial)::Vector{Int}
    filter!(premove_looks(t)) do l
        !isempty(l.fixations)
    end .|> get(:state)
end

include("parse_eyetracking.jl")
include("parse_trialdata.jl")

function get_experiment_data(uid)
    JSON.parsefile(only(glob("data/task/*/$uid.json")));
end

@memoize function get_trial_data(uid)
    cache("data/processed/$uid/trial_data") do
        preprocess!(get_experiment_data(uid)["trial_data"], uid)
    end
end

@memoize function get_practice_data(uid)
    cache("data/processed/$uid/practice_data") do
        preprocess!(get_experiment_data(uid)["practice_data"], uid)
    end
end

@memoize function get_eye_data(uid)
    cache("data/processed/$uid/eye_data") do
        parse_eyedata(uid)
    end
end

function get_gaze(t::HTrial)
    filter(get_eye_data(t.uid).gaze) do fix
        end_trial = actions(t)[end]
        t.start_time < fix.time < end_trial.time + 1
    end
end

@memoize function get_trials(uid)
    cache("data/processed/$uid/trials") do
        parse_trials(get_trial_data(uid))
    end
end

function get_uids(version)
    uids = @chain readdir("data/task/$version") begin
        filter(_) do p
            !occursin("tes", p) && !occursin(".txt", p)
        end
        map(p -> p[1:end-5], _)
    end
end

function get_fixations(uid, start, stop)
    filter(get_eye_data(uid).fixations) do fix
        start ≤ fix.time ≤ stop
    end
end


function height2pix((w, h), (x, y))
    # scale and invert y
    y *= -h
    x *= h

    # center
    x += w/2
    y +=  h/2

    return x, y
end

@memoize function get_mouse_data(uid)
    win = Int.(get_experiment_data(uid)["window"]) ./ 2  # eyetracker uses non-retina pixels
    map(get_trial_data(uid)) do d
        trial_id(d) => map(d.mouse, d.flips) do (x, y), t
            x, y = height2pix(win, (x, y))
            Mouse(t, x, y)
        end
    end |> Dict
end

function get_mouse_data(t::HTrial)
    get_mouse_data(t.uid)[trial_id(t)]
end

function expand_mouse(md::Vector{Mouse}, gd::Vector{Gaze})
    res = Mouse[]
    gt = get.(gd, :time)
    i = 1
    for t in gt
        while md[i].time < t && i < length(md)
            i += 1
        end
        push!(res, md[i])
    end
    res
end
