include("utils.jl")
include("problem.jl")
include("data.jl")

using StatsBase

NAME = "cogsci"
versions = ["e2"]

uids = flatmap(versions) do v
    uids = get_uids(v)
    if v == "e1"
        excludes = split("P1 P3 P4 P7 P8 P12 P13 P15 P16 P24 P27 P30 P33")
        global n_exclude_p = length(excludes)
        filter!(uids) do uid
            split(uid, "_")[end] ∉ excludes
        end
    elseif v == "e2"
        global n_exclude_p = 3
        setdiff!(uids, [
            "23-12-15-1419_P29",  # false start
            "23-12-05-1212_P3",  # false start
            "23-12-06-1105_P1",  # calibration
            "23-12-08-1111_P15", # calibration
            "23-12-08-1607_P20", # calibration
        ])
    end
    uids
end;


trials = mapreduce(vcat, uids) do uid
    get_trials(uid)
end;

@assert all(uids) do u
    length(get_trials(u)) > 90
end
# n_trial = countmap(get.(trials, :pid))
# drop = filter(keys(n_trial)) do pid
#     n_trial[pid] < 80
# end
# filter!(trials) do t
#     t.pid ∉ drop
# end

@assert all(trials) do t
    t.gaze_contingent
end
# filter!(trials) do t
#     t.gaze_contingent
# end


PID_LABELS = map(unique(t->t.pid, trials)) do t
    t.pid => t.uid
end |> Dict

# # exclude timeouts
# trials_with_timeout = copy(trials)
# filter!(!timeout, trials)

gtrials = group(t->t.pid, trials)
@info string(length(gtrials), " participants")

fig(f, name; kws...) = figure(f, "$NAME-$name"; kws...)

using RCall
@rput NAME
R"""
suppressPackageStartupMessages(source("base.r"))
FIGS_PATH = glue("figs/{NAME}/")
STATS_PATH = glue("stats/{NAME}/")
MAKE_PDF = TRUE
WRITE_REG = TRUE
"""

function trialframe(f, trials; keep_timeouts=false)
    map(trials) do t
        timeout(t) && !keep_timeouts && return missing
        row = f(t)
        ismissing(row) && return missing
        agent = startswith(t.pid, "P") ? "human" : t.pid
        (;agent, t.pid, t.trial_index, timeout = timeout(t), named_tuple(row)...)
    end |> skipmissing |> collect |> DataFrame
end

function multiframe(f, trials; keep_timeouts=false)
    flatmap(trials) do t
        timeout(t) && !keep_timeouts && return []
        rows = f(t)
        ismissing(rows) && return []
        map(rows) do row
            ismissing(row) && return missing
            agent = startswith(t.pid, "P") ? "human" : t.pid
            (;agent, t.pid, t.trial_index, named_tuple(row)...)
        end
    end |> skipmissing |> collect |> DataFrame
end

function ppc_frame(func, human_trials, model_trials)
    vcat(
        @transform!(func(human_trials), :agent = "human"),
        @transform!(func(model_trials), :agent = "model"),
    )
end

function is_chain(graph, states)
    all(2:length(states)) do i
        states[i] in graph[states[i-1]] &&  # connected
        (i == 2 || states[i] != states[i-2])  # not going backward
    end
end

function maximal_chain(graph, states)
    chain = [states[end]]
    for s in reverse(states[1:end-1])
        if s in chain
            break  # don't count loops
        end
        if s in graph[chain[end]]
            push!(chain, s)
        else
            break
        end
    end
    reverse(chain)
end

maximal_chain(problem::Problem, states) = maximal_chain(problem.graph, states)

function skip_repeats(xs)
    ys = empty(xs)
    for x in xs
        if isempty(ys) || x != ys[end]
            push!(ys, x)
        end
    end
    ys
end
