using Distributions
using JSON
using Random

struct Problem
    graph::Vector{Vector{Int}}
    rewards::Vector{Float64}
    start::Int
    n_steps::Int
    Problem(graph, rewards, start, n_steps) = begin
        if rewards[start] != 0
            error("rewards[start] is $(rewards[start])")
        end
        new(graph, rewards, start, n_steps)
    end
end
nstate(p::Problem) = length(p.rewards)
children(p::Problem, s) = p.graph[s]
reward(p::Problem, s) = p.rewards[s]
initial_state(p::Problem) = p.start

function descendants(p::Problem, s)
    res = Int[]
    for c in children(p, s)
        push!(res, c)
        push!(res, descendants(p, c)...)
    end
    res
end

function parents(p::Problem, s)
    findall(p.graph) do s0_out
        s in s0_out
    end
end

Base.parent(p::Problem, s) = safe_only(parents(p, s))

is_terminal(p::Problem, s) = isempty(children(p, s))
terminal_states(p::Problem) = filter(is_terminal, states(p))

function reachable(p::Problem, s=p.start)
    result = Set{State}()
    frontier = Set(s)
    while !isempty(frontier)
        s = pop!(frontier)
        push!(result, s)
        for s1 in children(p, s)
            if s1 ∉ result
                push!(frontier, s1)
            end
        end
    end
    result
end

path(p::Problem, s) = only(paths(p, s))
function path_type(problem::Problem, a::Int, b::Int)
    apath = path(problem, a)
    bpath = path(problem, b)
    n_back = length(setdiff(apath, bpath))
    n_forward = length(setdiff(bpath, apath))
    (n_back, n_forward)
end
adjacent(prob::Problem, s) = mod1(s+1, nstate(prob)), mod1(s-1, nstate(prob))


# converts to 0 indexing
function JSON.lower(problem::Problem)
    (;
        graph = map(x -> x .- 1, problem.graph),
        problem.rewards,
        start = problem.start - 1,
        problem.n_steps,
        # value = value(problem)
    )
end

states(p::Problem) = eachindex(p.graph)

Base.Broadcast.broadcastable(x::Problem) = Ref(x)

function paths(problem::Problem, goal::Int = -1; n_steps=problem.n_steps)
    frontier = [[]]
    result = Vector{Int}[]

    function search!(path)
        s = isempty(path) ? problem.start : path[end]
        if s == goal
           push!(result, path)
       end
        if length(path) == n_steps || isempty(children(problem, s))
            goal == -1 && push!(result, path)
            return
        end
        for child in children(problem, s)
            push!(frontier, [path; child])
        end
    end
    while !isempty(frontier)
        search!(pop!(frontier))
    end
    result
end

function value(problem::Problem, path)
    isempty(path) && return 0
    sum(unique(path)) do s
        reward(problem, s)
    end
end

function value(problem::Problem)
    maximum(paths(problem)) do pth
        value(problem, pth)
    end
end

function depth(problem, s)
    pths = paths(problem, s)
    isempty(pths) && return missing
    minimum(length, pths)
end

function shortest_path(problem::Problem, s::Int)
    error("this doesn't make sense")
    pths = paths(problem, s)
    isempty(pths) && return missing
    argmin(length, pths)
end

function shortest_paths(problem::Problem, s::Int)
    pths = paths(problem, s)
    isempty(pths) && return pths
    len = minimum(length, pths)
    filter(pths) do pth
        length(pth) == len
    end
end

function action_values(problem::Problem; kws...)
    Q = fill(NaN, length(states(problem)))
    action_values!(Q, problem; kws...)
end

function path_value(problem, s::Int, rewards=problem.rewards)
    safe_maximum(paths(problem, s); default=missing) do pth
        sum(rewards[pth])
    end
end

function action_values!(Q, problem::Problem, rewards=problem.rewards; state=problem.start, γ=1.)
    Q .= NaN
    function rec(s)
        if isnan(Q[s])
            Q[s] = rewards[s] + γ * safe_maximum(rec, children(problem, s); default=0)
        end
        Q[s]
    end
    rec(state)
    Q
end

function expected_value(problem::Problem, seen, path)
    sum(unique(path)) do s
        s in seen ? reward(problem, s) : 0
    end
end

function expected_value(problem::Problem, seen)
    maximum(paths(problem)) do path
        expected_value(problem, seen, path)
    end
end

function subjective_problem(problem::Problem, seen::BitVector; start=problem.start, unknown_reward=0.)
    rewards = imap(problem.rewards) do i, r
        i == start && return 0.
        seen[i] ? r : unknown_reward
    end
    mutate(problem; rewards, start)
end

function subjective_problem(problem::Problem, seen::Set; start=problem.start, unknown_reward=0.)
    rewards = imap(problem.rewards) do i, r
        i == start && return 0.
        i in seen ? r : unknown_reward
    end
    mutate(problem; rewards, start)
end

function optimal_paths(problem::Problem)
    pths = paths(problem)
    v = value.(problem, pths)
    idx = findall(isequal(maximum(v)), v)
    pths[idx]
end