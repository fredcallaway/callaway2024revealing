
"""
Abstract base class for likelihood models.
Must define the following methods:

    initial_state(model, problem)
    transition!(model, problem, m, a::Action)

    termination_probability(model, problem, m, s)
    move_probability(model, problem, m, s, target)
    fixate_probability(model, problem, m, s, target)

    Instead of defining *_probability directly, one can also define *_value
    along with ε_* and β_* for the noise model.
"""
abstract type Model end
Base.Broadcast.broadcastable(x::Model) = Ref(x)
State = Int

function transition(p::Problem, s::State, a::Move)
    @assert a.state in children(p, s)
    a.state
end
transition(::Problem, s::State, ::Look) = s
function transition(model::Model, problem::Problem, m, a)
    m = deepcopy(m)
    transition!(model, problem, m, a)
    m
end

initial_state(model::Model, t::Trial) = initial_state(model, t.problem)

function trace(model::Model, t::Trial; mode=:full)
    m = initial_state(model, t)
    s = t.problem.start
    map(actions(t)) do a
        l = likelihood(model, t.problem, m, s, a; mode)
        x = (;l, s, m, a)
        s = transition(t.problem, s, a)
        m = transition(model, t.problem, m, a)
        x
    end
end

function log_likelihood(model::Model, t::Trial; mode=:full)
    m = initial_state(model, t)
    s = t.problem.start
    sum(actions(t)) do a
        l = likelihood(model, t.problem, m, s, a; mode)
        s = transition(t.problem, s, a)
        transition!(model, t.problem, m, a)
        log(l)
    end
end

function log_likelihood(model::Model, trials::Vector; mode=:full)
    sum(trials) do t
        log_likelihood(model, t; mode)
    end
end

log_likelihood(::Model, ::Nothing; mode=:full) = NaN  # for generated_quantities


function likelihood(model::Model, problem::Problem, m, s::State, a::Action; mode=:full)
    if mode == :looks
        a isa Move && return 1.
        p_term = 0.
    elseif mode == :moves
        a isa Look && return 1.
        p_term = 1.
    else
        p_term = termination_probability(model, problem, m, s)
    end
    if a isa Move
        p_term * move_probability(model, problem, m, s, a.state)
    else
        (1 - p_term) * fixate_probability(model, problem, m, s, a.state)
    end
end

function move_probability(model, problem, m, s, target)
    choices = children(problem, s)
    i_target = findfirst(isequal(target), choices)
    isnothing(i_target) && error("Illegal move encountered in move_probability")
    ε = getprop(model, :ε_move, 0.)
    β = getprop(model, :β_move, 1.)

    lapse_softmax(choices, i_target; ε, β) do target
        move_value(model, problem, m, s, target)
    end
end

function sample_move(model, problem, m, s)
    choices = children(problem, s)
    ε = getprop(model, :ε_move, 0.)
    β = getprop(model, :β_move, 1.)
    sample_lapse_softmax(choices; ε, β) do target
        move_value(model, problem, m, s, target)
    end |> SMove
end

function termination_probability(model, problem, m, s)
    ε = getprop(model, :ε_term, 0.)
    β = getprop(model, :β_term, 1.)
    lapse(logistic(β * termination_value(model, problem, m, s)), ε, 2)
end

sample_terminate(model, problem, m, s) = rand(Bernoulli(termination_probability(model, problem, m, s)))


function fixate_probability(model, problem, m, s, target)
    ε = getprop(model, :ε_fixate, 0.)
    β = getprop(model, :β_fixate, 1.)
    lapse_softmax(states(problem), target; ε, β) do target
        fixate_value(model, problem, m, s, target)
    end
end

possible_fixate(model, problem, m, s) = states(problem)

function sample_fixate(model, problem, m, s)
    ε = getprop(model, :ε_fixate, 0.)
    β = getprop(model, :β_fixate, 1.)
    sample_lapse_softmax(possible_fixate(model, problem, m, s); ε, β) do target
        fixate_value(model, problem, m, s, target)
    end |> SLook
end

function sample_action(model::Model, problem, m, s)
    if sample_terminate(model, problem, m, s)
        sample_move(model, problem, m, s)
    else
        sample_fixate(model, problem, m, s)
    end
end

function simulate(model::Model, problem::Problem, conditioning_acts::Nothing; pid="P0", trial_index=0)
    s = initial_state(problem)
    m = initial_state(model, problem)
    acts = Action[]
    while !is_terminal(problem, s)
        a = sample_action(model, problem, m, s)
        # println(m, "  ", a); sleep(.3)
        if a isa Move && !isempty(acts) && acts[end] isa Look && acts[end].state == a.state
            acts[end] = mutate(acts[end], move=true)
        end
        push!(acts, a)
        transition!(model, problem, m, a)
        s = transition(problem, s, a)
    end
    STrial(pid, trial_index, problem, acts)
end

function simulate(model::Model, problem::Problem, conditioning_acts::Vector{<:Action}; pid="P0", trial_index=0)
    s = initial_state(problem)
    m = initial_state(model, problem)
    acts = Action[]
    for ca in conditioning_acts
        if ca isa Move
            a = sample_action(model, problem, m, s)
            @assert a isa Move
        else
            a = ca
        end
        # println(m, "  ", a); sleep(.3)
        push!(acts, a)
        transition!(model, problem, m, a)
        s = transition(problem, s, a)
        is_terminal(problem, s) && break
    end
    STrial(pid, trial_index, problem, acts)
end


function simulate(model::Model, t::Trial; mode=:full, kws...)
    @assert mode != :looks
    conditioning_acts = mode == :moves ? actions(t) : nothing
    simulate(model, t.problem, conditioning_acts; pid=t.pid, trial_index=t.trial_index, kws...)
end

function simulate(model::Model, problems::Vector; pid="P0")
    imap(problems) do trial_index, problem
        simulate(model, problem; pid, trial_index)
    end
end

function simulate_many(trials, models...; add_human=true, repeats=1)
    sim_trials = mapreduce(vcat, first.(models), last.(models)) do pid, model
        simulate(model, repeat(trials, repeats); pid)
    end
    if add_human
        vcat(sim_trials, trials)
    else
        sim_trials
    end
end


include("baseline_models.jl")
