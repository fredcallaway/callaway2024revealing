
struct FullyRandom <: Model end
initial_state(::FullyRandom, problem::Problem) = nothing
transition!(::FullyRandom, problem::Problem, m::Nothing, a) = nothing

move_value(model::FullyRandom, problem::Problem, m::Nothing, s::State, target::State) = 0.
termination_value(model::FullyRandom, problem::Problem, m::Nothing, s::State) = 0.
fixate_value(model::FullyRandom, problem, m, s, target) = 0.


struct RandomLooks <: Model
    p_stop::Float64
end

initial_state(model::RandomLooks, problem::Problem) = nothing
transition!(model::RandomLooks, problem, m, a) = nothing
termination_probability(model::RandomLooks, problem, m, s) = model.p_stop

move_value(model::RandomLooks, problem, m, s, target) = 0.

# function possible_fixate(model::RandomLooks, problem, m, s) = setdiff()
#     if m.state == 0 || isempty(children(problem, m.state))
#         [problem.start]
#     else
#         children(problem, m.state)
#     end
# end

fixate_value(model::RandomLooks, problem, m, s, target) = 0.

struct RandomMoves <: Model end

initial_state(::RandomMoves, problem::Problem) = nothing
transition!(::RandomMoves, problem::Problem, m::Nothing, a) = nothing

move_value(model::RandomMoves, problem::Problem, m::Nothing, s::State, target::State) = 0
termination_probability(model::RandomMoves, problem::Problem, m::Nothing, s::State) = 1
fixate_probability(model::RandomMoves, problem, m, s, target) = error("Not implemented")

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


@kwdef struct SoftmaxMoves{A,B} <: Model
    β_move::A = 1.
    ε_move::B = 0.
end

function initial_state(::SoftmaxMoves, problem::Problem)
    action_values(problem)
end

transition!(::SoftmaxMoves, problem::Problem, m, a::Action) = m

termination_probability(::SoftmaxMoves, _...) = 1.
fixate_probability(::SoftmaxMoves, _...) = error("Not implemented")
move_value(::SoftmaxMoves, problem::Problem, m, s::State, target::State) = m[target]


@kwdef struct DiscountMoves{A,B,C} <: Model
    β_move::A = 1.
    ε_move::B = 0.
    γ::C = 1.
end

function initial_state(model::DiscountMoves, problem::Problem)
    action_values(problem; model.γ)
end

transition!(::DiscountMoves, problem::Problem, m, a::Action) = m

# transition!(::DiscountMoves, problem::Problem, m, a::Look) = m
# function transition!(::DiscountMoves, problem::Problem, m, a::Move)
#     action_values!(m, problem; a.state, model.γ, )
# end

termination_probability(::DiscountMoves, _...) = 1.
fixate_probability(::DiscountMoves, _...) = error("Not implemented")
move_value(::DiscountMoves, problem::Problem, m, s::State, target::State) = m[target]


@kwdef struct MyopicMoves{A,B,C} <: Model
    β_move::A = 1.
    ε_move::B = 0.
    γ::C = 1.
end

initial_state(model::MyopicMoves, problem::Problem) = nothing
transition!(::MyopicMoves, problem::Problem, m, a::Action) = m

termination_probability(::MyopicMoves, _...) = 1.
fixate_probability(::MyopicMoves, _...) = error("Not implemented")
move_value(::MyopicMoves, problem::Problem, m, s::State, target::State) = problem.rewards[target]
