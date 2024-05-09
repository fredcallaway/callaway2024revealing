@isdefined(trials) || include("analysis_base.jl")
include("model.jl")

function myopic_value(t)
    monte_carlo(100) do
        value(simulate(MyopicMoves(β_move=10), t))
    end
end

tdf = trialframe(trials; keep_timeouts=true) do t
    sp = subjective_problem(t.problem, Set(premove_fixated(t)))

    (;
        score = value(t),
        optimal = value(t.problem),
        myopic = myopic_value(t),
        random = random_value(t.problem),
        timeout = timeout(t),
        duration = duration(t),
        # fixated_score = value(sp, path(t)),
        fixated_optimal = mean(optimal_paths(sp)) do pth
            value(t.problem, pth)
        end
    )
end
@rput tdf

# %% --------


R"""
tdf %>%
    filter(!timeout) %>%
    with(mean(score == optimal)) %>%
    fmt_percent %>%
    write_tex("optimal_path")

tdf %>%
    filter(!timeout) %>%
    with(sum(optimal))

tdf %>%
    filter(!timeout) %>%
    with(sum(score) / sum(optimal)) %>%
    fmt_percent %>%
    write_tex("score_percent")

tdf %>%
    with(sum(myopic) / sum(optimal)) %>%
    fmt_percent %>%
    write_tex("myopic_score_percent")
"""
