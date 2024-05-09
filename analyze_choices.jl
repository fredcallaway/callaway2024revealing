@isdefined(trials) || include("analysis_base.jl")

function choice_frame(trials)
    multiframe(trials) do t
        n_look = zeros(Int, nstate(t.problem))
        n_look_full = zeros(Int, nstate(t.problem))
        n_look_premove = zeros(Int, nstate(t.problem))
        state = t.problem.start
        actual_value = action_values(t.problem)
        seen = falses(nstate(t.problem))
        moved = false
        desc = map(1:nstate(t.problem)) do s
            descendants(t.problem, s)
        end
        n_desc = map(length, desc)

        imap(actions(t)) do i, a
            if a isa Look
                seen[a.state] = true
                n_look_full[a.state] += 1
                if !moved
                    n_look_premove[a.state] += 1
                end
                if !a.move
                    n_look[a.state] += 1
                end
                return missing
            else
                seen_reward = t.problem.rewards .* seen
                seen_value = action_values(mutate(t.problem; rewards = seen_reward))
                c1, c2 = children(t.problem, state)
                state = a.state

                first_move = !moved
                moved = true
                chosen, unchosen = a.state == c2 ? (c2, c1) : (c1, c2)

                function feature(name, vals)
                    Dict(
                        name * "1" => vals[c1],
                        name * "2" => vals[c2],
                        name * "_rel" => vals[c2] - vals[c1],
                        name * "_chosen" => vals[chosen],
                        name * "_unchosen" => vals[unchosen],
                        name * "_rel_chosen" => vals[chosen] - vals[unchosen],
                    ) |> namedtuple
                end
                n_look_desc = map(1:nstate(t.problem)) do s
                    sum(n_look[desc[s]])
                end

                last_fixated = 0
                if i > 1 && actions(t)[i-1] isa Look
                    if actions(t)[i-1].state == c2
                        last_fixated = 1
                    elseif actions(t)[i-1].state == c1
                        last_fixated = -1
                    end
                end

                (;
                    first_move,
                    last_fixated,
                    elapsed = a isa SMove ? missing : a.time - t.start_time,
                    depth = depth(t.problem, a.state),
                    choice = a.state == c2,
                    feature("true_reward", t.problem.rewards)...,
                    feature("true_value", actual_value)...,
                    feature("seen_reward", seen_reward)...,
                    feature("seen_value", seen_value)...,
                    feature("n_look", n_look)...,
                    feature("n_look_full", n_look_full)...,
                    feature("n_look_premove", n_look_premove)...,
                    feature("seen", seen)...,
                    feature("seen_before", n_look .> 0)...,
                    feature("n_desc", n_desc)...,
                    feature("terminal", n_desc .== 0)...,
                    feature("n_look_desc", n_look_desc)...,
                )
            end
        end
    end
end

df = choice_frame(trials)
@rput df

# %% ==================== performance ====================

R"""
report_percent(df, "choice_accuracy", true_value_rel_chosen >= 0)
report_percent(df, "choice_inaccuracy", true_value_rel_chosen < 0)
report_percent(df, "choice_accuracy_reward", true_reward_rel_chosen >= 0)
report_percent(df, "choice_accuracy_chance", if_else(true_value_rel == 0, 1, 0.5))

model = df %>%
    mutate(future_value_rel = true_value_rel - true_reward_rel) %>%
    regress(choice ~ true_reward_rel + future_value_rel, logistic=T, mixed=T, name="choice-reward-value")

model %>% coef_tibble(wide=T) %>% report_percent("choice_discounting", future_value_rel / true_reward_rel)

"""

# %% ==================== discounting ====================

R"""
report_percent(df, "choice_inaccuracy_seen", seen_value_rel_chosen < 0)

model = df %>%
    mutate(future_value_rel = seen_value_rel - seen_reward_rel) %>%
    regress(choice ~ seen_reward_rel + future_value_rel, logistic=T, mixed=T, name="choice-reward-value-seen")

model %>% coef_tibble(wide=T) %>% report_percent("choice_discounting_seen", future_value_rel / seen_reward_rel)
"""

# %% ==================== reward-fixation interaction (evidence accumulation) ====================

R"""
choice_plot = df %>%
    filter(n_look_full2 < 4) %>%
    ggplot(aes(n_look_full2, 1*choice, color=factor(true_reward2))) +
    point_line(min_n=10, position=position_dodge(width=.3)) +
    zissou_pal +
    labs(x="Number of Fixations", y="Choice Probability", color="Reward")

choice_plot +
    theme(legend.position=c(1.25,0.47), plot.margin = unit(c(6,65,6,6), "points"))

fig("choice-interaction")
"""


R"""
model = df %>%
    filter(seen1, seen2) %>%
    mutate(
        weighted_reward_rel = n_look2 * true_reward2 - n_look1 * true_reward1,
    ) %>%
    regress(choice ~ true_reward_rel + weighted_reward_rel + n_look_rel,
        logistic=T, mixed=T, intercept=F, name="choice-interaction-full")

"""
