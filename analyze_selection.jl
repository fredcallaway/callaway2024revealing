@isdefined(trials) || include("analysis_base.jl")

rel_names = Dict(
    # (0, 0) => "repeat",
    (0, 1) => "child",
    (1, 0) => "parent",
    (1, 1) => "sibling",
)
function get_relationship(n_back, n_forward)
    get(rel_names, (n_back, n_forward), string(n_back, "-", n_forward))
end

function get_relationship_simple(n_back, n_forward)
    if (n_back, n_forward) in keys(rel_names)
        rel_names[(n_back, n_forward)]
    # elseif n_forward == 1
    #     "aunt"
    else
        "other"
    end
end

df = multiframe(trials) do t
    n_fix = zeros(Int, nstate(t.problem))
    # @require length(looks_) > 1
    fix_number = 1
    move_number = 1
    current_state = t.problem.start
    prev_fix = 0
    flatmap(actions(t)) do a
        if a isa Move
            move_number += 1
            current_state = a.state
            return missing
        end
        @assert a isa Look
        this_fix = a.state
        if this_fix == prev_fix
            return missing  # treat contiguous repeats as one fixation
        end

        if prev_fix == 0
            res = missing
        else
            subj_problem = subjective_problem(t.problem, n_fix .> 0; start=current_state)
            q = action_values(subj_problem)
            prev_reward = reward(subj_problem, prev_fix)
            prev_depth = depth(t.problem, prev_fix)
            prev_relative_depth = let
                nb, nf = path_type(t.problem, current_state, prev_fix)
                nb == 0 ? nf : missing
            end
            if ismissing(prev_relative_depth)
                prev_reward = 0.
            end
            prev_path_value = path_value(subj_problem, prev_fix)
            current_path_value = path_value(subj_problem, current_state)

            res = map(1:nstate(t.problem)) do s
                @require s ≠ prev_fix
                n_back, n_forward = path_type(t.problem, prev_fix, s)
                relative_depth = let
                    nb, nf = path_type(t.problem, current_state, s)
                    nb == 0 ? nf : missing
                end
                (;
                    fix_number, move_number,
                    fixated = s == this_fix,
                    adjacent = s in adjacent(t.problem, prev_fix),
                    move_fix = a.move,
                    n_back, n_forward,
                    relationship = get_relationship(n_back, n_forward),
                    relationship_simple = get_relationship_simple(n_back, n_forward),
                    depth = depth(t.problem, s),  # SHOULD THIS BE SUBJECTIVE?
                    relative_depth,
                    reward = reward(subj_problem, s),
                    n_fix = n_fix[s],
                    path_value = path_value(subj_problem, s),
                    has_children = !isempty(children(t.problem, s)),
                    # n_fix_children = sum(n_fix[children(t.problem, s)]),
                    n_unseen_children = sum(n_fix[children(t.problem, s)] .== 0),
                    action_value = q[s],
                    prev_n_fix = n_fix[prev_fix],
                    prev_reward,
                    prev_depth,
                    prev_path_value,
                    prev_relative_depth,
                    current_path_value,
                )
            end
            @assert sum(get.(skipmissing(res), :fixated)) == 1
        end
        n_fix[this_fix] += 1
        prev_fix = this_fix
        fix_number += 1
        res
    end
end


full_df = df
@rput full_df

# %% ==================== exclusions ====================

R"""

report_n_percent = function(data, name, var) {
    data %>%
        summarise(pct=fmt_percent(mean({{var}})), total=sum({{var}})) %>%
        with(write_tex("{total} ({pct})", name))
}

exclude = function(data, name, cond, report_cond=T) {
    data %>% filter({{report_cond}}) %>%  report_percent(glue("exclude/{name}"), {{cond}})
    data %>% filter(!{{cond}})
}

df = full_df %>%
    # exclude("useless", is.na(relative_depth), fixated) %>%
    # exclude("prev_useless", is.na(prev_relative_depth), fixated) %>%
    exclude("move_fix", move_fix, fixated) %>%
    # fully exclude trials where we excluded the human fixation
    filter(sum(fixated) == 1, .by=c(pid, trial_index, fix_number))
"""

# %% ==================== SACCADE TYPES ====================

R"""
calculate_proportions = function(data, ...) {
    data %>%
    filter(sum(fixated) >= 100) %>%
    count(fixated, ...) %>%
    pivot_wider(names_from=fixated, values_from=n) %>%
    mutate(
        prop = `TRUE` / sum(`TRUE`),
        baseline = (`FALSE` + `TRUE`) / 10,
        relative = `TRUE` / baseline
    )
}

df %>%
    calculate_proportions(relationship) %>%
    filter(`TRUE` >  1.05 * baseline) %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{fmt_percent(prop)}", "transitions/{relationship}")
    ))
"""

R"""
df %>%
    group_by(relationship) %>%
    group_modify(function(data, grp) {
        prop_test(data, fixated ~ NULL, p = 1/10, alternative="greater")
    }) %>%
    filter(p_value < .05) %>%
    mutate(p = pval(p_value))
"""

# %% ==================== DEPTH ====================

depths = multiframe(trials) do t
    (depth = 0:5,
     n_fix = counts(Int.(depth.(t.problem, premove_looked(t))), 0:5),
     baseline = counts(depth.(t.problem, 1:nstate(t.problem)), 0:5)) |> invert
end

@rput depths

R"""
depths %>%
   summarise(y=mean(n_fix / baseline, na.rm=T), .by=depth) %>%
   rowwise() %>% group_walk(~ with(.x,
       write_tex("{y:.2}", "depths/{depth}")
   ))
"""

R"""
chance = depths %>%
    # group_by(agent,pid,trial_index) %>%
    summarise(across(c(n_fix, baseline), sum)) %>%
    with(n_fix / baseline)

write_tex("{chance:.2}", "depths_chance")
"""


R"""
depth_pal = concrete_palette(teals_pal(rev=F, drop=F), sort(unique(depths$depth)))

depths %>%
    ggplot(aes(depth, n_fix / baseline, fill=factor(depth))) +
    geom_hline(yintercept=chance) +
# geom_hline(yintercept=1) +
    bars() +
    depth_pal +
    ylab("Fixations per State") +
    no_legend

fig("depths", w=HEIGHT, h=HEIGHT)
"""


# %% ==================== CONTINUE OR SWITCH ===================

R"""
WIDTH = 5.2
HEIGHT = 1.95

continue_data = df %>%
    filter(relationship == "child") %>%
    group_by(pid, trial_index, fix_number) %>%
    summarise(
        move_fix = first(move_fix),
        best_child_path_value = max(path_value),
        best_child_action_value = max(action_value),
        best_child_reward = max(reward),
        min_n_fix = min(n_fix),
        sum_n_fix = sum(n_fix),
        sum_unseen = sum(n_fix==0),
        any_unseen = max(n_fix==0),
        fixate_child = sum(fixated) == 1,
        n_fix = first(prev_n_fix),
        depth = first(prev_relative_depth),
        reward = first(prev_reward),
        path_value = first(prev_path_value),
        prev_value = path_value - reward
    ) %>%
    ungroup()

plot_continue = function(data, var, ...) {
    data %>%
        ungroup() %>%
        ggplot(aes({{var}}, 1*fixate_child, ...)) +
        expand_limits(y = c(0.5, 1)) +
        ylab("Prob Fixate Child")
}

"""
R"""
stop_reward = continue_data %>%
    filter(depth != 0) %>%
    plot_continue(reward) +
    points() +
    gam_fit(k=8)

continue_data %>% filter(n_fix == 0)

stop_nfix = continue_data %>%
    # filter(depth != 0) %>%
    plot_continue(n_fix) +
    points(min_n=10) +
    xlim(1, 7) +
    gam_fit(k=7)

stop_prev = continue_data %>%
    filter(depth != 0) %>%
    plot_continue(prev_value) +
    point_bin(7, min_n=10) +
    gam_fit() +
    xlab("Previous Rewards")

stop_future = continue_data %>%
    filter(depth != 0) %>%
    plot_continue(future_value) +
    point_bin(7, min_n=10) +
    xlab("Future Rewards")

stop_depth = continue_data %>%
    plot_continue(depth, group=0) +
    mean_line(min_n=5, mapping=aes(group=pid), alpha=0.2, linewidth=.2) +
    gam_fit(k=5) +
    points() +
    # mean_line(color="gray") +
    # points(mapping=aes(color=factor(depth))) +
    depth_pal + no_legend

stop_unseen = continue_data %>%
    filter(!is.na(depth)) %>%
    plot_continue(sum_unseen, color=factor(depth)) +
    point_line(min_n=10) +
    depth_pal +
    scale_x_continuous(breaks=c(0,1,2)) +
    xlab("Unseen Children") + no_legend

wrap_plots(stop_reward, stop_depth+no_yaxis) +
    plot_annotation(tag_levels = 'A') &
    coord_cartesian(xlim=c(NULL), ylim=c(0., 1.)) &
    tight_margin() &
    ylab("Fixate a Child")

fig("continue", w=3.6)
"""

R"""
continue_model = continue_data %>%
    regress(fixate_child ~ reward + depth + prev_value + best_child_action_value,
            logistic=T, mixed=T, name="stop")
"""

# %% ==================== WHICH CHILD ====================

R"""
child_data = df %>%
    filter(relationship == "child") %>%
    group_by(pid, trial_index, fix_number) %>%
    filter(sum(fixated) == 1) %>%
    mutate(seen = n_fix > 0) %>%
    transmute(
        fixated,
        seen_type = case_match(paste0(first(seen), last(seen)),
            "TRUETRUE" ~ "both",
            "FALSEFALSE" ~ "neither",
            "TRUEFALSE" ~ "second",  # note it's reverse coded bc diff/tail
            "FALSETRUE" ~ "first"
        ),
        across(c(reward, path_value, action_value, n_fix, seen, has_children, n_unseen_children), diff)
    ) %>%
    slice_tail(n=1) %>%
    numerize(fixated) %>%
    mutate(future_value = action_value - reward)


df %>% tibble %>%
    filter(pid == "P01", trial_index==62, fix_number==6) %>%
    filter(relationship == "child")


plot_child = function(var, cond=T, ...) {
    child_data %>%
        ungroup() %>%
        filter({{cond}}) %>%
        ggplot(aes({{var}}, fixated, ...)) +
        ylab("Fixation Probability") +
        xlab(glue("Relative\n{ensym(var)}")) +
        ylim(0, 1)
}
"""
R"""
child_reward = plot_child(reward) +
    # gam_fit_ind(k=5) +
    gam_fit() +
    point_bin(bins=7, min_n=10) +
    xlab("Relative \nReward")

child_future = plot_child(future_value) +
    # gam_fit_ind(k=5) +
    gam_fit() +
    point_bin(bins=7, min_n=10) +
    xlab("Relative \nFuture Rewards")

child_value = plot_child(action_value) +
    # gam_fit_ind(k=5) +
    gam_fit() +
    point_bin(bins=7, min_n=10) +
    xlab("Relative \nAction Value")

child_fix = plot_child(n_fix) +
    gam_fit(k=7) +
    points(min_n=10) +
    xlim(-4, 4) +
    xlab("Relative\n# Fixations")

SEEN = "#238dbe"
BOTH = "#926aa7"
SECOND = "#d04848"

child_fix_alt = plot_child(n_fix, (seen_type != "neither"), color=seen_type, fill=seen_type) +
    gam_fit(k=3, alpha=0.25) +
    points(min_n=10, position=position_dodge(width=.2)) +
    scale_colour_manual(values=c(
        first = SEEN,
        second = SECOND,
        both = BOTH
    ), aesthetics=c("fill", "colour"), name="") +
    scale_x_continuous(breaks=seq(-4, 4, 2), limits=c(-4.1, 4.1)) +
    xlab("Relative\n# Fixations") +
    no_legend +
    annotate("text", x=-4, y=.9, label="second seen", color=SECOND, fontface="bold", size=3.5, hjust = 0) +
    annotate("text", x=4, y=.1, label="first seen", color=SEEN, fontface="bold", size=3.5, hjust = 1) +
    annotate("text", x=-.4, y=.35, label="both", color=BOTH, fontface="bold", size=3.5, hjust = 0.5)

wrap_plots(child_value, child_fix+no_yaxis, child_fix_alt+no_yaxis) &
    plot_annotation(tag_levels = 'A') &
    tight_margin() &
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1)) &
    ylab('Fixate Child 1')

fig("which_child", h=HEIGHT+.1)
"""

R"""
child_model = child_data %>%
    regress(fixated ~ reward + future_value + n_fix + seen, logistic=T, mixed=T, name="child")
"""

# %% ==================== WHERE TO JUMP ====================

R"""
jump_data = df %>%
    filter(relationship != "child") %>%
    group_by(pid, trial_index, fix_number) %>%
    filter(any(fixated)) %>%
    mutate(seen = n_fix > 0) %>%
    mutate(
        initial_state = depth == 0,
        depth = relative_depth,
        tree_distance = n_back + n_forward,
        current_state = depth == 0,
        future_value = action_value - reward,
        past_value = path_value - reward,
        total_value = future_value + past_value + reward
    ) %>% slice_tail(n=1)


plot_jump = function(data, var, ...) {
    data %>%
        ungroup() %>%
        ggplot(aes({{var}}, 1*fixated, ...)) +
        expand_limits(y = 0) +
        ylab("Jump Probability")
}

jump_reward = jump_data %>%
    filter(reward != 0) %>%
    plot_jump(reward) +
    points() +
    gam_fit(k=8) +
    depth_pal + no_legend

jump_prev = jump_data %>%
    filter(depth > 1, past_value != 0) %>%
    plot_jump(past_value) +
    gam_fit() +
    point_bin(bins=7, min_n=10) +
    xlim(-17, 17) +
    xlab("Previous Rewards")

jump_future = jump_data %>%
    filter(future_value != 0) %>%
    plot_jump(future_value) +
    gam_fit() +
    point_bin(bins=7, min_n=10) +
    depth_pal +
    xlim(-10, 10) +
    xlab("Future Rewards")

jump_nfix = jump_data %>%
    plot_jump(n_fix, color=factor(depth)) +
    point_line(min_n=10) +
    depth_pal + no_legend
    xlab("Fixations")

jump_depth = jump_data %>%
    mutate(seen = factor(if_else(n_fix > 0, "seen", "unseen"), levels=c("unseen", "seen"))) %>%
    plot_jump(depth, color=seen) +
    mean_line() +
    points() +
    scale_colour_manual(values=c(
        seen="#3694F0", unseen="gray50"
    ), aesthetics=c("fill", "colour"), name="") + no_legend

seen_pal = scale_colour_manual(values=c("gray60", SEEN))
jump_data %>%
    with(mean(n_unseen_children == 2))

jump_depth = jump_data %>%
    plot_jump(depth, color=seen) +
    point_line() +
    seen_pal + no_legend +
    annotate("text", x=2, y=.27, label="seen", color=SEEN, fontface="bold", size=3.5, hjust = 0) +
    annotate("text", x=2, y=.34, label="unseen", color="gray60", fontface="bold", size=3.5, hjust = 0)

jump_children = jump_data %>%
    plot_jump(n_unseen_children) +
    point_line() +
    scale_x_continuous(breaks=c(0,1,2)) +
    seen_pal + no_legend

wrap_plots(jump_reward, jump_depth+no_yaxis) &
    coord_cartesian(xlim=c(NULL), ylim=c(0, .55)) &
    tight_margin() &
    plot_annotation(tag_levels = 'A') &
    ylab("Fixate Non-Child 1")

fig("jumps", w=3.6)
"""


R"""
jump_model = jump_data %>%
    mutate(depth1 = (depth == 1), unseen= 1 * (n_fix == 0)) %>%
    regress(fixated ~ depth1 + depth1:unseen + depth + reward + past_value + future_value + n_fix,
            logistic=T, mixed=T, name="jump")
"""
