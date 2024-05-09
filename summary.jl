@isdefined(trials) || include("analysis_base.jl")
X = (;
    n_total = length(uids) + n_exclude_p,
    n_exclude = n_exclude_p,
    n_eye_exclude = length(uids) * 100 - length(trials),
    n_time_exclude = sum(timeout, trials),
    n_trial = sum(!timeout, trials),
    n_participant = length(gtrials),
)
@rput X

R"""
tibble(key=names(X), val=unlist(X)) %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{val}", key)
    ))
"""
