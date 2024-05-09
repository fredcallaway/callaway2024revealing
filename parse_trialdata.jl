
PID_LABELS = integer_labeler(String)

function uid2pid(uid)
    # "P" * lpad(PID_LABELS[uid], 2, '0')
    # uid == "23-12-06-1105_P1" && return "P01B"
    "P" * lpad(split(uid, "_P")[end], 2, '0')
end

function one_index!(d)
    for e in d["events"]
        if "state" in keys(e)
            e["state"] += 1
        end
    end
    d["trial"]["graph"] = map(x -> x .+ 1, d["trial"]["graph"])
    d["trial"]["start"] += 1
    d["trial"]["rewards"] = Int.(d["trial"]["rewards"])
    if haskey(d["trial"], "expansions")
        d["trial"]["expansions"] = map(x -> x .+ 1, d["trial"]["expansions"])
    end
end

function preprocess!(trial_data, uid)
    imap(trial_data) do i, d
        one_index!(d)
        d["uid"] = uid
        d["pid"] = uid2pid(uid)
        d["trial_index"] = i
        d["events"] = namedtuple.(d["events"])
        d["trial"] = namedtuple(d["trial"])
        d["trial_log"] = trial_log(uid, d["events"][1].time)

        timeout_i = findfirst(d["trial_log"]) do line
            occursin("timeout", line)
        end
        d["timeout"] = !isnothing(timeout_i)
        namedtuple(d)
    end
end

function parse_trial(d)
    actions = parse_actions(d.uid, d.events)
    ismissing(actions) && return missing
    i_start = findfirst(e -> e.event == "start", d.events)
    isnothing(i_start) && return missing
    start_time = d.events[i_start].time
    problem = parse_problem(d.trial)
    t = HTrial(d.pid, d.uid, start_time, d.trial_index, problem, actions,
        d.trial.gaze_contingent, d.trial.time_limit, Tuple{Float64, Float64}.(d.trial.node_positions))
    # @assert isempty(setdiff(path(t), looked(t)))
    # @assert looks(get(:move), t) .|> get(:state) == path(t)
    @assert path(t) in paths(t.problem)
    t
end

function parse_trials(data)
    map(enumerate(data)) do (i, d)
        try
            parse_trial(d)
        catch err
            @error "Problem parsing data[$i]" err
            rethrow()
        end
    end |> skipmissing |> collect
end

function parse_problem(trial_spec)
    rewards = copy(trial_spec.rewards)
    rewards[trial_spec.start] = 0
    Problem(
        trial_spec.graph,
        rewards,
        trial_spec.start,
        -1
    )
end

function parse_actions(uid, events)
    actions = Action[]

    start_time = nothing
    entered_time = -1
    entered_state = -1
    move_fix = false
    moved = false
    i = 1

    function check(condition, msg="")
        if !condition
            back = max(1, i-5)
            forward = min(length(events), i+5)
            for j in back:forward
                if j == i
                    print("❌ ")
                end
                println("   ", events[j])
            end
            error("check failed at line $i: $msg")
        end
    end

    function start_fix(s, t)
        if entered_state != -1
            end_fix(entered_state, t)
        end
        entered_state = s
        entered_time = t
    end

    function end_fix(s, t)
        check(entered_state == s, "leaving wrong state")
        check(entered_state != -1, "ending an unstarted fixation")
        push!(actions, HLook(entered_state, entered_time, t, move_fix, get_fixations(uid, entered_time, t)))
        entered_state = -1
        move_fix = false
    end

    function move(s, t)
        push!(actions, HMove(s, t))
        if s == entered_state
            move_fix = true
        end
    end

    for e in events
        if e.event == "start"
            check(start_time == nothing)
            check(isempty(actions))
            start_time = e.time
        elseif e.event == "press x"
            return missing
        elseif e.event == "fixate state"
            start_fix(e.state, e.time)
        elseif e.event == "unfixate state"
            end_fix(e.state, e.time)
        elseif e.event == "done"
            entered_state != -1 && end_fix(entered_state, e.time)
            sort!(actions, by=a->a.time)
            return actions
        elseif e.event == "visit"
            if !moved
                moved = true  # first visit is to initial state, not an action
            else
                move(e.state, e.time)
            end
        end
        i += 1
    end
    error("done event not found")
end


@memoize function trial_logs(uid::String)
    trial_logs = Vector{String}[]
    active = false
    for line in eachline("../experiment/log/$(uid).log")
        if active
            push!(trial_logs[end], line)
            if occursin("done", line) || occursin("stop_recording", line)
                active = false
            end
        else
            if occursin("start flip_time", line)
                active = true
                push!(trial_logs, String[])
                push!(trial_logs[end], line)
            end
        end
    end
    map(trial_logs) do lines
        m = match(r".*Trial.log (.+) start flip_time", lines[1])
        parse(Float64, m.captures[1]), lines
    end
end

function trial_log(uid, start_time)
    tlogs = trial_logs(uid)
    i = findfirst(tlogs) do tl
        tl[1] ≥ start_time
    end
    if isnothing(i)
        @warn "No trial log found for $uid"
        return String[]
    end
    tlogs[i][2]
end