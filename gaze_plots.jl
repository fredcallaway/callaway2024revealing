using GLMakie
using GraphMakie
using Graphs
using KissSmoothing
Makie.Point(g::Gaze) = Point(g.x, g.y)
Makie.Point(g::Fixation) = Point(g.x, g.y)
isdir("eye-videos") || mkdir("eye-videos")
isdir("trial-plots") || mkdir("trial-plots")


function Graphs.DiGraph(problem::Problem)
    g = DiGraph(length(states(problem)))
    for (i, js) in enumerate(problem.graph)
        for j in js
            add_edge!(g, i, j)
        end
    end
    g
end

function reward_string(r)
    r = Int(r)
    r == 0 ? "" : r > 0 ? "+$r" : "$r"
end

function plot_rewards!(t::Trial; scale=1.)
    text!(t.layout;
        text=reward_string.(t.problem.rewards),
        align=(:center, :center), fontsize=scale * 16, font=:bold,
        color=fill(:black, 11)
    )
end

function scale_up(points, factor; dx=0, dy=0)
    x, y = invert(points)
    mx, my = mean.((x, y))
    x .-= mx
    y .-= my
    x .*= factor
    y .*= factor
    x .+= mx .+ dy
    y .+= my .+ dy
    invert((x, y))
end

function label_looks!(t)
    fixs = premove_looked(t; keep_move=true)
    lab_pos = scale_up(t.layout, 1.2; dx=-50)

    labs = map(group(fixs, eachindex(fixs))) do fix_nums
        join(fix_nums, ", ")
    end
    locs = lab_pos[collect(keys(labs))]

    text!(locs;
        text=collect(labs),
        align=(:left, :center), fontsize=24, font=:bold,
        color="#3E9F6D"
        # colormap=:haline,
        # color=eachindex(fixs)
    )
end

function plot_problem(t::Trial; show_moves=false, scale=1)
    problem = t.problem
    g = DiGraph(problem)

    node_color = fill("white", 11)
    node_color[problem.start] = "#1D7EF0"
    # if show_moves
    #     for s in path(t)
    #         node_color[s] = "#95C6FF"
    #     end
    # end

    f = Figure(size=(500, 500))
    ax = Axis(f[1, 1]);
    gp = graphplot!(g,
        layout=g->t.layout,
        ;node_color,
        node_strokecolor = :black,
        node_strokewidth = 3,
        edge_width = 3,
        arrow_shift = :end,
        arrow_size=15 * √scale,
        node_size = scale * 39,  # 65
    )
    # ax.backgroundcolor = "#8B8B8B"
    ax.aspect = DataAspect()

    (x1, x2), (y1, y2) = map(invert(scale_up(t.layout, 1.25))) do x
        minimum(x), maximum(x)
    end
    xlims!(ax, x1, x2)
    ylims!(ax, y1, y2)
    hidedecorations!(ax); hidespines!(ax)
    ax.yreversed = true

    labs = plot_rewards!(t; scale)

    f, ax, gp, labs
end


function plot_trial_static(t; premove=false, show_moves=false, fixations=false, alpha=0.3)
    f, ax, gp, labs = plot_problem(t; show_moves)
    # ax = Axis(f[1,1])
    # problem_axis!(ax)


    end_time = premove ?
        moves(t)[1].time :
        actions(t)[end].time

    if fixations
        trial_eye = filter(get_eye_data(t.uid).fixations) do fix
            t.start_time < fix.time < end_time
        end
        # filter!(trial_eye) do f
        #     duration(f) > .15
        # end
    else
        trial_eye = filter(get_eye_data(t.uid).gaze) do fix
            t.start_time < fix.time < end_time
        end
    end

    time = getfield.(trial_eye, :time)
    time .-= first(time)
    if fixations
        scatter!(Point.(trial_eye), color=time, colormap=:haline)
        lines!(Point.(trial_eye), color=time, colormap=:haline, linewidth=4, alpha=1)
    else
        scatter!(Point.(trial_eye); color=time, colormap=:haline, alpha)
        # l = lines!(Point.(trial_eye), color=time, colormap=:inferno, linewidth=4, alpha=0.5)
    end

    scatter!(Point(trial_eye[1]), marker=:star5, markersize=30, color=0,
        colorrange=(minimum(time), maximum(time)), colormap=:haline)
    # plot_rewards!(t)
    # Colorbar(f[1,2], l)

    fn = string("trial-plots/", trial_id(t), ".png")
    save(fn, f)
    fn
end

# %% --------


function plot_trial_gaze(t; framerate = 50, slowdown = 2, smoothing=0., verbose=false, replace=true)
    fn = string("eye-videos/", trial_id(t), ".mp4")
    !replace && isfile(fn) && return fn
    f, ax, gp, labs = plot_problem(t)


    trial_eye = get_gaze(t)

    dt = map(trial_eye) do g
        g.time
    end |> diff |> median

    record_rate = Int(1 / round(dt; digits=3))

    keep_every = Int(record_rate / (framerate*slowdown))
    eye = trial_eye[1:keep_every:end]
    acts = reverse(actions(t))
    visited = falses(11)

    function set_fixation!(s)
        t.gaze_contingent || return
        lc = labs.color[]
        for i in 1:11
            if visited[i]
                lc[i] = :white
            elseif i == s
                lc[i] = :black
            else
                lc[i] = :lightgray
            end
        end
        labs.color = lc
    end
    set_fixation!(0)

    function set_state!(s)
        nc = gp.node_color[]
        for i in 1:11
            if visited[i]
                nc[i] = "#78B0F0"
            elseif i == s
                nc[i] = "#1D7EF0"
            end
        end
        gp.node_color = nc
        visited[s] = true
        lc = labs.color[]
        lc[s] = :white
        labs.color = lc
    end
    set_state!(t.problem.start)

    X = mapreduce(hcat, eye) do e
        [e.x, e.y]
    end
    signal, noise = denoise(X, factor=float(smoothing))
    gaze = Point2.(eachcol(signal))

    gaze_color = map(eye) do g
        "#FF2845"
        # g.fixation ? "#FF2845" : "#FD59D6"
    end

    trial_mouse = map(expand_mouse(get_mouse_data(t), eye)) do m
        Point2(m.x, m.y)
    end

    @assert length(trial_mouse) == length(gaze)

    time = Observable(1)
    fix = scatter!(@lift(gaze[$time]), color=@lift(gaze_color[$time]))


    scatter!(@lift(trial_mouse[$time]), color=:black)

    next = pop!(acts)
    unfixate = Inf

    record(f, fn, eachindex(eye); framerate) do step
        if eye[step].time > unfixate
            set_fixation!(0)
            unfixate = Inf
        end
        if !isnothing(next) && eye[step].time > next.time
            verbose && println(next)
            if next isa Look
                set_fixation!(next.state)
                unfixate = next.end_time
            else
                set_state!(next.state)
            end
            if isempty(acts)
                next = nothing
            else
                next = pop!(acts)
            end
        end
        time[] = step
    end
    fn
end