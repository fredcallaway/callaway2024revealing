include("analysis_base.jl")
include("gaze_plots.jl")
include("figure.jl")

kws = (;dir = "diagrams/example", resolution=10)
t = gtrials["P02"][84]

figure("problem"; pdf=true, kws...) do
    plot_problem(t; show_moves=false, scale=1.8)
end

# %% --------

figure("gaze"; kws..., pdf=true) do
    plot_trial_static(t; premove=true, alpha=1., fixations=true)
    b = Box(current_figure()[1, 1], color = (:white, 0.9), strokevisible=false)
    ax = current_axis()
    # plot_problem(t)
    label_looks!(t)
    xlim, ylim = juxt(minimum, maximum).(invert(scale_up(t.layout, 1.25)))
    moves(t)[1].time - t.start_time
    xlims!(ax, xlim)
    ylims!(ax, ylim)
    ax.yreversed = true
    for i in 3:6
        translate!(ax.scene.plots[i], 0, 0, 1)
    end
end

