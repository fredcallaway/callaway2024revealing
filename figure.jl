using CairoMakie
using Dates
mkpath(".fighist")
mkpath("figs")
# gr(label="", dpi=200, size=(400,300), lw=2)
# ENV["GKSwstype"] = "nul"


function figure(func, name="tmp", size=(600,300); dir="figs", resolution=1, pdf=false, kws...)
    f = Figure(;size, kws...);
    Axis(f[1, 1]);
    func()
    dt = Dates.format(now(), "mmdd-HHMMSSss")
    path = ".fighist/$dt-$(replace(name, "/" => "-")).png"
    save(path, current_figure(), px_per_unit=resolution)
    if name != "tmp"
        mkpath(dirname("$dir/$name"))
        if pdf
            save("$dir/$name.pdf", current_figure())
        else
            cp(path, "$dir/$name.png"; force=true)
        end
    end
end