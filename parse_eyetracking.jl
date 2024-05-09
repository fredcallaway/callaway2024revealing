function smooth_offsets(x; λ=.95)
    res = zeros(length(x))
    idx = collect(eachindex(x))
    for i in eachindex(x)
        d = abs.(idx .- i)
        res[i] = mean(x, Weights(λ .^ d))
    end
    res
end

function compute_offset(asc_file)
    offsets = get_offsets(asc_file)
    offset = median(offsets)
    max_diff = maximum(abs, offsets .- mean(offsets))
    if max_diff > .05
        @warn "highly variable offset ($max_diff from mean) found in $asc_file"
    end
    offset
end


function ensure_asc_file(uid)
    edf = "data/eyelink/$(uid)/raw.edf"
    @assert isfile(edf)
    dest = "data/eyelink/$(uid)/samples.asc"
    if isfile(edf) && !isfile(dest)
        output = readchomp(Cmd(`edf2asc $edf $dest`; ignorestatus=true))
        if !contains(output, "Converted successfully")
            println("Error parsing $edf\n", "-"^80, "\n$output\n", "-"^80)
        end
    end
    return dest
end

function parse_eyedata(uid)
    asc_file = ensure_asc_file(uid)
    fixations = Fixation[]
    gaze = Gaze[]
    offsets = Float64[]
    current_offset = NaN # set once at the beginning of each trial

    fixation = false
    for line in eachline(asc_file)
        if startswith(line, "SFIX")
            fixation = true
        end

        # parse time offsets
        # MSG 9206120 {"time": 90.68254398899808, "event": "drift check"}
        m = match(r"MSG\s+(\d+).*\"time\": (\d+\.\d+)", line)
        if !isnothing(m)
            t_el, t_py = parse.(Float64, m.captures)
            t_el /= 1000  # to seconds
            offset = t_py - t_el
            if isnan(current_offset) || occursin("drift check", line)
                current_offset = offset
            end
            push!(offsets, offset - current_offset)
            continue
        end

        # parse fixation
        # EFIX L   9108463    9108639 178   933.7    98.5    3846
        m = match(r"EFIX (L|R)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)", line)
        if !isnothing(m)
            fixation = false
            eye, start, stop, dur, x, y, pupil = m.captures
            start, stop, dur, pupil = parse.(Int, (start, stop, dur, pupil))
            start /= 1000; stop /= 1000; dur /= 1000  # to seconds
            x, y = parse.(Float64, (x, y))
            start += current_offset
            stop += current_offset
            push!(fixations, Fixation(start, stop, x, y))
            continue
        end

        # parse gaze
        # 9108639   934.5    97.3  3848.0 ...
        m = match(r"(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+).*\.\.\.", line)
        if !isnothing(m)
            t, x, y, pupil = m.captures
            t = parse(Float64, t) / 1000
            x, y, pupil = parse.(Float64, (x, y, pupil))
            push!(gaze, Gaze(t + current_offset, x, y, fixation))
            continue
        end
    end
    max_offset = maximum(abs.(offsets))
    if max_offset > .01
        @warn "highly variable offset ($max_offset) found in $asc_file"
    end
    (;gaze, fixations, offsets)
end