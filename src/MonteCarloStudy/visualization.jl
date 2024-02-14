# we can potentially share this type across multiple strategy types if we are fine with there being
# some access attributes that are not always used
Makie.@recipe(ContingencyStrategyViz) do sence
    Makie.Attributes(;
        player_color = get_player_colors()[begin],
        branch_colors = get_branch_colors(),
        trunk_color = :black,
        position_subsampling = 1,
        trajectory_point_size = 5,
        set_opacity_by_weight = false,
        filter_zero_weight_branches = true,
        highlight_branching_point = true,
        show_only_one_trunk_branch = false,
        starttime = 1,
        branch_indices = nothing,
        branch_visibility = nothing,
        scale_factor = 1.0,
    )
end

Makie.plottype(::ContingencyStrategy) = ContingencyStrategyViz

function Makie.plot!(viz::ContingencyStrategyViz{<:Tuple{ContingencyStrategy}})
    strategy = viz[1]

    is_certain = Makie.@lift(count($strategy.hypothesis_active_indicators) == 1)

    branch_colors = Makie.@lift(map($(viz.branch_colors), $strategy.weights) do color, w
        opacity = $(viz.set_opacity_by_weight) ? w : 1
        (color, opacity)
    end)
    branch_strategies = Makie.@lift($strategy.branch_strategies)

    # assuming for now that the number of branches doesn't change
    for branch_index in eachindex(branch_strategies[])
        branch_strategy = Makie.@lift($branch_strategies[branch_index])
        horizon = Makie.@lift(length($branch_strategy.xs))
        branch_color = Makie.@lift($branch_colors[branch_index])
        time_of_certainty = Makie.@lift let
            if $is_certain
                return 1
            end
            min($strategy.branching_time + 1, $horizon)
        end
        trunk_color = Makie.@lift(something($(viz.trunk_color), $branch_color))
        branching_point = Makie.@lift(Makie.Point2f($branch_strategy.xs[$time_of_certainty][1:2]))
        is_branch_visible = Makie.@lift(begin
            if !$strategy.hypothesis_active_indicators[branch_index]
                false
            elseif isnothing($(viz.branch_visibility))
                true
            else
                $(viz.branch_visibility)[branch_index]
            end
        end)
        branch_start_time = Makie.@lift(max($time_of_certainty, $(viz.starttime)))

        is_this_trunk_branch_visible = Makie.@lift(
            $is_branch_visible &&
            !(
                $(viz.show_only_one_trunk_branch) &&
                branch_index > @something(findfirst($strategy.hypothesis_active_indicators), 1)
            ) &&
            !$is_certain
        )

        # plot the trunk
        Makie.plot!(
            viz,
            branch_strategy;
            color = trunk_color,
            viz.starttime,
            endtime = time_of_certainty,
            visible = is_this_trunk_branch_visible,
            viz.scale_factor,
        )

        # plot the branches
        Makie.plot!(
            viz,
            branch_strategy;
            color = branch_color,
            starttime = branch_start_time,
            visible = Makie.@lift($is_branch_visible && $branch_start_time < $horizon),
            viz.scale_factor,
        )

        is_branching = Makie.@lift $strategy.branching_time < $horizon

        # highlight branching state
        Makie.scatter!(
            viz,
            branching_point;
            # WGLMakie currently does not support stroke in scatter plots. Thus we have to fill the
            # diamond with the player color to avoid an invisible marker.
            #color = Makie.current_backend() == WGLMakie ? viz.player_color : :white,
            color = :white,
            marker = :diamond,
            strokecolor = viz.player_color,
            strokewidth = Makie.@lift(1.5 * $(viz.scale_factor)),
            markersize = Makie.@lift(
                $(viz.highlight_branching_point) * $is_branching * 10 * $(viz.scale_factor)
            ),
            visible = Makie.@lift(
                $(viz.starttime) <= $time_of_certainty &&
                1 < $time_of_certainty < $horizon &&
                !$is_certain &&
                !($(viz.show_only_one_trunk_branch) && branch_index < length($strategy.weights))
            )
        )
    end

    viz
end

function visualize_average_tb_sweep(
    study_evaluation;
    resolution = (270, 187),
    canvas = Makie.Figure(; resolution),
    highlighted_branching_times = Int[],
    color = :black,
    style = :stderr,
    metric_type = :contingency, #[:upper_bound, :contingency],
    eval_type = :openloop,
    axis_kwargs = (;),
    scatter_opacity = 0.2,
    boxplot_opacity = 0.5,
    highlight_color = ColorSchemes.colorschemes[:Oranges_9][5],
    tick_distance = 5,
    show_title = false,
    title = show_title ? "Expectation vs. Contingency" : nothing,
)
    xticks = let
        tb_min, tb_max = extrema(run.branching_time for run in study_evaluation)
        inner_ticks = (tb_min + tick_distance - 1):tick_distance:tb_max

        union(tb_min, tb_max, inner_ticks, highlighted_branching_times)
    end

    if metric_type === :relative_gap
        ylabel = "relative cost gap"
    elseif metric_type === :absolute_gap
        ylabel = "absolute cost gap"
    else
        ylabel = "cost"
    end

    axis = Makie.Axis(
        canvas[1, 1];
        xlabel = L"branching time step, $t_b$ [#]",
        ylabel,
        axis_kwargs...,
        xticks,
    )

    if !isnothing(title)
        axis.title = title
    end

    # visualize raw data as scatter plot with low opacity
    branching_times = [run.branching_time for run in study_evaluation]
    if isnothing(highlighted_branching_times)
        colors = color
    else
        colors =
            [tb in highlighted_branching_times ? highlight_color : color for tb in branching_times]
        # also draw a box around these
        Makie.vlines!(axis, highlighted_branching_times; color = highlight_color, linestyle = :dash)
    end

    if metric_type isa Symbol
        metric_types = [metric_type]
    elseif metric_type isa Vector
        metric_types = metric_type
    else
        error("unknown metric type type $(typeof(metric_type))")
    end

    for metric_type in metric_types
        get_metric(run) = run[Symbol(eval_type, :_cost)][metric_type]
        metrics = [get_metric(run) for run in study_evaluation]
        if style === :boxplots
            Makie.boxplot!(
                axis,
                branching_times,
                metrics;
                color = tuple.(colors, boxplot_opacity),
                whiskerwidth = :match,
                markersize = 3,
            )
        elseif style === :stderr
            ## plot stats as well
            let
                runs_per_branching_time =
                    SplitApplyCombine.group(r -> r.branching_time, study_evaluation)
                group_branching_times = [tb for tb in keys(runs_per_branching_time)]

                # compute stats
                quantiles = map(runs_per_branching_time) do group
                    group_metrics = [get_metric(run) for run in group]
                    median = Statistics.mean(group_metrics)
                    std_err = Statistics.std(group_metrics) / sqrt(length(group_metrics))
                    lower = median - std_err
                    upper = median + std_err
                    (; lower, median, upper)
                end

                average_metrics = [quantile.median for quantile in quantiles]
                lower_metrics = [quantile.lower for quantile in quantiles]
                upper_metrics = [quantile.upper for quantile in quantiles]

                Makie.lines!(axis, group_branching_times, average_metrics; color)
                ## plot the stderr as ribbons on top
                Makie.band!(
                    axis,
                    group_branching_times,
                    lower_metrics,
                    upper_metrics;
                    color = (color, scatter_opacity),
                )

                if isnothing(highlighted_branching_times)
                    scatter_colors = color
                else
                    scatter_colors = [
                        tb in highlighted_branching_times ? highlight_color : color for
                        tb in group_branching_times
                    ]
                end
                Makie.scatter!(
                    axis,
                    group_branching_times,
                    average_metrics;
                    color = scatter_colors,
                    markersize = 5,
                )
            end
        else
            error("unknown style $style")
        end
    end

    canvas
end

"""
Visualize a metric for a given tb slice of the study evaluation.
"""
function visualize_spatially(
    demo_setup,
    study_evaluation,
    branching_time;
    show_colorbar = true,
    metric_type = :relative_gap,
    eval_type = :openloop,
    axis_kwargs = (;),
    number_of_levels = 20,
    canvas = Makie.Figure(; resolution = (375 + (show_colorbar ? 75 : 0), 900)),
    share_colorscale = true,
    contour_line_width = 0.0,
    contour_line_color = :white,
    player_colors = get_player_colors(),
    return_info = false,
    hidedecorations = true,
    create_environment_axis = true,
    hidexlabel = false,
    hideylabel = false,
)
    if create_environment_axis
        axis = TrajectoryGamesExamples.create_environment_axis(
            canvas[1, 1],
            demo_setup.contingency_game.env;
            viz_kwargs = (; strokewidth = 0, color = :white),
            xlabel = L"p_\text{lat}",
            ylabel = L"p_\text{lon}",
            aspect = Makie.DataAspect(),
            axis_kwargs...,
        )

    else
        axis = Makie.Axis(
            canvas[1, 1];
            xlabel = L"p_\text{lat}",
            ylabel = L"p_\text{lon}",
            aspect = Makie.DataAspect(),
            axis_kwargs...,
        )
    end
    if hidedecorations
        Makie.hidexdecorations!(axis; label = hidexlabel)
        Makie.hideydecorations!(axis; label = hideylabel)
    end

    # visualize cost metric over perturbations of the nominal position
    get_metric = run -> run[Symbol(eval_type, :_cost)][metric_type]

    last_plot = nothing

    local px_min, px_max, py_min, py_max

    let
        metric_min, metric_max = extrema(get_metric(run) for run in study_evaluation)

        filtered_runs = filter(r -> r.branching_time == branching_time, study_evaluation)

        grouped_runs = SplitApplyCombine.group(r -> r.opponent_index, filtered_runs)
        metric_min, metric_max = extrema(
            get_metric(run) for run in (share_colorscale ? study_evaluation : filtered_runs)
        )

        for (opponent_index, opponent_group) in pairs(grouped_runs)
            sorted_runs = sort(opponent_group; by = r -> r.position_index)
            xgrid_points, ygrid_points = sorted_runs[end].position_index.I
            results_on_grid = map(reshape(sorted_runs, xgrid_points, ygrid_points)) do run
                position = run.belief[begin].state[Block(opponent_index)][1:2]
                metric = get_metric(run)
                (; position, metric)
            end
            pxs = [r.position[1] for r in results_on_grid[:, 1]]
            pys = [r.position[2] for r in results_on_grid[1, :]]
            metrics = [r.metric for r in results_on_grid]
            levels = range(metric_min, metric_max; length = number_of_levels)
            last_plot = Makie.contourf!(axis, pxs, pys, metrics; levels)
            Makie.contour!(
                axis,
                pxs,
                pys,
                metrics;
                levels,
                color = contour_line_color,
                linewidth = contour_line_width,
            )

            let
                px_min, px_max = extrema(pxs)
                py_min, py_max = extrema(pys)

                Makie.poly!(
                    axis,
                    Makie.Rect(px_min, py_min, px_max - px_min, py_max - py_min);
                    color = (:white, 0),
                    strokecolor = :black,
                    strokewidth = 1,
                    linestyle = :dash,
                )
            end
        end

        if show_colorbar
            Makie.Colorbar(canvas[1, 2], last_plot)
        end
    end

    if create_environment_axis
        # render the "nominal state"
        # visualize nominal positions for every player
        nominal_state = demo_setup.initial_state
        for (xi, player_color) in zip(blocks(nominal_state), player_colors)
            Makie.scatter!(
                axis,
                Makie.Point2f(xi[1], xi[2]);
                color = player_color,
                strokecolor = :white,
                strokewidth = 1,
            )
        end
    end

    if return_info
        return (; last_plot, px_min, px_max, py_min, py_max)
    end

    canvas
end

function visualize_trajectory_sample(
    raw_study_results,
    branching_time;
    canvas = Makie.Figure(; resolution = (375, 9000)),
    axis_kwargs = (;),
    strategy_viz_kwargs = (;),
    eval_type = :openloop,
    strategy_type = :contingency,
    scenario_id = nothing,
    title = nothing,
    branch_index = 2,
)
    axis = TrajectoryGamesExamples.create_environment_axis(
        canvas[1, 1],
        raw_study_results.demo_setup.contingency_game.env;
        viz_kwargs = (; strokecolor = :black, strokewidth = 1),
        aspect = Makie.DataAspect(),
        xlabel = L"p_\text{lat}",
        ylabel = L"p_\text{lon}",
        axis_kwargs...,
    )

    if !isnothing(title)
        axis.title = title
    end

    # visualize one trajectory sample at the specified branching time
    # TODO: think about how to select the sample... For now just chosing the one with the highest
    # position difference at the branching time
    run = let
        filtered_runs = collect(
            Iterators.filter(r -> r.branching_time == branching_time, raw_study_results.runs),
        )
        if isnothing(scenario_id)
            # chose the point which has the highest state difference at the branching time
            scenario_id = argmax(eachindex(filtered_runs)) do ii
                run = filtered_runs[ii]
                contingency_branching_position =
                    run.contingency_strategy.substrategies[begin].branch_strategies[begin].xs[branching_time + 1][1:2]
                expectation_branching_position =
                    run.upper_bound_strategy.substrategies[begin].branch_strategies[begin].xs[branching_time + 1][1:2]
                norm(contingency_branching_position - expectation_branching_position)
            end
        end
        filtered_runs[scenario_id]
    end

    if eval_type === :closedloop
        if strategy_type === :contingency
            strategy = run.contingency_strategy
        elseif strategy_type === :expectation
            strategy = run.upper_bound_strategy
        else
            error("unknown strategy type $strategy_type")
        end
    elseif eval_type === :openloop
        if strategy_type === :contingency
            strategy = run.contingency_trunk.strategy
        elseif strategy_type === :expectation
            strategy = run.upper_bound_trunk.strategy
        else
            error("unknown strategy type $strategy_type")
        end
    else
        error("unknown planning type $eval_type")
    end

    if eval_type === :openloop
        visualize_joint_contingency_strategy!(axis, strategy; strategy_viz_kwargs...)
    elseif eval_type === :closedloop
        # TODO: this should just be a different function...
        visualize_joint_strategy_closed_loop!(axis, strategy, branch_index; strategy_viz_kwargs...)
    else
        error("unknown eval type $eval_type")
    end

    canvas
end

# TODO: factor this out into a generic method that uses the TrajectoryGamesBase.visualize! dispatch
function visualize_joint_contingency_strategy!(
    axis,
    strategy;
    player_colors = get_player_colors(),
    branch_colors = get_branch_colors(),
    trunk_color = :black,
    position_subsampling = 1,
    trajectory_point_size = 5,
    seperate_trunk_for_opponents = false,
    show_branch_point_for_opponents = false,
    set_opacity_by_weight = false,
    filter_zero_weight_branches = true,
    show_annotations = false,
)
    for (player_index, substrategy) in enumerate(strategy.substrategies)
        player_color = player_colors[player_index]
        for (branch_strategy, branch_color, weight) in
            zip(substrategy.branch_strategies, branch_colors, substrategy.weights)
            if filter_zero_weight_branches && weight ≈ 0.0
                continue
            end
            strategy_points = [Makie.Point2f(x[1:2]) for x in branch_strategy.xs]
            intial_point = strategy_points[begin]

            # color by branch and trunk/tail and set opacity according to weight
            color = map(eachindex(strategy_points)) do t
                if !seperate_trunk_for_opponents && player_index > 1
                    base_color = branch_color
                else
                    base_color = t <= substrategy.branching_time ? trunk_color : branch_color
                end
                opacity = set_opacity_by_weight ? weight : 1.0

                (base_color, opacity)
            end

            Makie.lines!(axis, strategy_points; color)

            # add annotation above at the end
            if show_annotations
                player_letter = 'A' + player_index - 1
                text_position =
                    Makie.Point3f(strategy_points[end]..., 1) + Makie.Point3f(0.0, 0.1, 1)
                # place a marker behind the text for better visibility
                Makie.scatter!(
                    axis,
                    text_position;
                    color = (:white, 0.2),
                    markerspace = :pixel,
                    markersize = 35,
                    marker = :circle,
                )

                Makie.text!(
                    axis,
                    text_position;
                    text = L"$z^%$player_letter$",
                    align = (:center, :center),
                )
            end

            # highlight points along the way
            Makie.scatter!(
                axis,
                strategy_points[1:position_subsampling:end];
                color = color[1:position_subsampling:end],
                markersize = trajectory_point_size,
            )
            # highlight initial state
            Makie.scatter!(
                axis,
                intial_point,
                color = player_color,
                strokecolor = :white, #player_index === 1 ? :white : :black,
                strokewidth = 1,
            )
            if substrategy.branching_time < length(strategy_points)
                branching_point = strategy_points[substrategy.branching_time + 1]
                if player_index === 1 || show_branch_point_for_opponents
                    # highlight branching state
                    Makie.scatter!(
                        axis,
                        branching_point,
                        color = :white,
                        marker = :diamond,
                        strokecolor = player_color,
                        strokewidth = 1.5,
                    )
                end
            end
        end
    end
end

function interpolate(xs::AbstractVector, t)
    if t < 1
        @assert false
        return xs[begin]
    elseif t > length(xs)
        @assert false
        return xs[end]
    else
        t_floor = floor(Int, t)
        t_ceil = ceil(Int, t)
        if t_floor == t_ceil
            return xs[t_floor]
        end
        return (t_ceil - t) * xs[t_floor] + (t - t_floor) * xs[t_ceil]
    end
end

function zipper_gui(
    strategy_matrix;
    scale_factor = 1.0,
    base_resolution = (470, 900),
    canvas = Makie.Figure(; resolution = base_resolution .* scale_factor),
    axis_kwargs = (; limits = ((-1.2, 1.2), (-2.5, 2.25))),
    title = nothing,
    visualize_decorations = visualize_crosswalk_decorations,
    get_player_markers = get_crosswalk_player_markers,
    marker_size_per_player = get_crosswalk_player_marker_sizes(),
    player_colors = get_player_colors(),
    return_handles = false,
    groundtruth_options = ["right", "left"],
    for_offline_state_recording = false,
)
    Makie.COLOR_ACCENT_DIMMED[] = colorant"lightgray"
    Makie.COLOR_ACCENT[] = colorant"gray"
    demo_setup = strategy_matrix[begin].demo_setup
    flattened_strategy_matrix = mapreduce(vcat, strategy_matrix) do row
        map(row.runs) do run
            (; run..., row.branching_time)
        end
    end

    branching_times = sort(unique(map(x -> x.branching_time, flattened_strategy_matrix)))
    h1_beliefs = sort(
        unique(map(x -> x.strategy.substrategies[begin].weights[end], flattened_strategy_matrix)),
    )

    axis = TrajectoryGamesExamples.create_environment_axis(
        canvas[1, 1],
        demo_setup.contingency_game.env;
        viz_kwargs = (; strokewidth = 0, color = :white),
        aspect = Makie.DataAspect(),
        ylabelpadding = 5, # hack to work around makie bug
        xlabel = L"p_\text{lat}",
        ylabel = L"p_\text{lon}",
        axis_kwargs...,
    )
    Makie.hidedecorations!(axis, label = false)
    visualize_decorations(axis)

    if !isnothing(title)
        axis.title = title
    end

    if for_offline_state_recording
        h1_belief_slider = JSServe.Slider(h1_beliefs; value = 0.5)
        branching_time_slider = JSServe.Slider(branching_times; value = 10)
        simulation_step_slider = JSServe.Slider(1:25; value = 1)
        # menues are not supported by the state map so we mock a dummy object
        groundtruth_menu = (;
            selection = Makie.Observable(groundtruth_options[begin]),
            options = groundtruth_options,
        )
    else
        h1_belief_slider, branching_time_slider, simulation_step_slider =
            Makie.SliderGrid(
                canvas[2, 1],
                (; label = L"b(\mathrm{left})", range = h1_beliefs, startvalue = 0.5),
                (; label = L"t_b", range = branching_times, startvalue = 10),
                (; label = L"t", range = 1:25, startvalue = 1),
            ).sliders
        menu_area = canvas[3, 1]
        Makie.Label(menu_area[1, 1], L"\theta_\mathrm{true}")
        groundtruth_menu = Makie.Menu(menu_area[1, 2], options = groundtruth_options)
    end

    branch_index = Makie.@lift let
        findfirst(==($(groundtruth_menu.selection)), groundtruth_options)
    end
    marker_per_player = Makie.@lift get_player_markers($branch_index)

    branching_time = Makie.@lift identity($(branching_time_slider.value))
    h1_belief = Makie.@lift identity($(h1_belief_slider.value))

    # visualize one trajectory sample at the specified branching time
    # TODO: think about how to select the sample... For now just chosing the one with the highest
    # position difference at the branching time
    run = Makie.@lift let
        filtered_runs = collect(
            Iterators.filter(flattened_strategy_matrix) do run
                run.branching_time == $branching_time &&
                    run.strategy.substrategies[begin].weights[end] == $h1_belief
            end,
        )
        only(filtered_runs)
    end

    simulation_step = Makie.@lift identity($(simulation_step_slider.value))

    strategy = Makie.@lift $run.strategy

    # assumes that the number of players doesn't change
    data_per_player = map(eachindex(strategy[].substrategies)) do player_index
        state =
            Makie.@lift $strategy.substrategies[$player_index].branch_strategies[$branch_index].xs[$simulation_step]
        position = Makie.@lift Makie.Point2f($state[1:2])
        orientation = Makie.@lift $state[4]

        (; position, orientation)
    end

    for (player_index, player_data) in enumerate(data_per_player)
        Makie.scatter!(
            axis,
            player_data.position;
            marker = Makie.@lift($marker_per_player[player_index]),
            markersize = marker_size_per_player[player_index],
            rotations = player_data.orientation,
            markerspace = :data,
            color = player_colors[player_index],
        )
    end

    branch_visibility = Makie.@lift begin
        if $simulation_step > $branching_time && $simulation_step > 1
            vs = falses(2)
            vs[$branch_index] = true
        else
            vs = trues(2)
        end
        vs
    end

    Makie.plot!(
        axis,
        strategy;
        substrategy_attributes = Makie.@lift(
            [
                [(; trunk_color = :black, branch_visibility = $branch_visibility, scale_factor)]
                [
                    (;
                        trunk_color = nothing,
                        highlight_branching_point = false,
                        branch_visibility = $branch_visibility,
                        scale_factor,
                    ) for _ in 1:(length(data_per_player) - 1)
                ]
            ]
        )
    )

    if return_handles
        return (;
            canvas,
            branching_time_slider,
            h1_belief_slider,
            simulation_step_slider,
            groundtruth_menu,
        )
    end

    canvas
end

function record_strategy_matrix_frames(
    strategy_matrix;
    file_base_name = "strategy_matrix_export/strategy_matrix",
    zipper_gui_kwargs = (;),
    scale_factor = 1.0,
)
    GLMakie.activate!()

    # generate the widget
    (; canvas, branching_time_slider, h1_belief_slider, simulation_step_slider, groundtruth_menu) =
        zipper_gui(
            strategy_matrix;
            zipper_gui_kwargs...,
            return_handles = true,
            for_offline_state_recording = true,
            scale_factor,
        )

    ## enumerate all the possible user selections
    for (cartesian_index, (tb, b, t, gt)) in pairs(
        collect(
            Iterators.product(
                branching_time_slider.values[],
                h1_belief_slider.values[],
                simulation_step_slider.values[],
                groundtruth_menu.options,
            ),
        ),
    )
        tb_idx, b_idx, t_idx, gt_idx = cartesian_index.I

        # set the sliders to the correct values
        branching_time_slider.value[] = tb
        h1_belief_slider.value[] = b
        simulation_step_slider.value[] = t
        groundtruth_menu.selection[] = gt

        Makie.save(
            "$(file_base_name)_b$(b_idx)_tb$(tb_idx)_t$(t_idx)_gt$(gt_idx).jpg",
            canvas,
            scale_factor = scale_factor,
        )
    end
end

function record_strategy_matrix_video(
    strategy_matrix,
    frame_specs;
    framerate = 8,
    zipper_gui_kwargs = (;),
    realtime = false,
    scale_factor = 2.0,
    filename = "supplementary.mp4",
)
    GLMakie.activate!()
    Makie.with_theme(;
        figure_padding = 10,
        linewidth = 1.5 * scale_factor,
        fontsize = 18 * scale_factor,
    ) do
        (;
            canvas,
            branching_time_slider,
            h1_belief_slider,
            simulation_step_slider,
            groundtruth_menu,
        ) = zipper_gui(strategy_matrix; zipper_gui_kwargs..., return_handles = true, scale_factor)

        display(canvas)

        default_slider_color = branching_time_slider.color_active[]
        default_slider_color_dimmed = branching_time_slider.color_active_dimmed[]

        highlight_color = colorant"black"
        highlight_color_dimmed = colorant"gray"

        function highlight_if_active(slider, is_active)
            if is_active
                slider.color_active[] = highlight_color
                slider.color_active_dimmed[] = highlight_color_dimmed
            else
                slider.color_active[] = default_slider_color
                slider.color_active_dimmed[] = default_slider_color_dimmed
            end
        end

        Makie.record(canvas, filename, frame_specs; framerate) do frame_description
            Makie.set_close_to!(branching_time_slider, frame_description.branching_time)
            Makie.set_close_to!(h1_belief_slider, frame_description.h1_belief)
            Makie.set_close_to!(simulation_step_slider, frame_description.simulation_step)
            groundtruth_menu.i_selected[] = frame_description.groundtruth_option

            highlight_if_active(branching_time_slider, frame_description.is_branching_time_active)
            highlight_if_active(h1_belief_slider, frame_description.is_h1_belief_active)
            highlight_if_active(simulation_step_slider, frame_description.is_simulation_step_active)

            if realtime
                sleep(1 / framerate)
            end
        end
    end
end

function record_crosswalk_strategy_matrix_video(
    strategy_matrix;
    filename = "crosswalk.mp4",
    kwargs...,
)
    frame_specs = [
        [
            # sweep over beliefs
            (;
                branching_time = 10,
                h1_belief = b,
                simulation_step = 1,
                groundtruth_option = 1,
                is_branching_time_active = false,
                is_h1_belief_active = true,
                is_simulation_step_active = false,
                is_groundtruth_menu_active = false,
            ) for b in [
                [0.5 for _ in 1:5]
                0.5:-0.1:0.0
                [0.0 for _ in 1:3]
                0.0:0.1:1.0
                [1.0 for _ in 1:3]
                1.0:-0.1:0.5
            ]
        ]
        [
            # sweep over branching times
            (;
                branching_time = tb,
                h1_belief = 0.5,
                simulation_step = 1,
                groundtruth_option = 1,
                is_branching_time_active = true,
                is_h1_belief_active = false,
                is_simulation_step_active = false,
                is_groundtruth_menu_active = false,
            ) for tb in [
                [10 for _ in 1:5]
                10:-1:0
                [0 for _ in 1:5]
                0:1:25
                [25 for _ in 1:5]
                25:-1:8
            ]
        ]
        [
            # sweep to an "interesting" intermediate belief
            (;
                branching_time = 8,
                h1_belief = b,
                simulation_step = 1,
                groundtruth_option = 1,
                is_branching_time_active = false,
                is_h1_belief_active = true,
                is_simulation_step_active = false,
                is_groundtruth_menu_active = false,
            ) for b in [
                [0.5 for _ in 1:5]
                0.5:-0.1:0.2#
                [0.2 for _ in 1:5]
            ]
        ]
        [
            # "play" the simulation by advancing the simulation steps
            (;
                branching_time = 8,
                h1_belief = 0.2,
                simulation_step = t,
                groundtruth_option = 1,
                is_branching_time_active = false,
                is_h1_belief_active = false,
                is_simulation_step_active = true,
                is_groundtruth_menu_active = true,
            ) for t in [
                [1 for _ in 1:5]
                1:25
                [25 for _ in 1:5]
            ]
        ]
        [
            # "play" again but with different ground truth option
            (;
                branching_time = 8,
                h1_belief = 0.2,
                simulation_step = t,
                groundtruth_option = 2,
                is_branching_time_active = false,
                is_h1_belief_active = false,
                is_simulation_step_active = true,
                is_groundtruth_menu_active = true,
            ) for t in [
                [1 for _ in 1:5]
                1:25
                [25 for _ in 1:3]
            ]
        ]
    ]

    record_strategy_matrix_video(strategy_matrix, frame_specs; filename, kwargs...)
end

function record_overtaking_strategy_matrix_video(
    strategy_matrix;
    filename = "overtaking.mp4",
    zipper_gui_kwargs = (;
        axis_kwargs = (; limits = ((-0.25, 0.75), (-4.3, 2.0)), xticks = [-0.25, 0.75]),
        visualize_decorations = visualize_overtaking_decorations,
        marker_size_per_player = [0.55, 0.55, 0.55],
        get_player_markers = let
            base_marker_per_player = [
                FileIO.load("$(@__DIR__)/../../media/car1.png"),
                FileIO.load("$(@__DIR__)/../../media/car2.png"),
                FileIO.load("$(@__DIR__)/../../media/car3.png"),
            ]
            branch_index -> base_marker_per_player
        end,
    ),
    kwargs...,
)
    frame_specs = [
        [
            # "play" the simulation by advancing the simulation steps
            (;
                branching_time = 10,
                h1_belief = 0.2,
                simulation_step = t,
                groundtruth_option = 1,
                is_branching_time_active = false,
                is_h1_belief_active = false,
                is_simulation_step_active = true,
                is_groundtruth_menu_active = true,
            ) for t in [
                [1 for _ in 1:5]
                1:25
                [25 for _ in 1:5]
            ]
        ]
        [
            # "play" again but with different ground truth option
            (;
                branching_time = 10,
                h1_belief = 0.2,
                simulation_step = t,
                groundtruth_option = 2,
                is_branching_time_active = false,
                is_h1_belief_active = false,
                is_simulation_step_active = true,
                is_groundtruth_menu_active = true,
            ) for t in [
                [1 for _ in 1:5]
                1:25
                [25 for _ in 1:3]
            ]
        ]
    ]

    record_strategy_matrix_video(
        strategy_matrix,
        frame_specs;
        filename,
        zipper_gui_kwargs,
        kwargs...,
    )
end

function linear_markersize_schedule(times; startsize = 5, endsize = 15)
    duration = times[end] - times[begin]
    map(times) do t
        startsize + (t - times[begin]) / duration * (endsize - startsize)
    end
end

function visualize_joint_strategy_closed_loop!(axis, strategy, hypothesis, args...; kwargs...)
    # extract joint strategy for this hypothesis
    joint_branch_trajectory = TrajectoryGamesBase.stack_trajectories(
        map(strategy.substrategies) do ss
            branch_strategy = ss.branch_strategies[hypothesis]
            (; branch_strategy.xs, branch_strategy.us)
        end,
    )

    visualize_joint_trajectory_closed_loop!(axis, joint_branch_trajectory, args...; kwargs...)
end

function visualize_joint_trajectory_closed_loop!(
    axis,
    trajectory,
    time::Makie.Observable = Makie.Observable(length(trajectory.xs));
    player_colors = get_player_colors(),
)
    subtrajectories = TrajectoryGamesBase.unstack_trajectory(trajectory)

    # visualize scatter points along trajectory
    for (player_index, subtrajectory) in enumerate(subtrajectories)
        player_color = player_colors[player_index]
        let trajectory_points = Makie.Observable(Makie.Point2f[]),
            markersize = Makie.Observable(Float32[])

            update_plot_data = function (t)
                copy!(markersize[], linear_markersize_schedule(1:t))
                copy!(
                    trajectory_points[],
                    [Makie.Point2f(x[1:2]) for x in first(subtrajectory.xs, t)],
                )
                trajectory_points[] = trajectory_points[]
            end
            Makie.on(update_plot_data, time; update = true)
            Makie.scatter!(axis, trajectory_points; color = player_color, markersize)
        end
    end
end

function visualize_closedloop_trajectory_banner(
    raw_study_results;
    scenario_id,
    hypothesis,
    branching_time,
    resolution = (285, 375),
    canvas = Makie.Figure(; resolution),
    axis_kwargs = NamedTuple(),
    player_colors = get_player_colors(),
    visualize_decorations = (axis) -> axis,
    kwargs...,
)
    filtered_runs = filter(r -> r.branching_time == branching_time, raw_study_results.runs)

    # max_ssid = argmax(eachindex(filtered_runs)) do ssid
    #     run = filtered_runs[ssid]

    #     p1 =
    #         run.contingency_strategy.substrategies[begin].branch_strategies[begin].xs[branching_time + 1][1:2]
    #     p2 =
    #         run.upper_bound_strategy.substrategies[begin].branch_strategies[begin].xs[branching_time + 1][1:2]
    #     norm(p1 - p2)
    # end

    # @show max_ssid

    run = filtered_runs[scenario_id]

    for (ii, strategy_type) in enumerate([:expectation, :contingency])
        axis = TrajectoryGamesExamples.create_environment_axis(
            canvas[1, ii],
            raw_study_results.demo_setup.contingency_game.env;
            viz_kwargs = (; color = :white, strokewidth = 0),
            aspect = Makie.DataAspect(),
            xlabel = L"p_\text{lat}",
            ylabel = L"p_\text{lon}",
            ylabelpadding = 5, # hack to work around makie bug
            axis_kwargs...,
            title = strategy_type === :expectation ? "Baseline" : "Ours",
            titlefont = "Time New Roman Bold",
        )

        if strategy_type === :contingency
            closedloop_strategy = run.contingency_strategy
            initial_openloop_strategy = run.contingency_trunk.strategy
        elseif strategy_type === :expectation
            closedloop_strategy = run.upper_bound_strategy
            initial_openloop_strategy = run.upper_bound_trunk.strategy
        else
            error("unknown strategy type $strategy_type")
        end

        visualize_decorations(axis)

        # viualize only the-player plan at the initial state
        let
            for branch_strategy in initial_openloop_strategy.substrategies[begin].branch_strategies
                initial_plan_points = [Makie.Point2f(x[1:2]) for x in branch_strategy.xs]

                Makie.lines!(axis, initial_plan_points; color = :gray)
                Makie.scatter!(axis, initial_plan_points; color = :gray, markersize = 5)
            end
        end

        # visualize the closed loop trajectory
        visualize_joint_strategy_closed_loop!(
            axis,
            closedloop_strategy,
            hypothesis;
            player_colors,
            kwargs...,
        )

        # higlight the branching state
        branching_state =
            closedloop_strategy.substrategies[begin].branch_strategies[hypothesis].xs[branching_time + 1]
        Makie.scatter!(
            axis,
            Makie.Point2f(branching_state[1:2]),
            color = :white,
            marker = :diamond,
            strokecolor = player_colors[begin],
            strokewidth = 1.5,
        )

        Makie.hidexdecorations!(axis, label = false)
        Makie.hideydecorations!(axis, label = ii != 1)
    end

    canvas
end

function visualize_spatial_banner(
    study_results;
    highlighted_branching_times = [5, 25],
    eval_type = :openloop,
    metric_type = :contingency,
    canvas = Makie.Figure(resolution = (450, 350)),
    spatial_viz_kwargs = (;),
    highlight_color = ColorSchemes.colorschemes[:Oranges_9][5],
    show_initial_state = true,
    player_colors = get_player_colors(),
    visualize_decorations = (axis) -> axis,
    show_average_inset = true,
)
    last_info = nothing

    reference_env_canvas = canvas[1, 1]
    canvas[1, 2] = heatmap_grid = Makie.GridLayout()

    demo_setup = study_results["raw_results"].demo_setup

    # render a raw env without any strategy for context on the left
    axis_kwargs = get(spatial_viz_kwargs, :axis_kwargs, (;))
    axis = TrajectoryGamesExamples.create_environment_axis(
        reference_env_canvas,
        demo_setup.contingency_game.env;
        viz_kwargs = (; strokewidth = 0, color = :white),
        xlabel = L"p_\text{lat}",
        ylabel = L"p_\text{lon}",
        ylabelpadding = 5, # hack to work around makie bug
        aspect = Makie.DataAspect(),
        axis_kwargs...,
    )
    Makie.hidedecorations!(axis, label = false)
    visualize_decorations(axis)

    if show_initial_state
        for (player_index, state) in enumerate(blocks(demo_setup.initial_state))
            Makie.scatter!(
                axis,
                Makie.Point2f(state[1], state[2]),
                color = player_colors[player_index],
                strokecolor = :white,
                strokewidth = 1,
            )
        end
    end

    if show_average_inset
        # TODO: forward all options
        this_inset_canvas = heatmap_grid[1, 1]
        Makie.rowsize!(heatmap_grid, 1, Makie.Relative(0.3))
        visualize_average_tb_sweep(
            study_results["evaluation"];
            canvas = this_inset_canvas,
            highlighted_branching_times,
        )
        heatmap_inset_indices = eachindex(highlighted_branching_times) .+ 1
    else
        heatmap_inset_indices = eachindex(highlighted_branching_times)
    end

    for (ii, branching_time) in zip(heatmap_inset_indices, highlighted_branching_times)
        heatmap_grid[ii, 1] = this_heatmap_canvas = Makie.GridLayout()
        Makie.Box(
            this_heatmap_canvas[1, 1];
            alignmode = Makie.Outside(8),
            color = (highlight_color, 0.5),
            strokecolor = highlight_color,
            tellwidth = false,
            tellheight = false,
        )
        Makie.Label(
            this_heatmap_canvas[1, 1],
            L"t_b = %$branching_time";
            padding = (10, 10, 10, 10),
            tellwidth = false,
            tellheight = true,
        )
        last_info = visualize_spatially(
            study_results["raw_results"].demo_setup,
            study_results["evaluation"],
            branching_time;
            canvas = this_heatmap_canvas[2, 1],
            show_colorbar = false,
            #spatial_viz_kwargs...,
            axis_kwargs = (;
                tellwidth = true,
                leftspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
                topspinevisible = false,
            ),
            hidexlabel = ii != length(highlighted_branching_times),
            return_info = true,
            create_environment_axis = false,
            eval_type,
            metric_type,
        )
        Makie.rowgap!(this_heatmap_canvas, 0)
    end

    # render a dashed box for the origin of the heatmap
    let
        Makie.poly!(
            axis,
            Makie.Rect(
                last_info.px_min,
                last_info.py_min,
                last_info.px_max - last_info.px_min,
                last_info.py_max - last_info.py_min,
            ),
            color = (:white, 0),
            strokewidth = 1,
            linestyle = :dash,
        )
    end

    label = let
        eval_type_string = eval_type === :closedloop ? "closed-loop" : "open-loop"
        if metric_type === :relative_gap
            "relative $eval_type_string cost gap"
        elseif metric_type === :absolute_gap
            "absolute $eval_type_string cost gap"
        elseif metric_type === :contingency
            "cost"
        else
            error("unknown metric type")
        end
    end

    Makie.Colorbar(
        heatmap_grid[heatmap_inset_indices, 2],
        last_info.last_plot;
        label,
        tellwidth = true,
    )

    Makie.rowgap!(heatmap_grid, 0)
    if show_average_inset
        Makie.rowgap!(heatmap_grid, 1, 10)
    end
    Makie.colgap!(heatmap_grid, 5)

    canvas
end

function visualize_openloop_trajectory_example(
    strategy;
    env,
    resolution = (168, 375),
    canvas = Makie.Figure(; resolution),
    title = "",
    visualize_decorations = (axis) -> axis,
    axis_kwargs = (;),
    strategy_viz_kwargs = (;),
    return_info = false,
    hidexlabel = false,
    hideylabel = false,
)
    axis = TrajectoryGamesExamples.create_environment_axis(
        canvas[1, 1],
        env;
        viz_kwargs = (; color = :white, strokewidth = 0),
        xlabel = L"p_\text{lat}",
        ylabel = L"p_\text{lon}",
        ylabelpadding = 5, # hack to work around makie bug
        aspect = Makie.DataAspect(),
        title = title,
        axis_kwargs...,
    )
    visualize_decorations(axis)
    Makie.hidexdecorations!(axis; label = hidexlabel)
    Makie.hideydecorations!(axis; label = hideylabel)

    visualize_joint_contingency_strategy!(axis, strategy; strategy_viz_kwargs...)

    if return_info
        return (; canvas, axis)
    end

    canvas
end

function visualize_strategy_matrix_element(
    run,
    env;
    canvas = Makie.Figure(resolution = (168, 300)),
    preamble = L"\theta = \mathrm{left}) = ",
    show_title = false,
    kwargs...,
)
    title = show_title ? L"b(%$preamble %$(run.weight_distribution[2])" : ""

    visualize_openloop_trajectory_example(
        run.strategy;
        env,
        canvas,
        title,
        kwargs...,
        hidexlabel = true,
        hideylabel = true,
    )
end

function visualize_overtaking_average_tb_sweep_openloop(overtaking_results; kwargs...)
    visualize_average_tb_sweep(overtaking_results["evaluation"]; eval_type = :openloop, kwargs...)
end

function visualize_overtaking_average_tb_sweep_closedloop(overtaking_results; kwargs...)
    visualize_average_tb_sweep(overtaking_results["evaluation"]; eval_type = :closedloop, kwargs...)
end

function visualize_overtaking_closedloop_trajectory_examples(overtaking_results)
    visualize_closedloop_trajectory_banner(
        overtaking_results["raw_results"];
        #scenario_id = 27,
        scenario_id = 28,
        hypothesis = 2,
        #branching_time = 10
        branching_time = 5,
        axis_kwargs = (; limits = ((-0.25, 0.75), (-4.15, 2.0)), xticks = [-0.25, 0.75]),
        resolution = (172, 375),
        visualize_decorations = visualize_overtaking_decorations,
    )
end

function generate_all_overtaking_figures(
    #overtaking_results = JLD2.load("results/1_to_25_overtaking_results.jld2"),
    overtaking_results = JLD2.load("results/overtaking_results.jld2"),
)
    CairoMakie.activate!()
    save_options = (; px_per_unit = 2, pt_per_unit = 1)
    set_rss_theme!()

    Makie.save(
        "results/figures/overtaking_average_tb_sweep_banner_openloop.pdf",
        visualize_overtaking_average_tb_sweep_openloop(overtaking_results);
        save_options...,
    )

    Makie.save(
        "results/figures/overtaking_average_tb_sweep_banner_closedloop.pdf",
        visualize_overtaking_average_tb_sweep_closedloop(overtaking_results);
        save_options...,
    )

    Makie.save(
        "results/figures/overtaking_closedloop_trajectory_examples.pdf",
        visualize_overtaking_closedloop_trajectory_examples(overtaking_results);
        save_options...,
    )

    GLMakie.activate!()
end

function visualize_crosswalk_average_tb_sweep_openloop(crosswalk_results; kwargs...)
    visualize_average_tb_sweep(
        crosswalk_results["evaluation"];
        highlighted_branching_times = [5, 25],
        eval_type = :openloop,
        kwargs...,
    )
end

function visualize_crosswalk_average_tb_sweep_closedloop(crosswalk_results; kwargs...)
    visualize_average_tb_sweep(crosswalk_results["evaluation"]; eval_type = :closedloop, kwargs...)
end

function visualize_crosswalk_spatial(crosswalk_results; kwargs...)
    visualize_spatial_banner(
        crosswalk_results;
        spatial_viz_kwargs = (;
            axis_kwargs = (; limits = ((-1.2, 1.2), (-2.5, 2.5)), xticks = [-1, 1])
        ),
        visualize_decorations = visualize_crosswalk_decorations,
        kwargs...,
    )
end

function visualize_crosswalk_closedloop_trajectory_examples(crosswalk_results)
    visualize_closedloop_trajectory_banner(
        crosswalk_results["raw_results"];
        scenario_id = 33,
        hypothesis = 2,
        branching_time = 5, #10
        #axis_kwargs = (; limits = ((-0.5, 1.2), (-2.2, 1.8)), xticks = [-1, 1]),
        axis_kwargs = (; limits = ((-0.5, 1.2), (-2.5, 2.2)), xticks = [-1, 1]),
        visualize_decorations = visualize_crosswalk_decorations,
    )
end

function set_rss_theme!()
    Makie.set_theme!(;
        figure_padding = 2,
        fontsize = 14,
        fonts = (;
            regular = "URW Palladio",
            italic = "URW Palladio Italic",
            bold = "URW Palladio Bold",
        ),
    )
end

function generate_all_crosswalk_figures(;
    #crosswalk_results = JLD2.load("results/1_to_25_crosswalk_results.jld2"),
    crosswalk_results = nothing, #JLD2.load("results/crosswalk_nonlinear_results.jld2"),
    receding_horizon_crosswalk_benchmark = JLD2.load_object(
        "results/crosswalk_receding_horizon_benchmark.jld2",
    ),
)
    CairoMakie.activate!()
    save_options = (; px_per_unit = 2, pt_per_unit = 1)
    set_rss_theme!()

    if !isnothing(crosswalk_results)
        Makie.save(
            "results/figures/crosswalk_average_tb_sweep_contingency_cost.pdf",
            visualize_crosswalk_average_tb_sweep_openloop(crosswalk_results);
            save_options...,
        )

        Makie.save(
            "results/figures/crosswalk_spatial_contingency_cost.pdf",
            visualize_crosswalk_spatial(crosswalk_results; eval_type = :openloop);
            save_options...,
        )

        #Makie.save(
        #    "results/figures/crosswalk_closedloop_trajectory_examples.pdf",
        #    visualize_crosswalk_closedloop_trajectory_examples(crosswalk_results);
        #    save_options...,
        #)

    end

    if !isnothing(receding_horizon_crosswalk_benchmark)
        Makie.save(
            "results/figures/crosswalk_receding_horizon_benchmark.pdf",
            visualize_receding_horizon_benchmark(receding_horizon_crosswalk_benchmark);
        )
        Makie.save(
            "results/figures/crosswalk_receding_horizon_tb_over_rationality.pdf",
            visualize_tb_over_rationality(receding_horizon_crosswalk_benchmark);
            save_options...,
        )
    end

    GLMakie.activate!()
end

# used for the interactive widget on the website
function generate_individual_crosswalk_strategy_matrix_plots(crosswalk_strategy_matrix)
    CairoMakie.activate!()
    save_options = (; px_per_unit = 2, pt_per_unit = 1)
    set_rss_theme!()

    for (ii, row) in enumerate(crosswalk_strategy_matrix)
        env = row.demo_setup.contingency_game.env
        for (jj, element) in enumerate(row.runs)
            figure = visualize_strategy_matrix_element(
                element,
                env;
                axis_kwargs = (; limits = ((-1.2, 1.2), (-2.2, 2.1)), xticks = [-1, 1]),
                visualize_decorations = visualize_crosswalk_decorations,
            )
            Makie.save(
                "results/figures/strategy_matrix_elements/strategy_matrix_element-$((ii, jj)).pdf",
                figure;
                save_options...,
            )
        end
    end
    GLMakie.activate!()
end
