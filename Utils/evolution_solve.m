function [sol] = evolution_solve(problem, solver, dt, options)
    arguments
        problem EvolutionProblem % TODO: Generalize
        solver Solver % Generalized to support any Solver subclass
        dt double

        options.save_every int32 = 1
        options.monitors {mustBeListOfMonitors} = []
        options.termination_event function_handle = @(u, t) false;
    end

    % Assign the problem to the solver
    solver.init_problem(problem, dt);

    t_grid = problem.tspan(1) : dt : problem.tspan(2);
    save_indices = 1:options.save_every:length(t_grid);
    num_saves = length(save_indices);
  
    %% Create solution history buffer
    % Preallocate an (N+1)-Dimensional Array
    sz = size(problem.u0);
    u_hist = zeros([sz, num_saves]); 

    % Create a generic indexer for the spatial dimensions
    % If u0 is 2D, this creates {':', ':'}
    % If u0 is 3D, this creates {':', ':', ':'}
    spatial_idx = repmat({':'}, 1, ndims(problem.u0));

    
    u = problem.u0;
    
    u_hist(spatial_idx{:}, 1) = u;
    t_hist = t_grid(save_indices);

    %% Initialize monitors
    for i = 1:length(options.monitors)
        options.monitors(i).initialize(problem.u0, t_grid);
    end

    %% Main loop
    save_idx = 1;
    for n = 2 : length(t_grid)
        u = solver.step(u, t_grid(n-1), dt);

        if options.save_every > 0 && mod(n-1, options.save_every) == 0
            save_idx = save_idx + 1;            
            u_hist(spatial_idx{:}, save_idx) = u;
        end
        
        % Update monitors
        if ~isempty(options.monitors)
            for i = 1:length(options.monitors)
                options.monitors(i).update(u, t_grid(n));
            end
        end

        % If termination event is triggered, exit loop
        if ~isempty(options.termination_event) && options.termination_event(u, t_grid(n))
            break;
        end
    end

    %% Finalize monitors
    for i = 1:length(options.monitors)
        options.monitors(i).finalize();
    end

    sol = EvolutionResults(squeeze(t_hist), squeeze(u_hist));
end

function mustBeListOfMonitors(c)
    % Validates that every item inside the cell array inherits from SimulationMonitor
    for i = 1:length(c)
        if ~isa(c(i), 'SimulationMonitor')
            error('Arguments:InvalidMonitor', ...
                'All elements in options.monitors must inherit from SimulationMonitor.');
        end
    end
end