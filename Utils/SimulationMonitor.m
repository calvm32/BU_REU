classdef (Abstract) SimulationMonitor < handle
    %SIMULATIONMONITOR Abstract base class for observing PDE evolutions.
    
    methods (Abstract)
        % Called once before the loop starts to preallocate arrays and setup plots
        initialize(obj, u0, t_grid)
        
        % Called by the solver at specific intervals (e.g., save_every)
        update(obj, u, t, step_idx)
        
        % Called after the loop finishes (for cleanup, saving videos, etc.)
        finalize(obj)
    end
end