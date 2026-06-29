classdef L2Monitor < SimulationMonitor
    properties (Access = private)
        Laplacian_hat (1,:) double
        Nx double
        Lx double

        % Tracking arrays
        l2_history double

        step_idx int = 1
    end

    methods
        function obj = L2Monitor(Lx)
            obj.Lx = Lx;
        end

        function initialize(obj, u0, t_grid)
            num_steps = length(t_grid);
            obj.Nx = length(u0);            

            % Preallocate tracking arrays
            obj.l2_history = zeros(1, num_steps);

            obj.update(u0, t_grid(1));
        end

        function update(obj, u_hat, ~)
            obj.l2_history(obj.step_idx) = sqrt(obj.Lx) * norm(u_hat) / obj.Nx;
            obj.step_idx = obj.step_idx + 1;
        end

        function finalize(~)
            return;
        end
    end
end

