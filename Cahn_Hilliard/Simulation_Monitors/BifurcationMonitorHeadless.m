classdef BifurcationMonitorHeadless < SimulationMonitor
    %BIFURCATIONMONITORNOPLOT A monitor for bifurcation and mode surpass analysis.
    % Records L2 norm and dominant Fourier mode history

    properties (SetAccess = private)
        params struct
        t_grid (1, :) double

        % Tracking arrays
        l2_history double
        dominate_mode double % 2D array: [t, dom_mode]
        dom_mode_ind = 1

        step_idx = 0
    end

    methods
        function obj = BifurcationMonitorHeadless(params)
            % Code assumes mu is a function of time
            if isnumeric(params.mu)
                params.mu = @(t) params.mu * ones(size(t));
            end
            obj.params = params;
        end

        function initialize(obj, u0_hat, t_grid)
            num_steps = length(t_grid);
            obj.t_grid = t_grid;

            % Initial dominant mode tracking
            obj.dominate_mode = NaN(num_steps, 2);
            u0 = real(ifft(u0_hat));
            [~, max_index] = max(abs(u0_hat - mean(u0)));
            obj.dominate_mode(1, :) = [t_grid(1), obj.params.kx(max_index)];
            obj.dom_mode_ind = 1;

            obj.l2_history = zeros(1, num_steps);

            
            obj.update(u0_hat, t_grid(1));
        end

        function update(obj, u_hat, t)
            obj.step_idx = obj.step_idx + 1;
            
            u = real(ifft(u_hat));

            % L2 computation
            obj.l2_history(obj.step_idx) = sqrt(obj.params.Lx) * norm(u_hat) / length(u_hat);

            % Dominate mode computation (excludes the mean mode)
            [~, max_index] = max(abs(u_hat - mean(u)));
            dom_mode = obj.params.kx(max_index);

            if obj.dom_mode_ind == 0 || obj.dominate_mode(obj.dom_mode_ind, 2) ~= dom_mode
                obj.dom_mode_ind = obj.dom_mode_ind + 1;
                obj.dominate_mode(obj.dom_mode_ind, :) = [t, dom_mode];
            end
        end

        function finalize(~)
            
        end

        % Public accessor for dominant mode data (helpful for analytical comparisons in calling scripts)
        function dominate_mode = get_dominate_mode(obj)
            dominate_mode = obj.dominate_mode(1:obj.dom_mode_ind, :);
        end
    end
end
