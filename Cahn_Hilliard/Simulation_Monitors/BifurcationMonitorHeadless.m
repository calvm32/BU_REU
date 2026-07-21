classdef BifurcationMonitorHeadless < SimulationMonitor
    % BIFURCATIONMONITORNOPLOT A monitor for bifurcation and mode surpass analysis.
    % Records L2 norm and dominant Fourier mode history

    properties (SetAccess = private)
        mu
        kx (1, :) double
        Lx (1, 1) double
        Nx (1, 1) double
        t_grid (1, :) double
        mu_grid (1, :) double
        num_amplitudes double
        subtract_mass (1, 1) logical

        % Tracking arrays
        l2_history double
        dominate_mode double % 2D array: [t, dom_mode]
        dom_mode_ind = 1
        amp_history double
        mass double

        step_idx = 0
    end

    methods
        function obj = BifurcationMonitorHeadless(mu, kx, Lx, Nx, num_amplitudes, options)
            arguments
                mu
                kx (1, :) double
                Lx (1, 1) double
                Nx (1, 1) double
                num_amplitudes double = 5
                options.subtract_mass (1, 1) logical = false
            end
            if isnumeric(mu)
                obj.mu = @(t) mu * ones(size(t));
            else
                obj.mu = mu;
            end
            obj.kx = kx;
            obj.Lx = Lx;
            obj.Nx = Nx;
            obj.num_amplitudes = num_amplitudes;
            obj.subtract_mass = options.subtract_mass;
        end

        function initialize(obj, u0_hat, t_grid)            
            if obj.subtract_mass
                u0_hat(1) = 0;
            end

            obj.mass = u0_hat(1) / obj.Nx;
            num_steps = length(t_grid);
            obj.t_grid = t_grid;
            obj.mu_grid = obj.mu(t_grid);

            % Initial dominant mode tracking
            obj.dominate_mode = NaN(20, 2);
            u0 = real(ifft(u0_hat));
            [~, max_index] = max(abs(u0_hat - mean(u0)));
            obj.dominate_mode(1, :) = [t_grid(1), obj.kx(max_index)];
            obj.dom_mode_ind = 1;

            obj.l2_history = zeros(1, num_steps);
            obj.amp_history = zeros(obj.num_amplitudes, num_steps); % TODO: Add 5 to parameters

            obj.update(u0_hat, t_grid(1));
        end

        function update(obj, u_hat, t)
            
            if obj.subtract_mass
                u_hat(1) = 0;
            end

            obj.step_idx = obj.step_idx + 1;

            % L2 computation
            obj.l2_history(obj.step_idx) = sqrt(obj.Lx) * norm(u_hat) / length(u_hat);

            % Amplitude history saving
            obj.amp_history(:, obj.step_idx) = abs(u_hat(2:obj.num_amplitudes+1) / obj.Nx); % TODO: Update so that 5 is a parameter

            % Dominate mode computation (excludes the mean mode)
            u_hat_ac = u_hat;
            u_hat_ac(1) = 0;
            [~, max_index] = max(abs(u_hat_ac));
            dom_mode = obj.kx(max_index);

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
