classdef DensityMonitor < SimulationMonitor
    properties (Access = private)
        params struct
        t_grid (1, :) double

        % Tracking arrays
        theoretical_density double
        computed_density double

        step_idx = 1
    end

    methods
        function obj = DensityMonitor(params)
            obj.params = params;
        end

        function initialize(obj, u0, t_grid)
            num_steps = length(t_grid);
            obj.t_grid = t_grid;

            % Preallocate tracking arrays
            obj.theoretical_density = zeros(1, num_steps);
            obj.computed_density = zeros(1, num_steps);

            obj.update(u0, t_grid(1));
        end

        function update(obj, u_hat, t)
            u = real(ifft(u_hat));

            % Compute power spectrum
            ksq = -obj.params.Laplacian_hat;
            P = abs(u_hat).^2 .* exp(t^2*ksq*obj.params.epsilon - 2*t*ksq.^2);

            % Theoretical density
            num = sum(ksq.*P, 'all');
            den = sum(P, 'all');
            obj.theoretical_density(obj.step_idx) = (1/pi)*sqrt(num/den);

            % Computed density
            walls = sum(u .* circshift(u,-1) < 0);
            obj.computed_density(obj.step_idx) = walls / obj.params.Lx;

            obj.step_idx = obj.step_idx + 1;
        end

        function finalize(obj)
            return;
        end
    end
end
