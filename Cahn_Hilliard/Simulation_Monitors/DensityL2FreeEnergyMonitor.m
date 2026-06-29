classdef DensityL2FreeEnergyMonitor < SimulationMonitor
    properties (Access = private)
        params struct
        t_grid (1, :) double

        % Tracking arrays
        l2_history double
        energy_history
        theoretical_density double
        computed_density double

        step_idx = 1

        % Plotting
        fig
        ax2
        ax3
        u_line
        l2_line
        energy_line
        computed_density_line
        theory_density_line
        hTitle1

        % Video
        v
    end

    methods
        function obj = DensityL2FreeEnergyMonitor(params)
            obj.params = params;
        end

        function initialize(obj, u0_hat, t_grid)
            num_steps = length(t_grid);
            obj.t_grid = t_grid;

            %% Preallocate tracking arrays
            obj.l2_history = zeros(1, num_steps);
            obj.theoretical_density = zeros(1, num_steps);
            obj.computed_density = zeros(1, num_steps);

            %% Plot setup
            obj.fig = figure('Position',[100 100 1400 450],'Resize', 'off');

            %% Subplot 1: Solution
            ax1 = subplot(2,2,1);
            obj.u_line = plot(ax1, obj.params.x, zeros(size(obj.params.x)), 'LineWidth', 1.5);
            ylim(ax1, [-1.2 1.2]); 
            xlabel(ax1, 'x'); 
            ylabel(ax1, 'u');
            obj.hTitle1 = title(ax1, sprintf('t = %.3f, \\mu = %.3f', 0, obj.params.mu(0)));
            obj.hTitle1.Interpreter = 'tex';
            grid(ax1, 'on');

            %% Subplot 2: L2 norm (semilogy)
            obj.ax2 = subplot(2,2,2);
            obj.l2_line = semilogy(obj.ax2, 0, NaN, 'LineWidth', 2);
            xlabel(obj.ax2, 't'); 
            ylabel(obj.ax2, '||u||_{L^2}');
            title(obj.ax2, 'L2 Norm of u');
            grid(obj.ax2, 'on');

            %% Subplot 3: Domain Wall Density
            obj.ax3 = subplot(2,2,3);
            obj.theory_density_line = plot(obj.ax3, NaN, NaN, ...
                'LineWidth',2,'DisplayName','theoretical');
            hold(obj.ax3, 'on');
            obj.computed_density_line = plot(obj.ax3, NaN, NaN, ...
                'LineWidth',2,'DisplayName','computed');
            title(obj.ax3, 'Domain Wall Densities');
            xlabel(obj.ax3, 't');
            ylabel(obj.ax3, 'n(t)');
            grid(obj.ax3, 'on');
            legend(obj.ax3, 'show');

            %% Subplot 4: Free Energy
            ax4 = subplot(2,2,4);
            obj.energy_line = plot(ax4, NaN, NaN, 'LineWidth',2);
            title(ax4, 'Free Energy');
            xlabel(ax4, 't');
            ylabel(ax4, 'E(t)');
            grid(ax4, 'on');

            %% Setup video
            if obj.params.save_video
                obj.v = VideoWriter(obj.params.video_filename, 'MPEG-4');
                obj.v.FrameRate = 30;
                obj.v.Quality = 100;
                open(obj.v);
            end

            obj.update(u0_hat, t_grid(1));
        end

        function update(obj, u_hat, t)
            u = real(ifft(u_hat));

            % L2 norm
            obj.l2_history(obj.step_idx) = sqrt(obj.params.Lx) * norm(u_hat) / obj.params.Nx;

            % Free energy
            ux = gradient(u, obj.params.dx);
            density = 0.25*u.^4 - 0.5*obj.params.mu(t)*u.^2 + 0.5*ux.^2;
            obj.energy_history(obj.step_idx) = sum(density)*obj.params.dx;

            % Theoretical density via power spectrum
            ksq = -obj.params.Laplacian_hat;
            P = abs(u_hat).^2 .* exp(t^2*ksq*obj.params.epsilon - 2*t*ksq.^2);
            num = sum(ksq.*P, 'all');
            den = sum(P, 'all');
            obj.theoretical_density(obj.step_idx) = (1/pi)*sqrt(num/den);

            % Computed density
            walls = sum(u .* circshift(u,-1) < 0);
            obj.computed_density(obj.step_idx) = walls / obj.params.Lx;

            if mod(obj.step_idx - 1, obj.params.plot_every) == 0
                obj.u_line.YData = u;

                obj.hTitle1.String = sprintf('t = %.3f, \\mu = %.3f', t, obj.params.mu(t));
                obj.hTitle1.Interpreter = 'tex';

                obj.l2_line.XData = obj.t_grid(1:obj.step_idx);
                obj.l2_line.YData = obj.l2_history(1:obj.step_idx);
                axis(obj.ax2, 'tight');

                blowup_mask = obj.t_grid(1:obj.step_idx) >= obj.params.blowup_time;
                obj.theory_density_line.XData = obj.t_grid(blowup_mask);
                obj.theory_density_line.YData = obj.theoretical_density(blowup_mask);
                obj.computed_density_line.XData = obj.t_grid(blowup_mask);
                obj.computed_density_line.YData = obj.computed_density(blowup_mask);
                axis(obj.ax3, 'tight');

                obj.energy_line.XData = obj.t_grid(1:obj.step_idx);
                obj.energy_line.YData = obj.energy_history(1:obj.step_idx);

                drawnow limitrate

                if obj.params.save_video && ~isempty(obj.v)
                    frame = getframe(obj.fig);
                    writeVideo(obj.v, frame);
                end
            end

            obj.step_idx = obj.step_idx + 1;
        end

        function finalize(obj)
            if obj.params.save_video && ~isempty(obj.v)
                close(obj.v);
            end
        end
    end
end
