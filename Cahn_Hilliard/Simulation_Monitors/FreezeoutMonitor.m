classdef FreezeoutMonitor < SimulationMonitor
    %FREEZEOUTMONITOR A monitor for Cahn-Hilliard freezeout analysis.
    % Tracks domain wall densities and Fourier spectrum, marking freeze and 
    % equilibrium points on finalization.

    properties (Access = private)
        params struct
        t_grid (1, :) double

        % Tracking arrays
        l2_history double
        computed_density double
        theory_density double

        step_idx = 1

        % Plotting handles
        fig
        u_line
        theory_density_line
        computed_density_line
        freeze_points
        eq_points
        fourier_line
        peak_point_comput
        peak_point_theory
        ax1
        ax3
        ax4
        hTitle1
        k_shift

        % Video
        v
    end

    methods
        function obj = FreezeoutMonitor(params)
            obj.params = params;
        end

        function initialize(obj, u0_hat, t_grid)
            num_steps = length(t_grid);
            obj.t_grid = t_grid;

            % Preallocate tracking arrays
            obj.l2_history = zeros(1, num_steps);
            obj.computed_density = zeros(1, num_steps);
            obj.theory_density = zeros(1, num_steps);

            % Plot setup
            obj.fig = figure('Position',[100 100 1100 1100], 'Resize', 'off');

            % Subplot 1: Solution
            obj.ax1 = subplot(2,2,[1,2]);
            u0 = real(ifft(u0_hat));
            obj.u_line = plot(obj.ax1, obj.params.x, u0, 'LineWidth', 1.5);
            ylim(obj.ax1, [-1.2 1.2]); 
            xlabel(obj.ax1, 'x'); 
            ylabel(obj.ax1, 'u(x)');
            obj.hTitle1 = title(obj.ax1, sprintf('t = %.3f, \\mu = %.3f, \\epsilon = %.2f', t_grid(1), obj.params.mu(t_grid(1)), obj.params.epsilon));
            obj.hTitle1.Interpreter = 'tex';
            grid(obj.ax1, 'on');

            % Subplot 2: Domain Wall Density
            obj.ax3 = subplot(2,2,3);
            obj.theory_density_line = plot(obj.ax3, NaN, NaN, 'LineWidth',2,'DisplayName','theoretical');
            hold(obj.ax3, 'on');
            obj.computed_density_line = plot(obj.ax3, NaN, NaN, 'LineWidth',2,'DisplayName','computed');

            obj.freeze_points = plot(obj.ax3, NaN, NaN, ...
                'ro', 'MarkerSize', 8, 'LineWidth', 2, 'Color', 'black', 'DisplayName', '$\hat{t}$');
            obj.eq_points = plot(obj.ax3, NaN, NaN, ...
                'rs', 'MarkerSize', 8, 'LineWidth', 2, 'Color', 'black', 'DisplayName', '$\hat{t}_{eq}$');

            title(obj.ax3, 'Domain Wall Densities');
            xlabel(obj.ax3, 't'); 
            ylabel(obj.ax3, 'n(t)');
            lgd = legend(obj.ax3, ...
                [obj.theory_density_line, obj.computed_density_line, obj.freeze_points, obj.eq_points], ...
                {'theoretical','computed','$\hat{t}$','$\hat{t}_{eq}$'}, ...
                'Location','northwest');
            lgd.Interpreter = 'latex';
            fontsize(lgd, scale=1.4);
            grid(obj.ax3, 'on');

            % Subplot 3: Fourier Modes (semilogy)
            obj.ax4 = subplot(2,2,4);
            obj.k_shift = fftshift(obj.params.kx);
            u0_hat_shift = fftshift(u0_hat);
            [~, idx] = max(abs(u0_hat_shift).^2);

            obj.fourier_line = semilogy(obj.ax4, obj.k_shift, abs(u0_hat_shift).^2, 'LineWidth', 1.5);
            obj.fourier_line.HandleVisibility = 'off';
            hold(obj.ax4, 'on');

            obj.peak_point_comput = semilogy(obj.ax4, obj.k_shift(idx), abs(u0_hat_shift(idx)).^2, ...
                'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'computed mode');

            k_theory = obj.params.epsilon^(1/6);
            [~, idx_theory] = min(abs(obj.k_shift - k_theory));
            obj.peak_point_theory = semilogy(obj.ax4, obj.k_shift(idx_theory), abs(u0_hat_shift(idx_theory)).^2, ...
                'o', 'MarkerFaceColor', 'g', 'DisplayName', 'theoretical mode');

            xlabel(obj.ax4, 'k');
            ylabel(obj.ax4, '$\|\hat{u}(k)\|^2$', 'Interpreter', 'latex');
            title(obj.ax4, 'Fourier Spectrum and Dominant Modes');
            lgd2 = legend(obj.ax4, 'show', 'Location', 'northwest');
            lgd2.Interpreter = 'latex';
            fontsize(lgd2, scale=1.4);
            grid(obj.ax4, 'on');

            % Video setup
            if isfield(obj.params, 'save_video') && obj.params.save_video
                obj.v = VideoWriter(obj.params.video_filename, 'MPEG-4');
                obj.v.FrameRate = 15;
                obj.v.Quality = 100;
                open(obj.v);
            end

            obj.update(u0_hat, t_grid(1));
        end

        function update(obj, u_hat, t)
            u = real(ifft(u_hat));

            % Computed density
            walls = sum(u .* circshift(u,-1) < 0);
            obj.computed_density(obj.step_idx) = walls / obj.params.Lx;

            % Theoretical density via power spectrum
            ksq = -obj.params.Laplacian_hat;
            P = abs(u_hat).^2 .* exp(t^2 * ksq * obj.params.epsilon - 2 * t * ksq.^2);
            num = sum(ksq .* P, 'all');
            den = sum(P, 'all');
            obj.theory_density(obj.step_idx) = (1/pi) * sqrt(num/den);

            obj.l2_history(obj.step_idx) = sqrt(obj.params.Lx) * norm(u_hat) / obj.params.Nx;

            % Update plots
            if mod(obj.step_idx - 1, obj.params.plot_every) == 0
                obj.u_line.YData = u;
                obj.hTitle1.String = sprintf('t = %.3f, \\mu = %.3f, \\epsilon = %.2f', t, obj.params.mu(t), obj.params.epsilon);

                blowup_mask = obj.t_grid(1:obj.step_idx) >= obj.params.blowup_time;
                obj.theory_density_line.XData = obj.t_grid(blowup_mask);
                obj.theory_density_line.YData = obj.theory_density(blowup_mask);
                obj.computed_density_line.XData = obj.t_grid(blowup_mask);
                obj.computed_density_line.YData = obj.computed_density(blowup_mask);
                axis(obj.ax3, 'tight');

                % Update Fourier spectrum (semilogy handles zeros cleanly)
                u_hat_shift = fftshift(u_hat);
                spectrum = abs(u_hat_shift).^2;
                obj.fourier_line.YData = spectrum;

                [~, idx] = max(spectrum);
                obj.peak_point_comput.XData = obj.k_shift(idx);
                obj.peak_point_comput.YData = spectrum(idx);

                k_theory = obj.params.epsilon^(1/6);
                [~, idx_theory] = min(abs(obj.k_shift - k_theory));
                obj.peak_point_theory.XData = obj.k_shift(idx_theory);
                obj.peak_point_theory.YData = spectrum(idx_theory);
                axis(obj.ax4, 'tight');

                drawnow;

                if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v)
                    frame = getframe(obj.fig, [0 0 1100 1100]);
                    writeVideo(obj.v, frame);
                end
            end

            obj.step_idx = obj.step_idx + 1;
        end

        function finalize(obj)
            % Mark freeze and equilibrium points
            mask = obj.t_grid > 10;
            density_masked = obj.computed_density(mask);
            max_density = max(density_masked);
            tol = 1e-4 * max_density;

            mask_idx = find(mask);
            eq_idx_local = find(density_masked >= max_density - tol, 1, 'first');
            if ~isempty(eq_idx_local)
                eq_idx = mask_idx(eq_idx_local);
                obj.eq_points.XData = obj.t_grid(eq_idx);
                obj.eq_points.YData = obj.computed_density(eq_idx);
            end

            [~, freeze_idx] = min(abs(obj.t_grid - obj.params.blowup_time));
            obj.freeze_points.XData = obj.t_grid(freeze_idx);
            obj.freeze_points.YData = obj.theory_density(freeze_idx);

            axis(obj.ax4, 'tight');
            drawnow;

            if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v)
                frame = getframe(obj.fig, [0 0 1100 1100]);
                writeVideo(obj.v, frame);
                close(obj.v);
            end
        end
    end
end
