classdef BifurcationMonitor < SimulationMonitor
    %BIFURCATIONMONITOR A monitor for bifurcation and mode surpass analysis.
    % Displays the solution, dominant Fourier mode history, and the Fourier spectrum.

    properties (Access = private)
        params struct
        t_grid (1, :) double

        % Tracking arrays
        dominate_mode double % 2D array: [t, dom_mode]
        dom_mode_ind = 1

        step_idx = 1

        % Plotting handles
        fig
        hLine1
        hLine3
        hFreq
        ax1
        ax3
        ax4
        hTitle1

        % Video
        v
    end

    methods
        function obj = BifurcationMonitor(params)
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

            % Plot setup
            obj.fig = figure('Position',[100 100 1200 700], 'Resize', 'off');

            % Subplot 1: Solution (Spans the entire top row)
            obj.ax1 = subplot(2,2,[1 2]);
            obj.hLine1 = plot(obj.ax1, obj.params.x, u0, 'LineWidth', 1.5);
            ylim(obj.ax1, 1.2 * [-sqrt(obj.params.mu) sqrt(obj.params.mu)]); 
            xlabel(obj.ax1, 'x'); 
            ylabel(obj.ax1, 'u');
            obj.hTitle1 = title(obj.ax1, sprintf('t = %.3f', t_grid(1)));
            grid(obj.ax1, 'on');

            % Subplot 3: Dominant mode
            obj.ax3 = subplot(2,2,3);
            obj.hLine3 = stairs(obj.ax3, obj.dominate_mode(1, 1), obj.dominate_mode(1, 2), 'LineWidth', 2);
            labels = cellstr("k_" + (0:numel(obj.params.k_j)-1));
            yline(obj.ax3, obj.params.k_j, '--r', labels, 'LineWidth', 2);    
            xlabel(obj.ax3, 't'); 
            ylabel(obj.ax3, 'Wavenumber (k)');
            if isfield(obj.params, 'log_xscale') && obj.params.log_xscale
                set(obj.ax3, 'XScale', 'log');
                xlim(obj.ax3, [max(1e-2, t_grid(1)), t_grid(end)]);
            else
                xlim(obj.ax3, [t_grid(1), t_grid(end)]);
            end
            ylim(obj.ax3, [0, obj.params.k_j(end)]); 
            grid(obj.ax3, 'on');

            % Subplot 4: Fourier Spectrum (semilogy: handles zeros cleanly)
            obj.ax4 = subplot(2,2,4);
            obj.hFreq = semilogy(obj.ax4, ifftshift(obj.params.kx), ifftshift(abs(u0_hat)), 'LineWidth', 2);
            xlabel(obj.ax4, 'k'); 
            ylabel(obj.ax4, '$|\hat{u}|$', 'Interpreter', 'latex');
            xlim(obj.ax4, [-10, 10]);
            grid(obj.ax4, 'on');

            drawnow;

            % Video setup
            if isfield(obj.params, 'save_video') && obj.params.save_video
                obj.v = VideoWriter(obj.params.video_filename, 'MPEG-4');
                obj.v.FrameRate = obj.params.video_framerate;
                obj.v.Quality = 100;
                open(obj.v);
            end

            obj.update(u0_hat, t_grid(1));
        end

        function update(obj, u_hat, t)
            u = real(ifft(u_hat));

            % Dominate mode computation
            [~, max_index] = max(abs(u_hat - mean(u)));
            dom_mode = obj.params.kx(max_index);

            if obj.dom_mode_ind == 0 || obj.dominate_mode(obj.dom_mode_ind, 2) ~= dom_mode
                obj.dom_mode_ind = obj.dom_mode_ind + 1;
                obj.dominate_mode(obj.dom_mode_ind, :) = [t, dom_mode];
            end

            % Update plots
            if mod(obj.step_idx - 1, obj.params.plot_every) == 0
                obj.hLine1.YData = u;
                obj.hTitle1.String = sprintf('t = %.4f', t);

                valid_modes = obj.dominate_mode(1:obj.dom_mode_ind, :);
                set(obj.hLine3, 'XData', [valid_modes(:, 1); t], 'YData', [valid_modes(:, 2); valid_modes(end, 2)]);

                obj.hFreq.YData = ifftshift(abs(u_hat));

                drawnow;

                if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v) && isgraphics(obj.fig)
                    frame = getframe(obj.fig);
                    writeVideo(obj.v, frame);
                end
            end

            obj.step_idx = obj.step_idx + 1;
        end

        function finalize(obj)
            if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v)
                close(obj.v);
            end
            if isgraphics(obj.fig)
                close(obj.fig);
            end
        end

        % Public accessor for dominant mode data (helpful for analytical comparisons in calling scripts)
        function dominate_mode = get_dominate_mode(obj)
            dominate_mode = obj.dominate_mode(1:obj.dom_mode_ind, :);
        end
    end
end
