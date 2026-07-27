classdef CH2DMonitor < SimulationMonitor
    %CH2DMonitor A monitor for 2D Cahn-Hilliard simulations.
    % Supports solution plotting (imagesc), L2 norm history (semilogy), and either 
    % defect/droplet counting or 2D domain wall density.

    properties (Access = private)
        params struct
        t_grid (1, :) double

        % Tracking arrays
        l2_history double
        defect_history double
        defect_max_history double
        defect_min_history double
        theory_density_history double
        comput_density_history double

        step_idx = 1

        % Plotting handles
        fig
        im
        ax1
        ax2
        ax3
        hline
        dline
        dlinemax
        dlinemin
        theory_density_line
        comput_density_line

        % Video
        v
    end

    methods
        function obj = CH2DMonitor(params)
            obj.params = params;
        end

        function initialize(obj, u0_hat, t_grid)
            num_steps = length(t_grid);
            obj.t_grid = t_grid;

            % Preallocate tracking arrays
            obj.l2_history = zeros(1, num_steps);
            if strcmp(obj.params.mode, 'count_droplets')
                obj.defect_history = zeros(1, num_steps);
            elseif strcmp(obj.params.mode, 'count_minmax')
                obj.defect_max_history = zeros(1, num_steps);
                obj.defect_min_history = zeros(1, num_steps);
            elseif strcmp(obj.params.mode, 'domain_wall_density')
                obj.theory_density_history = zeros(1, num_steps);
                obj.comput_density_history = zeros(1, num_steps);
            end

            % Plot setup
            obj.fig = figure('Position',[100 100 1800 600]);

            % Subplot 1: 2D solution
            obj.ax1 = subplot(1,3,1);
            u_phys = real(ifft2(u0_hat));
            obj.im = imagesc(obj.ax1, obj.params.x, obj.params.y, u_phys');
            axis(obj.ax1, 'equal', 'tight');
            set(obj.ax1, 'YDir', 'normal');
            colormap(obj.ax1, obj.seismic_colormap());
            caxis(obj.ax1, [-1.2 1.2]);
            colorbar(obj.ax1);
            title(obj.ax1, sprintf('t = %.3f', t_grid(1)));
            L = obj.params.L;
            xticks(obj.ax1, [-L*pi 0 scale*pi]);
            xticklabels(obj.ax1, {['-' num2str(scale,'%.0f') '\pi'], '0', [num2str(scale,'%.0f') '\pi']});
            yticks(obj.ax1, [-scale*pi 0 scale*pi]);
            yticklabels(obj.ax1, {['-' num2str(scale,'%.0f') '\pi'], '0', [num2str(scale,'%.0f') '\pi']});

            % Subplot 2: L2 norm (semilogy)
            obj.ax2 = subplot(1,3,2);
            obj.hline = semilogy(obj.ax2, t_grid(1), NaN, 'LineWidth', 2);
            xlabel(obj.ax2, 't');
            ylabel(obj.ax2, '||u||_{L^2}');
            title(obj.ax2, 'L2 Norm');
            grid(obj.ax2, 'on');

            % Subplot 3: Defect count or wall density
            obj.ax3 = subplot(1,3,3);
            if strcmp(obj.params.mode, 'count_droplets')
                obj.dline = plot(obj.ax3, NaN, NaN, 'LineWidth', 2);
                xlabel(obj.ax3, 't');
                ylabel(obj.ax3, '# negative droplets');
                title(obj.ax3, 'Droplet Count');
            elseif strcmp(obj.params.mode, 'count_minmax')
                obj.dlinemax = plot(obj.ax3, NaN, NaN, 'LineWidth', 2, 'DisplayName', '# maxima');
                hold(obj.ax3, 'on');
                obj.dlinemin = plot(obj.ax3, NaN, NaN, 'LineWidth', 2, 'DisplayName', '# minima');
                xlabel(obj.ax3, 't');
                title(obj.ax3, 'Extrema Count');
                legend(obj.ax3, 'show');
            elseif strcmp(obj.params.mode, 'domain_wall_density')
                obj.theory_density_line = plot(obj.ax3, NaN, NaN, 'LineWidth', 2, 'DisplayName', 'theoretical');
                hold(obj.ax3, 'on');
                obj.comput_density_line = plot(obj.ax3, NaN, NaN, 'LineWidth', 2, 'DisplayName', 'computed');
                xlabel(obj.ax3, 't');
                ylabel(obj.ax3, 'density');
                title(obj.ax3, 'Domain Wall Density');
                legend(obj.ax3, 'show');
            end
            grid(obj.ax3, 'on');

            % Video setup
            if isfield(obj.params, 'save_video') && obj.params.save_video
                obj.v = VideoWriter(obj.params.video_filename, 'Motion JPEG AVI');
                obj.v.FrameRate = 30;
                obj.v.Quality = 100;
                open(obj.v);
            end

            obj.update(u0_hat, t_grid(1));
        end

        function update(obj, u_hat, t)
            u_phys = real(ifft2(u_hat));

            % L2 norm (2D periodic)
            Nx = size(u_hat, 1);
            Ny = size(u_hat, 2);
            obj.l2_history(obj.step_idx) = sqrt(obj.params.Lx * obj.params.Ly) * ...
                sqrt(sum(abs(u_hat).^2, 'all')) / (Nx * Ny);

            if strcmp(obj.params.mode, 'count_droplets')
                BW = bwareaopen(u_phys < 0, 20);
                CC = bwconncomp(BW, 8);
                obj.defect_history(obj.step_idx) = CC.NumObjects;

            elseif strcmp(obj.params.mode, 'count_minmax')
                obj.defect_max_history(obj.step_idx) = nnz(imregionalmax(u_phys) & (u_phys > 0.5));
                obj.defect_min_history(obj.step_idx) = nnz(imregionalmin(u_phys) & (u_phys < -0.5));

            elseif strcmp(obj.params.mode, 'domain_wall_density')
                perim = bwperim(u_phys > 0, 8);
                Lwall = sum(perim(:)) * obj.params.dx;
                obj.comput_density_history(obj.step_idx) = Lwall / (obj.params.Lx * obj.params.Ly);
                ksq = -obj.params.Laplacian_k;
                P = abs(u_hat).^2 .* exp(t^2 * ksq * obj.params.epsilon - 2 * t * ksq.^2);
                num = sum(ksq .* P, 'all');
                den = sum(P, 'all');
                obj.theory_density_history(obj.step_idx) = (pi/2) * sqrt(num/den);
            end

            % Update plots
            if mod(obj.step_idx - 1, obj.params.plot_every) == 0
                set(obj.im, 'CData', u_phys');
                title(obj.ax1, sprintf('t = %.3f', t));

                set(obj.hline, 'XData', obj.t_grid(1:obj.step_idx), 'YData', obj.l2_history(1:obj.step_idx));
                axis(obj.ax2, 'tight');

                if strcmp(obj.params.mode, 'count_droplets')
                    set(obj.dline, 'XData', obj.t_grid(1:obj.step_idx), 'YData', obj.defect_history(1:obj.step_idx));
                elseif strcmp(obj.params.mode, 'count_minmax')
                    set(obj.dlinemax, 'XData', obj.t_grid(1:obj.step_idx), 'YData', obj.defect_max_history(1:obj.step_idx));
                    set(obj.dlinemin, 'XData', obj.t_grid(1:obj.step_idx), 'YData', obj.defect_min_history(1:obj.step_idx));
                elseif strcmp(obj.params.mode, 'domain_wall_density')
                    blowup_mask = obj.t_grid(1:obj.step_idx) >= obj.params.blowup_time;
                    set(obj.theory_density_line, 'XData', obj.t_grid(blowup_mask), 'YData', obj.theory_density_history(blowup_mask));
                    set(obj.comput_density_line, 'XData', obj.t_grid(blowup_mask), 'YData', obj.comput_density_history(blowup_mask));
                end
                axis(obj.ax3, 'tight');

                drawnow;

                if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v)
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
        end
    end

    methods (Access = private)
        function cmap = seismic_colormap(~)
            n = 256;
            r = [(0:n/2-1)/(n/2), ones(1,n/2)];
            g = [(0:n/2-1)/(n/2), (n/2-1:-1:0)/(n/2)];
            b = [ones(1,n/2), (n/2-1:-1:0)/(n/2)];
            cmap = [r(:), g(:), b(:)];
        end
    end
end
