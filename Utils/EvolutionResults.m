classdef EvolutionResults
    %EVOLUTIONRESULTS Container for time-series integration results.
    %   S = EvolutionResults(time, solution) constructs an object with a
    %   read-only row vector Time and a read-only Solution matrix.
    %   Code generated with Gemini 3.1 Pro.
    %
    %   The Solution is an (N+1)-dimensional numeric matrix, where the 
    %   LAST dimension corresponds to the time steps.
    %   (i.e., size(solution, ndims(solution)) == numel(time)).
    %
    %   Example (1D Space):
    %       t = 0:0.1:1;                   % 1x11 time vector
    %       y = rand(1024, 11);            % 1024 spatial points x 11 time steps
    %       S = EvolutionResults(t, y);
    %
    %   Example (2D Space):
    %       t = 0:0.1:1;                   % 1x11 time vector
    %       y = rand(256, 256, 11);        % 256x256 spatial grid x 11 time steps
    %       S = EvolutionResults(t, y);

    properties (SetAccess = private)
        % Time vector of evaluation points
        time double

        % Solution matrix (N-dimensional spatial + 1 temporal dimension)
        solution double 
    end

    methods
        function obj = EvolutionResults(time, solution)
            % Validate inputs
            arguments
                time (1,:) double  {mustBeReal, mustBeFinite, mustBeVector} 
                solution double
            end

            % Ensure time is a row vector
            time = reshape(time, 1, []);
            nTime = numel(time);

            % Validate solution dimensions against time
            if isempty(solution)
                if ~isempty(time)
                    error('EvolutionResult:InvalidSolution', ...
                        'Solution is empty but Time is not.');
                end
            else
                % Get the number of dimensions of the solution array
                nd = ndims(solution);
                
                % MATLAB dynamically drops trailing dimensions of size 1.
                % If we only have 1 time step, we bypass the dimension check.
                if nTime > 1
                    lastDimSize = size(solution, nd);
                    
                    if lastDimSize ~= nTime
                        error('EvolutionResult:SizeMismatch', ...
                            'The last dimension of the Solution matrix (%d) must equal length(Time) (%d).', ...
                            lastDimSize, nTime);
                    end
                end
            end

            obj.time = time;
            obj.solution = solution;
        end
    end
end