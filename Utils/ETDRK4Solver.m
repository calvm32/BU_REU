classdef ETDRK4Solver < Solver
    %ETDRK4Solver A ETDRK4 time stepper for solving odes/evolution pdes.
    %   The parameter M (default M=16) specifies how many samples are taken during initial
    %   contour integration.
    
    properties
        M int32 {mustBePositive, mustBeFinite} = 16;
    end

    properties (Access = private)
        % Constants 
        E (1, :) double
        E2 (1, :) double
        Q (1, :) double
        f1 (1, :) double
        f2 (1, :) double
        f3 (1, :) double
    end

    methods
        function obj = ETDRK4Solver(M)
            % Validate inputs
            arguments
                M int32 
            end

            obj.M = M;
        end

        function obj = init_problem(obj, problem, dt)
            arguments
                obj
                problem EvolutionProblem 
                dt double
            end

            % Compute ETDRK4 coefficients (vectorize over entries of L_operator)
            obj.E  = exp(dt * problem.linearOperator);
            obj.E2 = exp(dt * problem.linearOperator / 2);

            % contour integration constants
            dM = double(obj.M);
            r = exp(1i*pi*((1:dM)-0.5)/dM);
            Lvec = problem.linearOperator(:);
            LR = dt*Lvec(:,ones(obj.M,1)) + r(ones(numel(problem.linearOperator),1),:);

            obj.Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
            obj.f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3,2)).';
            obj.f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3,2)).';
            obj.f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3,2)).';            

            obj.problem = problem;
        end

        function [up] = step(obj, u, t, dt)
            arguments
                obj
                u 
                t double
                dt double
            end

            if isempty(obj.problem)
                error("The problem has not been set");
            end

            tmid = t + 0.5 * dt; 
            tend = t + dt;

            % Stage 1
            Nu = obj.problem.nonlinearOperator(u, t);

            % Stage 2
            a = obj.E2 .* u + obj.Q .* Nu;
            Na = obj.problem.nonlinearOperator(a, tmid);

            % Stage 3
            b = obj.E2 .* u + obj.Q .* Na;
            Nb = obj.problem.nonlinearOperator(b, tmid);

            % Stage 4
            c = obj.E2 .* a + obj.Q .* (2*Nb - Nu);
            Nc = obj.problem.nonlinearOperator(c, tend);

            % Final Time Step Combination
            up = obj.E .* u + obj.f1 .* Nu ...
                + 2*obj.f2 .* (Na + Nb) + obj.f3 .* Nc;
        end
    end
end
