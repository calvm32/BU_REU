classdef EvolutionProblem
    % Stores the linear operator 

    properties (SetAccess = private)
        linearOperator double {mustBeNonempty}
        nonlinearOperator (1,1) function_handle
        u0 double {mustBeNonempty}
        tspan (1,2) double
    end

    methods
        function obj = EvolutionProblem(linearOperator, nonlinearOperator, u0, tspan)
            arguments
                linearOperator double {mustBeNonempty}
                nonlinearOperator (1,1) function_handle
                u0 double {mustBeNonempty}
                tspan (1,2) double
            end

            obj.linearOperator = linearOperator;
            obj.nonlinearOperator = nonlinearOperator;
            obj.u0 = u0;
            obj.tspan = tspan;
        end
    end
end
