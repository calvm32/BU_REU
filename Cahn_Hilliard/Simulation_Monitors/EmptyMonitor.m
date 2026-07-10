classdef EmptyMonitor < SimulationMonitor
    properties (Access = private)
        ...
    end

    methods
        function L2Monitor(~)
        end

        function initialize(~, ~, ~)
        end

        function update(~, ~, ~)
        end

        function finalize(~)
        end
    end
end

