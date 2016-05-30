classdef PuckWorld < matlab.mixin.SetGet
    properties
		puck
		green
		red 
		rad = 0.05
		t = 0
        agent
        showevery
        render 
	end


	methods

	    function obj = reset(obj)
	    	obj.puck.x = rand();
	    	obj.puck.y = rand();
	    	obj.puck.vx = rand() * 0.05 - 0.025;
	    	obj.puck.vy = rand() * 0.05 - 0.025;
	    	obj.green.x = rand();
	    	obj.green.y = rand();
	    	obj.red.x = rand();
	    	obj.red.y = rand();
	    	obj.red.badrad = 0.25;
	    	obj.showevery = 1;
	    	obj.render.tderr = []; 
	    	obj.render.tr = [];

	    	p.alpha = 0.01;
	    	p.gamma = 0.9;
	    	p.clamp = 1;
	    	p.experience_size = 5000;
	    	p.experience_add_every = 10;
	    	p.learning_steps_per_iteration = 30;
	    	p.nnsz = [obj.get_num_states() 100 obj.get_num_actions()];
	    	p.env = obj;
            obj.agent = DQNAgent(p);
	    end


	    function n = get_num_states(obj)
            n = 8;
	    end


	    function n = get_num_actions(obj)
            n = 4;
	    end


	    function s = get_state(obj)
	    	s = [obj.puck.x - 0.5, obj.puck.y - 0.5,...
	    	     obj.puck.vx * 10, obj.puck.vy * 10,...
	    	     obj.green.x - 0.5, obj.green.y - 0.5,...
	    	     obj.red.x - 0.5, obj.red.y - 0.5];
	    end


	    function [s, r] = sampleNextState(obj, a)
	    	% World dynamics
	    	obj.puck.x = obj.puck.x + obj.puck.vx; % Position + Velocity
	    	obj.puck.y = obj.puck.y + obj.puck.vy;
	    	obj.puck.vx = obj.puck.vx * 0.95; % Damping
	    	obj.puck.vy = obj.puck.vy * 0.95;

            accel = 0.002;
            switch a
                case 1
                    obj.puck.vy = obj.puck.vy + accel;
                case 2
                    obj.puck.vy = obj.puck.vy - accel;
                case 3
                    obj.puck.vx = obj.puck.vx - accel;
                case 4
                    obj.puck.vx = obj.puck.vx + accel;
                case 5
                    obj.puck.vx = obj.puck.vx + accel;
                    obj.puck.vy = obj.puck.vy + accel;
                case 6
                    obj.puck.vx = obj.puck.vx - accel;
                    obj.puck.vy = obj.puck.vy - accel;
                case 7
                    obj.puck.vx = obj.puck.vx + accel;
                    obj.puck.vy = obj.puck.vy - accel;
                case 8
                    obj.puck.vx = obj.puck.vx - accel;
                    obj.puck.vy = obj.puck.vy + accel;
                otherwise
            end

            % Handle boundary conditions and bounce
			if obj.puck.x < obj.rad
				obj.puck.vx = obj.puck.vx * -0.5; % bounce!
				obj.puck.x = obj.rad;
			end

			if obj.puck.x > 1 - obj.rad
				obj.puck.vx = obj.puck.vx * -0.5;
				obj.puck.x = 1 - obj.rad;
			end

			if obj.puck.y < obj.rad
				obj.puck.vy = obj.puck.vy * -0.5; % bounce!
				obj.puck.y = obj.rad;
			end

			if obj.puck.y > 1 - obj.rad
				obj.puck.vy = obj.puck.vy * -0.5;
				obj.puck.y = 1 - obj.rad;
			end

			obj.t = obj.t + 1;

			if rem(obj.t, 100) == 0
				obj.green.x = rand();
				obj.green.y = rand();
			end

			% Compute distances
			greendist = norm([obj.green.x - obj.puck.x, obj.green.y - obj.puck.y]);
			red2puck = [obj.puck.x - obj.red.x, obj.puck.y - obj.red.y]; % Displacement
			red2puckdist = norm(red2puck); % Distance
			red2pucknorm = red2puck / red2puckdist; % Normalised dispacement vector

			% Move red to puck
			obj.red.x = obj.red.x + 0.001 * red2pucknorm(1);
			obj.red.y = obj.red.y + 0.001 * red2pucknorm(2);

			% Compute reward (one-liner! Don't burn me!)
			r = -greendist + double(red2puckdist < obj.red.badrad) * (2 * (red2puckdist - obj.red.badrad) / obj.red.badrad);
			s = obj.get_state();
		end

		function obj = start(obj)
			obj.render.f = figure();
		    while true
		    	% Get Keyboard input to change showspeed
                k = get(gcf, 'CurrentCharacter');
                if k~='@'
                    set(gcf, 'CurrentCharacter', '@');
                    if double(k) == 30
                        obj.showevery = obj.showevery * 2; 
                    elseif double(k) == 31
                        obj.showevery = max([obj.showevery/2, 1]);
                    elseif strcmp(k, 'f')
                        obj.showevery = 256;
                    elseif strcmp(k, 'q')
                        close(1);
                        break;
                    elseif strcmp(k, 'r')
                        obj.showevery = 2; 
                    else
                    end
                else
                end

		    	fprintf('step:%d\n', obj.t);
		    	s = obj.get_state();
		    	[~, a] = obj.agent.act(s);
		    	[s, r] = obj.sampleNextState(a);
		    	obj.agent.learn(r);
		    	obj.puck.action = a;
		    	obj.render.tderr = [obj.render.tderr, abs(obj.agent.tderr)];
		    	obj.render.tr = [obj.render.tr, r];

		    	if rem(obj.t, obj.showevery) == 0
		    		obj.draw();
		    	end
		    end
		end
	end

	methods (Access = private)
	    function obj = draw(obj)
	    	clf(obj.render.f)
	    	figure(obj.render.f)
	    	subplot(2, 2, 1);
	    	obj.drawEnv();

            % Draw Error 
            subplot(2, 2, 2)
            axis auto
            plot(obj.render.tderr(max([1; obj.t-1000]): end))
            title('TD Error')

            % % Draw Temporary Rewards
            % subplot(2, 2, 2)
            % axis auto
            % plot(obj.avgQ(1:obj.stepidx));
            % title('AVG Q')

            % Draw Accumulated Rewards
            subplot(2, 2, 3)
            axis auto
            plot(obj.render.tr(max([1; obj.t-1000]): end));
            title('Temporary Rewards')

            drawnow
	    end


	    function obj = drawEnv(obj)
            hold on
            axis equal 
            axis([0 1 0 1])
            axis manual
            rectangle('Position', [0 0 1 1]);

            % Draw red
            viscircles([obj.red.x, obj.red.y], 0.01, 'EdgeColor', 'black');
			red2puckdist = norm([obj.red.x - obj.puck.x, obj.red.y - obj.puck.y]); % Distance
            if red2puckdist < obj.red.badrad
                boundcolor = [ 1 - red2puckdist/ obj.red.badrad, red2puckdist/ obj.red.badrad, 0];
            else
                boundcolor = [0, 1, 0];
            end
            viscircles([obj.red.x, obj.red.y], obj.red.badrad, 'EdgeColor', boundcolor);

            % Draw Green
            viscircles([obj.green.x, obj.green.y], 0.01, 'EdgeColor', 'g');

            % Draw Puck
            viscircles([obj.puck.x, obj.puck.y], obj.rad, 'EdgeColor', 'b');
            switch obj.puck.action
                case 1
                    line([obj.puck.x, obj.puck.x], [obj.puck.y, obj.puck.y + 2 * obj.rad]);
                case 2
                    line([obj.puck.x, obj.puck.x], [obj.puck.y, obj.puck.y - 2 * obj.rad]);
                case 3
                    line([obj.puck.x, obj.puck.x - 2 * obj.rad], [obj.puck.y obj.puck.y]);
                case 4
                    line([obj.puck.x, obj.puck.x + 2 * obj.rad], [obj.puck.y, obj.puck.y]);
                case 5
                case 6
                case 7
                case 8
                otherwise
            end
            title(sprintf('Train Step:%d - Show Every:%d', obj.t, obj.showevery))
	    end
	end
end