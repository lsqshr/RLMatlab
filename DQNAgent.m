classdef DQNAgent < matlab.mixin.SetGet
    properties
        p % p.* contains the hyperparameters above
        exppool % Experience pool
        expi % Experience pointer
        t % Timer
        e % The current experience
        nn % its network
        tderr
        eps
	end

	methods
	    function obj = DQNAgent(p)
	    	obj.p = p;
	    	obj.reset();
	    end


	    function obj = reset(obj)
	    	opt.sz = obj.p.nnsz;
	    	opt.actfun = {sigmoid(), lin()};
	    	obj.nn = NN(opt);
	    	obj.exppool = cell(obj.p.experience_size, 1);
	    	obj.expi = 1;
	    	obj.t = 1;

            % obj.e.r0 = ;
            % obj.e.s0 = -1;
            % obj.e.s1 = -1;
            % obj.e.a0 = -1;
            % obj.e.a1 = -1;
            obj.tderr = 0; % For visualisation only
	    end
        

	    function obj = save(obj)
	    	% TODO
	    end


	    function obj = load(obj)
	    	% TODO
	    end


	    function [obj, a] = act(obj, s)
            if (rand() < obj.eps)
            	a = randi(obj.nn.na); % This might be the only difference I used to have between the Karpathy js version
            else
            	[~, out] = obj.nn.forward(s);
            	[~, a] = max(out(:));
            end

            % Shift state memory
            obj.e.s0 = obj.e.s1;
            obj.e.a0 = obj.e.a1;
            obj.s1 = s;
            obj.a1 = a;
	    end

	    function obj = learn(obj, r1)
	    	if (obj.e.r0 ~= null) && obj.nn.alpha > 0
	    		% Learn from this tuple to get a sense of how "surprising" it is to the agent
	    		[obj, tderr] = obj.learnFromTuple(obj.e);
	    		obj.ederr = tderr;

	    		% Decide whether to keep this experience in the replay
	    		if rem(obj.t, obj.p.experience_add_every) == 0
	    			obj.exppool{obj.expi} = obj.e;
	    			obj.expi = obj.expi + 1;
	    			if obj.expi > obj.experience_size
	    				obj.expi = 0;
	    			end
	    		end

	    		obj.t = obj.t + 1; % Increment timer

	    		% Sample some additional experience from replay memory and learn from it
	    		for k = 1 : obj.learning_steps_per_iteration
	    			ri = randi(min([obj.expi, numel(obj.exppool)]));
	    		    obj.learn_from_tuple(obj.exppool{ri});
	    		end
	    	end
	    	obj.e.r0 = r1;
	    end

	    function [obj, tderr] = learnFromTuple(obj, e)
	    	% Want: Q(s,a) = r + gamma * max_a' Q(s',a')

	    	% Compute the target Q Value
	    	out = obj.nn.forward(e.s1);
	    	tdtarget = e.r0 + obj.p.gamma * max(out(:));

	    	% Predict
	    	out = obj.nn.forward(e.s0)
	    	tderr = out(e.a0) - tdtarget;
	    	if abs(tderr) > obj.p.clamp
	    		if tderr > obj.p.clamp
                    tderr = clamp;
	    		end

	    		if tderr < -obj.p.clamp
	    			tderr = -clamp;
	    		end
	    	end

	    	% nn.dW{2}(e.a0) = tderr;
	    	fullerr = zeros(obj.env.get_num_actions(), 1);
	    	fullerr(e.a0) = tderr;
	    	obj.nn.backward(tderr);

	    	% Update Net
	    	obj.nn.update();
	    end
	end
end