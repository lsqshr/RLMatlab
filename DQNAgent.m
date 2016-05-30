classdef DQNAgent < matlab.mixin.SetGet
    properties
        p % p.* contains the hyperparameters above
        exppool % Experience pool
        expi % Experience pointer
        t % Timer
        e % The current experience
        nn % its network
        tderr
        eps = 0.2
	end

	methods (Access = public)

	    function obj = DQNAgent(p)
	    	obj.p = p;
	    	obj.reset();
	    end


	    function obj = reset(obj)
	    	opt.sz = obj.p.nnsz;
	    	opt.actfun = {tanh_mrl(), lin_mrl()};
	    	opt.alpha = obj.p.alpha;
	    	obj.nn = NN(opt);
	    	obj.exppool = cell(obj.p.experience_size, 1);
	    	obj.expi = 0;
	    	obj.t = 1;

            obj.e.r0 = NaN;
            obj.e.s0 = NaN;
            obj.e.s1 = NaN;
            obj.e.a0 = NaN;
            obj.e.a1 = NaN;
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
            	a = randi(obj.p.env.get_num_actions()); % This might be the only difference I used to have between the Karpathy RL.JS version
            else
            	[~, out] = obj.nn.forward(s);
            	[~, a] = max(out(:));
            end

            % Shift state memory
            obj.e.s0 = obj.e.s1;
            obj.e.a0 = obj.e.a1;
            obj.e.s1 = s;
            obj.e.a1 = a;
	    end

	    function obj = learn(obj, r1)

	    	if ~isnan(obj.e.r0) && obj.nn.opt.alpha > 0
	    		% Learn from this tuple to get a sense of how "surprising" it is to the agent
	    		[obj, tderr] = obj.learn_from_tuple(obj.e);
	    		obj.tderr = tderr;

	    		% Decide whether to keep this experience in the replay
	    		if rem(obj.t, obj.p.experience_add_every) == 0
	    			obj.expi = obj.expi + 1;
	    			obj.exppool{obj.expi} = obj.e;
	    			if obj.expi > obj.p.experience_size
	    				obj.expi = 0;
	    			end
	    		end

	    		obj.t = obj.t + 1; % Increment timer

	    		% Sample some additional experience from replay memory and learn from it
	    		if obj.expi > 1
		    		for k = 1 : obj.p.learning_steps_per_iteration

		    			ri = randi(min([obj.expi, numel(obj.exppool)]));
		    		    obj.learn_from_tuple(obj.exppool{ri});
		    		end
		    	end
	    	end
	    	obj.e.r0 = r1;
	    end

	    function [obj, tderr] = learn_from_tuple(obj, e)
	    	% Want: Q(s,a) = r + gamma * max_a' Q(s',a')

	    	% Compute the target Q Value
	    	[~, out] = obj.nn.forward(e.s1);
	    	tdtarget = e.r0 + obj.p.gamma * max(out(:));

	    	% Predict
	    	[~, out] = obj.nn.forward(e.s0);
	    	tderr = out(e.a0) - tdtarget;
	    	if abs(tderr) > obj.p.clamp
	    		if tderr > obj.p.clamp
                    tderr = obj.p.clamp;
	    		end

	    		if tderr < -obj.p.clamp
	    			tderr = - obj.p.clamp;
	    		end
	    	end

	    	% nn.dW{2}(e.a0) = tderr;
	    	fullerr = zeros(1, obj.p.env.get_num_actions());
	    	fullerr(e.a0) = tderr;
	    	obj.nn.backward(fullerr);

	    	% Update Net
	    	obj.nn.update();
	    end
	end
end