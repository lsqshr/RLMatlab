class DQNAgent < matlab.mixin.SetGet
    properties
        % gamma = 0.75; % Future reward discount factor
        % episilon = 0.1; % epsilon-greedy exploration probability
        % alpha = 0.01; % Learning rate
        % experience_add_every = 25;
        % experience_size = 5000;
        % learning_steps_per_iteration = 10;
        % tderror_clamp = 1.0;
        % num_hidden_units = 100;
        % env = env;
        p % p.* contains the hyperparameters above
        exppool % Experience pool
        expi % Experience pointer
        t % Timer
        e % The current experience

       
        % For network
        nn % its network
	end

	methods
	    function obj = DQNAgent(p)
	    	obj.p = p;
	    	obj.nn = VanillaNeualNet(p.na, p.nh, p.na); % Hard coded single layered NN
	    end

	    function obj = reset(obj)
            % obj.nn.nh = obj.p.nh; % #hidden units
            % obj.nn.ns = obj.p.ns; % #states
            % obj.nn.na = obj.p.na; % #actions
            obj.nn.alpha = obj.p.alpha;
	    	obj.nn = VanillaNeualNet(obj.p.ns, obj.p.nh, obj.p.na); % Hard coded single layered NN

	    	obj.exppool = cell(obj.p.experience_size, 1);
	    	obj.expi = 1;
	    	obj.t = 1;

            obj.e.r0 = null;
            obj.e.s0 = null;
            obj.e.s1 = null;
            obj.e.a0 = null;
            obj.e.a1 = null;
            obj.tderr = 0; % For visualisation only
	    end
        
	    function obj = save(obj)
	    	% TODO
	    end

	    function obj = load(obj)
	    	% TODO
	    end

	    function [obj, out] = forwardQ(obj, s, needs_backprop) % Might not use it
	    	out = obj.nn.forward(s, needs_backprop);
	    end

	    function [obj, a] = act(obj, s)
            if (rand() < obj.eps)
            	a = randi(obj.nn.na); % This might be the only difference I used to have between the Karpathy js version
            else
            	out = obj.nn.forward(s, false);
            	[~, a] = max(out(:));
            end

            % Shift state memory
            obj.e.s0 = obj.e.s1;
            obj.e.a0 = obj.e.a1;
            obj.s1 = s;
            obj.a1 = a;

            return a;
	    end

	    function obj = learn(obj, r1)
	    	if (obj.e.r0 ~= null) && obj.nn.alpha > 0
	    		% Learn from this tuple to get a sense of how "surprising" it is to the agent
	    		obj.tderr = obj.learnFromTuple(obj.e);

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
	    	out = obj.nn.forward(e.s1, false);
	    	tdtarget = e.r0 + obj.p.gamma * max(out(:));

	    	% Predict
	    	out = obj.nn.forward(e.s0, true)
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
	    	fullerr = zeros(obj.p.na, 1);
	    	fullerr(e.a0) = tderr;
	    	obj.nn.backward(tderr);

	    	% Update Net
	    	obj.nn.update();
	    end

	end
end