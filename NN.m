classdef NN < matlab.mixin.SetGet
    properties
        opt
        W = cell(0);
        b = cell(0);
        dW = cell(0);
        db = cell(0);
        a = cell(0);
        actfun = cell(0); % Function to activation
	end

	methods
		function obj = NN(opt)
			obj.opt = opt;
			obj.opt.nlayer = numel(opt.sz);
			npara = 0;

			for i = 1 : obj.opt.nlayer - 1
                npara = npara + opt.sz(i) * opt.sz(i + 1) + opt.sz(i + 1);
			end

            obj.opt.npara = npara;
            theta = normrnd(0, 0.01, npara, 1);
            gtheta = zeros(size(theta));
	        obj.W = cell(obj.opt.nlayer, 1);
	        obj.b = cell(obj.opt.nlayer, 1);
	        obj.dW = cell(obj.opt.nlayer, 1);
	        obj.db = cell(obj.opt.nlayer, 1);
	        obj.a = cell(obj.opt.nlayer, 1);

            obj.parafromvec(theta);
            obj.gradfromvec(gtheta);
            obj.actfun = opt.actfun; % Transfer activation functions from strings: tanh, sigmoid, linear, etc
		end


		function [obj, out] = forward(obj, x)
			ncase = size(x, 1);
			obj.a{1} = x;
			% Forward the network layer by layer; input considered as activation as well 
			for i = 1 : obj.opt.nlayer - 1
				obj.a{i + 1} = transpose(obj.actfun{i}.forward(obj.W{i} * obj.a{i}' + repmat(obj.b{i}, 1, ncase))); % hidden activation
			end

			out = obj.a{obj.opt.nlayer};
		end


		function obj = backward(obj, err)
			ncase = size(err, 1);
			e = cell(obj.opt.nlayer - 1, 1);
            e{obj.opt.nlayer} = - err .* obj.actfun{obj.opt.nlayer - 1}.backward(err);

            % Chain rule
            for i = obj.opt.nlayer - 1 : -1 : 2
	            e{i} = transpose((obj.W{i}' * e{i + 1}') .* obj.actfun{i-1}.backward(obj.a{i})');
	        end

	        for i = obj.opt.nlayer - 1 : -1 : 1
	            obj.dW{i} = obj.dW{i} + e{i+1}' * obj.a{i};
	            obj.db{i} = sum(repmat(obj.db{i}, 1, ncase) + e{i+1}', 2);
	        end
		end


		function obj = update(obj)
			theta = obj.para2vec();
			grad = obj.grad2vec();
			theta = theta + obj.opt.alpha * grad;
			obj.parafromvec(theta);
			grad(:) = 0;
			obj.gradfromvec(grad);
		end	


		function obj = checkgrad(obj)
            eps = 1e-4;
            % Generate random input and ground truth output
            x = rand(4, obj.opt.sz(1));
            y = rand(4, obj.opt.sz(obj.opt.nlayer));
            
            % Get the gradients from BP
            obj.forward(x);
            e = y - obj.a{obj.opt.nlayer};
            obj.backward(e);
            theta = obj.para2vec();
            gtheta = obj.grad2vec();

            for i = 1 : numel(theta)
                tpos = theta;
                tneg = theta;
                e = zeros(numel(theta), 1);
                e(i) = eps;
                tpos = tpos + e;
                tneg = tneg - e;
                obj.parafromvec(tpos);
                obj.forward(x);
                q = obj.a{obj.opt.nlayer};
                jpos = sum(1/2 *sum((y - q).^ 2));
                obj.parafromvec(tneg);
                obj.forward(x);
                q = obj.a{obj.opt.nlayer};
                jneg = sum(1/2 * sum((y - q).^ 2));
                gapx = (jpos - jneg) / (2 * eps); % Approximated gradient
                gerr = gtheta(i) - gapx; % Gradient Error 
                fprintf('Checking %dth theta - gtheta:%f - gapprox:%f - gerr: %f - eps:%f\n', i, gtheta(i), gapx, gerr, eps);
                assert(abs(gerr) < eps);
            end
        end		


        function obj = testall(obj)
        	t1 = rand(obj.opt.npara, 1);
			obj.parafromvec(t1);
			t2 = obj.para2vec();
			assert(all(t1 == t2));
			disp('=== Vectorisation passed ===')

			% Check BP gradients
			obj.checkgrad();
			disp('=== Gradient Check Passed ===')
        end
	end


	methods (Access = private)

		function obj = parafromvec(obj, theta)
		    for i = 1 : obj.opt.nlayer - 1
		    	if i == 1
		    		ts = 1;
		    	else
			    	ts = te + 1;
			    end

                te = ts + obj.opt.sz(i) * obj.opt.sz(i + 1) - 1;
		    	obj.W{i} = reshape(theta(ts:te), obj.opt.sz(i + 1), obj.opt.sz(i));
		    end

		    for i = 1 : obj.opt.nlayer - 1
		    	ts = te + 1;
                te = ts + obj.opt.sz(i + 1) - 1;
		    	obj.b{i} = reshape(theta(ts:te), obj.opt.sz(i+1), 1);
		    end
		end


		function g = para2vec(obj)
            g = obj.getvec(obj.W, obj.b);
		end


		function obj = gradfromvec(obj, theta)

		    for i = 1 : obj.opt.nlayer - 1
		    	if i == 1
		    		ts = 1;
		    	else
			    	ts = te + 1;
			    end

                te = ts + obj.opt.sz(i) * obj.opt.sz(i + 1) - 1;
		    	obj.dW{i} = reshape(theta(ts:te), obj.opt.sz(i + 1), obj.opt.sz(i));
		    end

		    for i = 1 : obj.opt.nlayer - 1
		    	ts = te + 1;
                te = ts + obj.opt.sz(i + 1) - 1;
		    	obj.db{i} = reshape(theta(ts:te), obj.opt.sz(i+1), 1);
		    end
		end

		function g = grad2vec(obj)
            g = obj.getvec(obj.dW, obj.db);
		end


		function t = getvec(obj, W, b)
			t = [];
            for i = 1 : numel(W) 
                t = [t;W{i}(:)];
            end

            for i = 1 : numel(b)
                t = [t;b{i}(:)];
            end
		end


	end
end