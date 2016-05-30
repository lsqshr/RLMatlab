function s = tanh_mrl()
	s.forward = inline('2 ./ (1 + exp(-2*z)) - 1');
	s.backward = inline('1 - z .^ 2');
end