function s = sigmoid()
	s.forward = inline('1 ./ (1 + exp(-z))');
	s.backward = inline('z .* (1 - z)');
end