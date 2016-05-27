function s = lin()
	s.forward = inline('z');
	s.backward = inline('ones(size(z))');
end