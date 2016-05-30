function s = lin_mrl()
	s.forward = inline('z');
	s.backward = inline('ones(size(z))');
end