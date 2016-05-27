opt.sz = [2 3 2];
opt.act = {sigmoid(), lin()}; 
opt.alpha = 1e-2;
nn = NN(opt);
nn.testall();

