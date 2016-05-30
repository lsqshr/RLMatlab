clear all;

addpath('utils')
addpath('tests')

pw = PuckWorld();
pw.reset();
pw.start();