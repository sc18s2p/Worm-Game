#!/bin/bash
# Execute this file to recompile locally
/home/stefanos/anaconda3/envs/simple-worm/bin/x86_64-conda-linux-gnu-c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/home/stefanos/anaconda3/envs/simple-worm/include -I/home/stefanos/anaconda3/envs/simple-worm/include/eigen3 -I/home/stefanos/anaconda3/envs/simple-worm/.cache/dijitso/include dolfin_expression_3203b9db9fa2db182b752e7ffc28c01d.cpp -L/home/stefanos/anaconda3/envs/simple-worm/lib -L/home/stefanos/anaconda3/envs/simple-worm/home/stefanos/anaconda3/envs/simple-worm/lib -L/home/stefanos/anaconda3/envs/simple-worm/.cache/dijitso/lib -Wl,-rpath,/home/stefanos/anaconda3/envs/simple-worm/.cache/dijitso/lib -lmpi -lmpicxx -lpetsc -lslepc -lhdf5 -lboost_timer -ldolfin -olibdijitso-dolfin_expression_3203b9db9fa2db182b752e7ffc28c01d.so