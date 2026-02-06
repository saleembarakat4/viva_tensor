@echo off
set PATH=C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin;%PATH%
set MKL_NUM_THREADS=8
cd /d C:\Users\gabrielmaia\viva_gleam\repos\viva_tensor
erl -noshell -pa priv -s bench_mkl run -s init stop
