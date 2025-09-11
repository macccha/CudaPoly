jsonInclude := /data/others/ciarchi/PolymerDyn/CudaPoly
OptFlag := -O3
CudaCC := nvcc
objects := polydyn.o polydynOpt.o

$(objects): mainc.cu particleslist.h
	$(CudaCC) -I$(jsonInclude) -c mainc.cu -o $@ -O3

polydyn: polydyn.o
	$(CudaCC) -I$(jsonInclude) $< -o $@

polydynOpt: polydynOpt.o
	$(CudaCC) -I$(jsonInclude) $< -o $@ $(OptFlag)

clean:
	rm -f polydyn

cleanOpt:
	rm -f polydynOpt

cleanall:
	rm -f *.o polydyn polydynOpt