jsonInclude := /data/others/ciarchi/PolymerDyn/CudaPoly
OptFlag := -O3
CudaCC := nvcc
objects := polydyn.o polydynOpt.o

$(objects): mainc.cu particleslist.h
	$(CudaCC) -I$(jsonInclude) -c mainc.cu -o $@

polydyn: polydyn.o
	$(CudaCC) $< -o $@

polydynOpt: polydynOpt.o
	$(CudaCC) $< -o $@ $(OptFlag)

polydynTest: polydynTest.o
	$(CudaCC) $< -o $@ $(OptFlag)

clean:
	rm -f polydyn

cleanOpt:
	rm -f polydynOpt

cleanall:
	rm -f *.o polydyn polydynOpt