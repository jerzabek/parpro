%: %.cu
	nvcc -o $@ $<
	rm -f $@.o $@.lib $@.exp


clean:
	rm -f *.exe *.o *.lib *.exp