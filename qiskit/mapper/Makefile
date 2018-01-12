CFLAGS = -std=c++11 -O3 -fPIC -c -Wl,--no-undefined

mapper: a_star_mapping.cpp a_star_mapping.i
	swig -c++ -python a_star_mapping.i
	g++ $(CFLAGS) a_star_mapping.cpp -I.
	g++ $(CFLAGS) a_star_mapping_wrap.cxx -I$(CONDA_PREFIX)/include/python3.6m
	g++ -std=c++11 -O3 -shared a_star_mapping.o a_star_mapping_wrap.o -o _a_star_mapping.so
	rm a_star_mapping.o a_star_mapping_wrap.o a_star_mapping_wrap.cxx
