# 1. Validate installation process:
#
# $ make
#
# 2. Install qiskit-c-api:
#
# $ sudo make install

OS = $(shell uname -s)
ifeq ($(OS), Linux)
	INCLUDE_DIR = /usr/include
	LIB_DIR = /usr/lib
endif
ifeq ($(OS), Darwin)
	INCLUDE_DIR = /usr/local/include
	LIB_DIR = /usr/local/lib
endif

check:
	@if [ ! -d "$(INCLUDE_DIR)" ]; then \
		echo "INCLUDE_DIR=$(INCLUDE_DIR) does not exist, cannot install qiskit-c-api. "; \
		echo "To use a custom include directory, run `make install INCLUDE_DIR=\"</path/to/include/dir>\"`" \
		echo "Exiting..."; \
		exit 1; \
	fi
	@if [ ! -d "$(LIB_DIR)" ]; then \
		echo "LIB_DIR=$(LIB_DIR) does not exist, cannot install qiskit-c-api. Exiting..."; \
		echo "To use a custom lib directory, run `make install LIB_DIR=\"</path/to/lib/dir>\"`" \
		exit 1; \
	fi
install: check
	cp dist/c/include/qiskit.h $(INCLUDE_DIR)
	cp -r dist/c/include/qiskit $(INCLUDE_DIR)
	cp dist/c/lib/libqiskit.* $(LIB_DIR)
uninstall:
	rm -f $(INCLUDE_DIR)/qiskit.h
	rm -rf $(INCLUDE_DIR)/qiskit
	rm -f $(LIB_DIR)/libqiskit.so
