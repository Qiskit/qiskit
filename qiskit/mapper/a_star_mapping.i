/* a_star_mapping.i */
%module a_star_mapping
%include "std_string.i"
%include "std_vector.i"

%{ 
 /* Put header files here or function declarations like below */

#include "a_star_mapping.h"

using namespace std;

extern mapping_result map(std::string qasm, std::vector<edge> coupling, int nqubits);

%}

namespace std {
  %template(vector_edge) vector<edge>;
};

typedef struct edge {
	int v1;
	int v2;
} edge;

typedef struct mapping_result
{
	std::string qasm;
	std::vector<edge> initial_layout;
} mapping_result;

extern mapping_result map(std::string qasm, std::vector<edge> coupling, int nqubits);

