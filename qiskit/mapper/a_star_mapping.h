#include <iostream>
#include <queue>
#include <algorithm>
#include <string.h>
#include <set>
#include <climits>
#include <fstream>

struct edge {
	int v1;
	int v2;
};

struct gate {
	int target;
	int control;
	char type[128];
};

inline bool operator<(const edge& lhs, const edge& rhs) {
	if (lhs.v1 != rhs.v1) {
		return lhs.v1 < rhs.v1;
	}
	return lhs.v2 < rhs.v2;
}

struct node {
	int cost_fixed;
	int cost_heur;
	int cost_heur2;
	int depth;
	int* qubits; // get qubit of location -> -1 indicates that there is "no" qubit at a certain location
	int* locations; // get location of qubits -> -1 indicates that a qubit does not have a location -> shall only occur for i > nqubits
	int nswaps;
	int done;
	std::vector<std::vector<edge> > swaps;
};

struct node_cmp {
	bool operator()(node& x, node& y) const {
		if ((x.cost_fixed + x.cost_heur + x.cost_heur2) != (y.cost_fixed + y.cost_heur + y.cost_heur2)) {
			return (x.cost_fixed + x.cost_heur + x.cost_heur2) > (y.cost_fixed + y.cost_heur + y.cost_heur2);
		}

		if(x.done == 1) {
			return false;
		}
		if(y.done == 1) {
			return true;
		}

		return x.cost_heur + x.cost_heur2 > y.cost_heur + y.cost_heur2;
	}
};

typedef struct mapping_result
{
	std::string qasm;
	std::vector<edge> initial_layout;
} mapping_result;

mapping_result map(std::string qasm, std::vector<edge> coupling, int nqubits);
