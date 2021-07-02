OPENQASM 3;
include "stdgates.inc";

gate pre q { }   // pre-rotation
gate post q { }  // post-rotation

qubit q;
bit c;
reset q;
pre q;
barrier q;
h q;
barrier q;
post q;
c = measure q;
