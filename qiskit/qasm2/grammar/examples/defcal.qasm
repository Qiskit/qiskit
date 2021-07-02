defcalgrammar "openpulse";

defcal x $0 {
   play drive($0), gaussian(...);
}

defcal x $1 {
  play drive($1), gaussian(...);
}

defcal rz(angle[20]:theta) $q {
  shift_phase drive($q), -theta;
}

defcal measure $0 -> bit {
  complex[int[24]] iq;
  bit state;
  complex[int[12]] k0[1024] = [i0 + q0*j, i1 + q1*j, i2 + q2*j, ...];
  play measure($0), flat_top_gaussian(...);
  iq = capture acquire($0), 2048, kernel(k0);
  return threshold(iq, 1234);
}

defcal zx90_ix $0, $1 {
  play drive($0, "cr1"), flat_top_gaussian(...);  // uses a non-default
                                                  // frame labeled "cr1"
}

defcal cx $0, $1 {
  zx90_ix $0, $1;
  x $0;
  shift_phase drive($0, "cr1");
  zx90_ix $0, $1;
  x $0;
  x $1;
}
