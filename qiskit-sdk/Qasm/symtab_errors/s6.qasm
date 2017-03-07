
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; } // ok
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) r; } // notok
