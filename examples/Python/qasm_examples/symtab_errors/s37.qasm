gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
qreg r[2];

u3() r;
