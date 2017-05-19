gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
qreg r[2];

gate foo q {
    u3(1, 2, 3, 4) q;
}
