gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
gate cx c,t { CX c,t; }

// testing deeply nexted expression with junk in it
gate cu3(theta,phi,lambda) c, t 
{
  u1((lambda-phi)/2) t;
  cx c,t;
  u3(-theta/2,0,-((3*(phi+1+d))+lambda)/2) t;
  cx c,t;
  u3(theta/2,phi,0) t;
}
