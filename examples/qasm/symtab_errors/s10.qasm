
gate boo a,b,c 
{ 
    barrier a;           // ok
    barrier a, b, c;     // ok
    barrier c, d, e;     // not ok
}
