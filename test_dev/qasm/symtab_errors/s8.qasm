
gate toffoli a,b,c 
{ 
     // just for the test
}

gate majority a,b,c           // this one is ok
{ 
  toffoli () a,b,c; 
}

gate majorityx a,b,c           // this one ain't
{ 
  toffoli () a,b,d; 
}
