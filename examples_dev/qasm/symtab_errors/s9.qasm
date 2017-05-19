
gate toffoli a,b,c 
{ 
     // just for the test
}

gate majority a,b,c           // this one is ok
{ 
  toffoli (123) a,b,c; 
}

gate majorityx a,b,c           // this one ain't
{ 
  toffoli (123) a,b,d; 
}
