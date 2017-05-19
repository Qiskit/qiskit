gate ccx a,b,c 
{ 
  h c; 
  cx b,c; tdg c; 
  cx a,c; t c; 
  cx b,c; tdg c; 
  cx a,c; t b; t c; h c; 
  cx a,b; t a; tdg b; 
  cx a,b;
}

