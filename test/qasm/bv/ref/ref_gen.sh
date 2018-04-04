for num in `seq 10 19`
do
  filenum=`expr $num + 1`
  filename=bv_n$filenum.qasm.ref 
  echo -n "{\"" > $filename
  for ((i  = 0; i < $num; ++i))
  do
    echo -n "1" >> $filename
    done
  echo "\": 1}" >> $filename
done
