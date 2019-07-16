#!/bin/bash
for f in `ls`
do
cat $f | perl -pe 's/assert_product/assert_product/g' > 'temp1.py';
cat temp1.py | perl -pe 's/assert_classical/assert_classical/g' > 'temp2.py';
cat temp2.py | perl -pe 's/assert_uniform/assert_uniform/g' > 'temp3.py';
cat temp3.py | perl -pe 's/uniform/uniform/g' > 'temp4.py';
cat temp4.py | perl -pe 's/Uniform/Uniform/g' > 'temp5.py';
mv temp5.py $f
rm temp1.py
rm temp2.py
rm temp3.py
rm temp4.py
done


