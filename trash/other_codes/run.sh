
#part=dpt-cpu
part=shared-cpu


for i in 0 1; do
for j in 1st 2nd 3rd; do
#for k in {0..8}; do
for k in 4 6 ; do

vsub -c "python try1.py $i $j $k" --part $part --name $i-$j-$k -N 4 --mem 40000 --time 11:00:00

done
done
done

for j in 1st 2nd 3rd; do
#for k in {0..8}; do
for k in 4 6 7; do

vsub -c "python try2.py $j $k" --part $part --name $j-$k -N 4 --mem 40000 --time 11:00:00

done
done


#for i in 0 1; do
#for j in 1st 2nd 3rd; do
#for k in {1..2}; do

#vsub -c "python try0.py $i $j $k" --part $part --name $i-$j-$k -N 2 --mem 20000 --time 11:00:00

#done
#done
#done


#for j in 1st 2nd 3rd; do
#vsub -c "python try0.py 1 $j 7" --part $part --name 1-$j-7 -N 8 --mem 20000 --time 11:00:00
#done

#for i in 1 ;do
#vsub -c "python run.py $i" --part $part --name 21cm-$i -N 4 --mem 40000 --time 24:00:00
#done


