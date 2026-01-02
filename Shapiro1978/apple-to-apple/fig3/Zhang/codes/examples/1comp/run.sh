ntasks=($(awk "NR==2" model.in))
echo number of tasks=$ntasks
echo 'run ini'
mpirun -np $ntasks ./ini
echo 'nohup run main'
mpirun -np $ntasks ./main 


