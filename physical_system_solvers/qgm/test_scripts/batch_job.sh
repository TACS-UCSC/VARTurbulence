#!/bin/bash
#rm *.err
#rm *.out
#rm *.log

for SEED in {1..2} ;do
	export SEED
                    
	echo $SEED
                
	sbatch --export=all job_submit_3d.sh $SEED

done
