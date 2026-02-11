#!/bin/bash
#rm *.err
#rm *.out
#rm *.log

for SEED in {1..1} ;do
	export SEED
                    
	echo $SEED
                
	# sbatch --export=all job_submit_3d.sh $SEED
	qsub -v SEED=$SEED ../qsub_qgm_derecho1.sh
done
