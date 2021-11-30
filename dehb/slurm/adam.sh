#!/bin/sh
#SBATCH -p mlhiwidlc_gpu-rtx2080 #partition
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-24:00:00 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-10 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /work/dlclarge1/sharmaa-dehb # Change working_dir
#SBATCH -o /work/dlclarge1/sharmaa-dehb/logs_dehb/dehb_exp.%A.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge1/sharmaa-dehb/logs_dehb/dehb_exp.%A.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sharmaa@informatik.uni-freiburg.de
# Activate virtual env so that run_experiment can load the correct packages

cd $(ws_find dehb)
cd DEHB

#pip install -r requirements.txt

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 1
			                                    exit $?
fi
if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 2
				                                                               exit $?
fi
if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 3
					                                                                                                  exit $?
fi
if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 4
						                                                                                                                                              exit $?
fi
if [ 5	-eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 5
							                                                                                                                                                                                                  exit $?
fi
if [ 6  -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 6
								                                                                                                                                                                                                                                                              exit $?
fi
if [ 7  -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 7

									                                                                                                                                                                                                                  exit $?
fi
if [ 8  -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 8

									                                                                                                                                                                                                                  exit $?
fi
if [ 9  -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 9

									                                                                                                                                                                                                                  exit $?
fi
if [ 10  -eq $SLURM_ARRAY_TASK_ID ]; then
	                   python3 -m dehb.examples.toy_examples.adam_env --loss 'branin' --run_id 10

									                                                                                                                                                                                                                  exit $?
fi
