salloc --nodes=1 --exclusive --ntasks-per-node=1 --cpus-per-task=64 --mem=500G --gres=gpu:8
srun --nodes=1 --exclusive --ntasks-per-node=1 --cpus-per-task=64 --mem=500G --gres=gpu:8 --pty bash