#BSUB -W 4:00
#BSUB -o /cluster/work/igp_psr/nassar/pytorch-retina/output.txt
#BSUB -e /cluster/work/igp_psr/nassar/pytorch-retina/error.txt
#BSUB -n 1
#BSUB -R "rusage[mem=32000,ngpus_excl_p=4]"
#### BEGIN #####
module load python_gpu/3.7.4

module load hdf5/1.10.1
module load eth_proxy
module load gcc/6.3.0

python3 train.py --dataset Pasadena --dataset_root ../datasets/VOC/ --depth 101 --batch_size 5

#### END #####
