#!/usr/bin/bash
#SBATCH --job-name=pad1
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --output job-revision-ens-PNE1-%j.out
#SBATCH --error job-revision-ens-PNE1-%j.err
#SBATCH -p v100-16gb-hiprio
#SBATCH --gres=gpu:2

# load modules
module load cuda/11.1
module load python3/anaconda/ai-lab

# setup environment
echo '>>> Preparing environment'
# source activate <your_environment>

echo '>>> Generating AEs (genAE-cifar100-wb-pgd-nonEOT.sh), it may take a while...'
#python eval_cifar.py
#eval_whitebox_cifar100.py -c confs/wresnet28x10_cifar10_b128.yaml --aug fa_reduced_cifar10 --dataroot=data --dataset cifar100 --save cifar100_wres28x10.pth --only-eval
#export CUDA_VISIBLE_DEVICES=0

ATTACK_CONFIG="../configs/experiment/cifar100/revision-attack-pgd-nonEot.json"
ATTACK="group3"
EOT="False"
echo $ATTACK_CONFIG
echo $ATTACK $EOT

declare -a TARGET_MODELS=("revisionES1-ens1"
                          "revisionES4-ens2" "revisionES8-ens2" "revisionES16-ens2"
                          )

#declare -a TARGET_MODELS=("adt_pgd")

for t in "${TARGET_MODELS[@]}"
do
  bash genAE-cifar100-wb-pgd-nonEOT.sh $ATTACK_CONFIG $ATTACK $EOT $t
  echo "----------------------"
done

echo '>>> DONE! <<<'

source deactivate
