#!/usr/bin/bash
#SBATCH --job-name=adthsji
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --output job-revision-adt-bb-HSJA_li-%j.out
#SBATCH --error job-revision-adt-bb-HSJA_li-%j.err
#SBATCH -p v100-16gb-hiprio
#SBATCH --gres=gpu:2

# load modules
module load cuda/11.1
module load python3/anaconda/ai-lab


# setup environment
echo '>>> Preparing environment'
# source activate <your_environment>

echo '>>> Generating AEs (HSJA linf), it may take a while...'
#python eval_cifar.py
#eval_whitebox_cifar100.py -c confs/wresnet28x10_cifar10_b128.yaml --aug fa_reduced_cifar10 --dataroot=data --dataset cifar100 --save cifar100_wres28x10.pth --only-eval
#export CUDA_VISIBLE_DEVICES=0

ATTACK_CONFIG="../configs/experiment/cifar100/attack-untargeted-hsja.json"
ATTACK="li_norm"
EOT="False"
echo $ATTACK_CONFIG
echo $ATTACK $EOT
#declare -a TARGET_MODELS=("revisionDivHigh-s4h0" "revisionDivHigh-s4h1" "revisionDivHigh-s4h2" "revisionDivHigh-s4h3"
#                          "revisionDivHigh-s4h4" "revisionDivHigh-s4h5" "revisionDivHigh-s4h6" "revisionDivHigh-s4h7"
#                          "revisionDivHigh-s4h8" "revisionDivHigh-s4h9" "revisionDivLow-s4l0" "revisionDivLow-s4l1"
#                          "revisionDivLow-s4l2" "revisionDivLow-s4l3" "revisionDivLow-s4l4" "revisionDivLow-s4l5"
#                          "revisionDivLow-s4l6" "revisionDivLow-s4l7" "revisionDivLow-s4l8" "revisionDivLow-s4l9"
#                          )

declare -a TARGET_MODELS=("adt_pgd")

for t in "${TARGET_MODELS[@]}"
do
  bash gen-ae-cifar-bb_hsja.sh $ATTACK_CONFIG $ATTACK $EOT $t
  echo "-------"
done

echo '>>> DONE! <<<'

source deactivate
