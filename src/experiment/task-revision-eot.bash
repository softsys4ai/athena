#!/usr/bin/bash
#SBATCH --job-name=PR1
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --output job-revision-wb-adt_pgd-PERotation1-%j.out
#SBATCH --error job-revision-wb-adt_pgd-PERotation1-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:2

# load modules
module load cuda/11.1
module load python3/anaconda/ai-lab

# setup environment
echo '>>> Preparing environment'
#conda create -n ym_adt python=3.7
source activate ym_adt

#conda create -n athena python=3.7
#source activate athena

#conda install pytorch torchvision -c pytorch
#pip install -r requirements.txt
#pip install --user scipy
##conda install tensorflow
#conda install opencv
#pip install git+https://github.com/wbaek/theconf.git
#pip install scikit-image
#pip install scikit-learn
#pip install Keras
#pip install adversarial-robustness-toolbox
#pip install tqdm
#pip install pytorch-warmup

echo '>>> Generating AEs (genAE-cifar100-wb-pgd-EOT_rotation.sh), it may take a while...'
ATTACK_CONFIG="../configs/experiment/cifar100/revision-attack-pgd-eot.json"
EOT="True"
ATTACK="div_rot"
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
  bash genAE-cifar100-wb-pgd-EOT_rotation.sh $ATTACK_CONFIG $ATTACK $EOT $t
  echo "-------"
done

echo '>>> DONE! <<<'

source deactivate
