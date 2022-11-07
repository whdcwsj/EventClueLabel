home_dir="/home/yxb/MyCoding/NTMModelCode/GraphNeuralTopicModel"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
seeds=(
1
10
999
1423
2542
)
# --use_pretrained_embedding False
CLIP=0.0
EPOCH=1500
if [ $1 == 50 ]; then
  CLIP=1.0
  EPOCH=1500
fi
for seed in ${seeds[*]}
do
  echo $seed
  echo $CLIP
  python run_graph_ntm.py --data_dir input/yahoo_answers/  --output_dir output/yahoo_answers/ \
    --topwords 10 --topic_num $1 --average_kind bow --seed $seed \
    --clip $CLIP --epoch $EPOCH
done