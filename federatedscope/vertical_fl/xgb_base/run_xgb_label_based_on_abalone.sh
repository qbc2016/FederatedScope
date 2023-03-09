set -e


seed=(1 2 3 4 5)

echo "start..."

for (( i=0; i<${#seed[@]}; i++))
do
  nohup python federatedscope/main.py --cfg federatedscope/vertical_fl/xgb_base/baseline/xgb_feature_order_dp_on_abalone.yaml seed ${seed[$i]} outdir exp_out/abalone_${seed[$i]}.txt &
done