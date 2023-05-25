set -e

#cd ..

echo "Test distributed mode with LLM..."

# server
nohup python federatedscope/main.py --cfg federatedscope/llm/distribute/distribute_server.yaml seed 1 outdir zfed_0.001/server.txt&
sleep 2

# clients
nohup python federatedscope/main.py --cfg federatedscope/llm/distribute/distribute_client_one.yaml seed 1 outdir zfed_0.001/c1.txt &
sleep 2
nohup python federatedscope/main.py --cfg federatedscope/llm/distribute/distribute_client_two.yaml seed 1 outdir zfed_0.001/c2.txt &
sleep 2
nohup python federatedscope/main.py --cfg federatedscope/llm/distribute/distribute_client_three.yaml seed 1 outdir zfed_0.001/c3.txt &
