# Rouge-L

To assess the performance of our fine-tuned model, we leverage the Rouge-L 
metric and conduct experiments with a large number of clients, utilizing the 
Dolly-15K dataset as our training corpus. The Dolly-15K dataset encompasses 
a total of 15,015 data points, distributed across eight distinct tasks. For 
a more comprehensive evaluation, we allocate the final task exclusively for 
evaluation purposes, while dedicating the remaining ones to the training 
phase. Our experimental setup involves a network of 200 clients, utilizing a Dirichlet distribution for data partitioning to emulate non-IID conditions across the client base.

To do the evaluation, run
```bash
python federatescope/eval/eval_for_rougel/eval.py --cfg 
federatescope/llm/baselime/xxx.yaml
```