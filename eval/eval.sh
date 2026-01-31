DEVICE=0 #,5,6,7

echo $(date +%Y-%m-%d_%H-%M-%S)

export CUDA_VISIBLE_DEVICES=$DEVICE

# python temp.py

# python ./eval/evaluate_ceval.py -d /hdd2/giri/datasets/ceval-exam --checkpoint-path /hdd2/giri/Qwen1.5-MoE-A2.7B
# python ./eval/evaluate_mmlu.py -d /hdd2/giri/datasets/mmlu/data/ --checkpoint-path /hdd2/giri/Qwen1.5-MoE-A2.7B
# python ./eval/evaluate_cmmlu.py -d /hdd2/giri/datasets/cmmlu/ --checkpoint-path /hdd2/giri/Qwen1.5-MoE-A2.7B

# python ./eval/evaluate_gsm8k.py  --checkpoint-path /hdd2/giri/Qwen1.5-MoE-A2.7B

# python ./eval/evaluate_humaneval.py  --checkpoint-path /hdd2/giri/Qwen1.5-MoE-A2.7B -f /hdd2/giri/repos/Qwen/eval/HumanEval.jsonl

python ./eval/evaluate_opus.py  --checkpoint-path /hdd2/giri/Qwen1.5-MoE-A2.7B  -d /hdd2/giri/datasets/opus/ --batch-size 32
