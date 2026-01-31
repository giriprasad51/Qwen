import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu, CHRF
from tqdm import tqdm
import random
import numpy as np

# -----------------------------
# UTILITIES
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device).eval()
    return tokenizer, model

def translate_batch(model, tokenizer, texts, device, batch_size=32, max_length=256):
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            preds = model.generate(**inputs, max_length=max_length)
        decoded = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
        outputs.extend(decoded)
    return outputs

def compute_metrics(hypotheses, references):
    bleu = corpus_bleu(hypotheses, [references]).score
    chrf = CHRF().corpus_score(hypotheses, [references]).score
    return {"BLEU": bleu, "chrF": chrf}

def benchmark_opus(
    checkpoint_path,
    eval_data_path,
    device="cuda",
    batch_size=32,
    max_length=256,
    limit=None,
    debug=False,
):
    tokenizer, model = load_model(checkpoint_path, device)

    # Expect OPUS dataset files: test.{src}, test.{tgt}
    src_file = os.path.join(eval_data_path, "test.de")
    tgt_file = os.path.join(eval_data_path, "test.en")

    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_lines = f.readlines()

    if limit:
        src_lines = src_lines[:limit]
        tgt_lines = tgt_lines[:limit]

    if debug:
        print(f"Loaded {len(src_lines)} examples from {eval_data_path}")

    hypotheses = translate_batch(model, tokenizer, src_lines, device, batch_size, max_length)

    metrics = compute_metrics(hypotheses, tgt_lines)

    print(f"\nOPUS Evaluation Metrics for {eval_data_path}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

# -----------------------------
# MAIN
# -----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    domains = ["it", "koran", "law", "medical", "subtitles"]

    for domain in domains:
        opus_path = f"{args.eval_data_path}/{domain}"
        print(f"\nEvaluating domain: {domain}")
        benchmark_opus(
            checkpoint_path=args.checkpoint_path,
            eval_data_path=opus_path,
            device=device,
            batch_size=args.batch_size,
        max_length=args.max_seq_len,
        debug=args.debug,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OPUS dataset with Qwen model")
    parser.add_argument(
        "-c", "--checkpoint-path", type=str, default="Qwen/Qwen-7B", help="Model checkpoint"
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, required=True, help="Path to eval data")
    group.add_argument("--max-seq-len", type=int, default=2048, help="Maximum generated tokens")
    group.add_argument("--debug", action="store_true", default=False, help="Print debug info")
    group.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")

    args = parser.parse_args()
    main(args)
