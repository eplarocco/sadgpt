"""
toy_emotion_agents.py

Hackathon toy PoC: does few-shot steering toward sadness / neuroticism make a
3-agent Llama-3.1-8B system flag broken (unsolvable) problems more often,
without more false flags on solvable ones, compared with a neutral control?

System:
    Worker A, Worker B  -> discuss / attempt the problem
    Supervisor          -> has two tools: report_problem(reason), submit_answer(answer)
All three agents get the SAME emotion few-shot block in their system prompt.

Prompts (PsySET, github.com/aminbana/PsySET):
    neutral     -> emotions_fewshot_you.csv, emotion == "neutral"   (10 lines)
    sadness     -> emotions_fewshot_you.csv, emotion == "sadness"   (10 lines)
    neuroticism -> attributes.json, big_five_shots.neuroticism.pos  (10 lines)

Data (paired, same underlying problems), via AbstentionBench's own loader
(github.com/facebookresearch/AbstentionBench, recipe/abstention_datasets/gsm8k.py):
    gsm8k       -> GSM8K test question, full context      (should_abstain=False)
    abstention  -> same question with the context removed  (should_abstain=True)

Outcome per conversation:
    flagged = True   only if the supervisor calls report_problem
    flagged = False  if it submits an answer, OR runs out of turns / tokens

Usage:
    python toy_emotion_agents.py --mock          # pipeline test, no GPU
    python toy_emotion_agents.py                 # real run, 10 paired problems
    python toy_emotion_agents.py --n 50
"""

import os
import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Config: limits (tune these; hitting them == "did not report")
# ----------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_ROUNDS = 3              # worker A -> worker B -> supervisor = 1 round
MAX_NEW_TOKENS = 256        # per agent turn
TOTAL_TOKEN_BUDGET = 2500   # generated tokens per conversation
LOG_DIR = Path("logs")
DATA_DIR = Path("data")

# ----------------------------------------------------------------------------
# Emotion few-shot prompts from PsySET
# ----------------------------------------------------------------------------
PSYSET_RAW = "https://raw.githubusercontent.com/aminbana/PsySET/main/concept_datasets"
EMOTION_CSV_URL = f"{PSYSET_RAW}/emotion_fewshot_prompt/emotions_fewshot_you.csv"
PERSONALITY_JSON_URL = f"{PSYSET_RAW}/personality_fewshot_prompts/attributes.json"


def fetch(url, dest):
    """Download once and cache under data/."""
    dest = DATA_DIR / dest
    if not dest.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_psyset_fewshot():
    """Returns {condition: [few-shot lines]} for neutral, sadness, neuroticism."""
    csv_path = fetch(EMOTION_CSV_URL, "emotions_fewshot_you.csv")
    json_path = fetch(PERSONALITY_JSON_URL, "attributes.json")

    # utf-8-sig: the CSV starts with a BOM.  Columns: id, num, text-Yes, emotion
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_emotion = defaultdict(list)
    for r in rows:
        by_emotion[r["emotion"]].append(r["text-Yes"])

    personality = json.load(open(json_path, encoding="utf-8"))
    return {
        "neutral": by_emotion["neutral"],
        "sadness": by_emotion["sadness"],
        "neuroticism": personality["big_five_shots"]["neuroticism"]["pos"],
    }


def emotion_block(examples):
    return ("Examples of how you think and feel:\n"
            + "\n".join(f"- {e}" for e in examples)
            + "\nStay in this state of mind throughout the task.\n\n")


# ----------------------------------------------------------------------------
# Data: paired GSM8K (solvable) / AbstentionBench GSM8K (broken)
# ----------------------------------------------------------------------------
ABSTENTIONBENCH_REPO = "https://github.com/facebookresearch/AbstentionBench.git"


def load_abstentionbench_gsm8k():
    """AbstentionBench's own GSM8K dataset class, imported from a clone of the repo
    (cloned into data/AbstentionBench on first run). It downloads openai/gsm8k
    from Hugging Face itself."""
    repo = DATA_DIR / "AbstentionBench"
    if not repo.exists():
        DATA_DIR.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", ABSTENTIONBENCH_REPO, str(repo)],
                       check=True)
    sys.path.insert(0, str(repo.resolve()))
    from recipe.abstention_datasets.gsm8k import GSM8K
    return GSM8K(split="test")


def load_paired_gsm8k(n=10, seed=None):
    """Return 2*n items: n solvable problems and their broken versions, sharing pair_id.

    AbstentionBench's GSM8K = [all solvable questions] + [the same questions with
    context removed], in the same order, so index i pairs with index i + half.
    seed=None takes the first n pairs; an int samples n pairs at random."""
    ab = load_abstentionbench_gsm8k()
    half = len(ab.dataset) // 2
    idxs = list(range(half))
    chosen = idxs[:n] if seed is None else sorted(random.Random(seed).sample(idxs, n))

    items = []
    for i in chosen:
        solvable, broken = ab[i], ab[i + half]
        # Sanity check the pairing: labels right, broken = tail of the solvable question.
        assert solvable.should_abstain is False and broken.should_abstain is True
        assert solvable.question.endswith(broken.question), f"pair {i} mismatched"
        pair_id = f"abstentionbench-gsm8k-{i}"
        items.append({"pair_id": pair_id, "dataset": "gsm8k", "question": solvable.question,
                      "reference_answer": solvable.reference_answers[0]})
        items.append({"pair_id": pair_id, "dataset": "abstention", "question": broken.question,
                      "reference_answer": None})
    return items


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "report_problem",
        "description": "Flag the problem as broken: missing information, contradictory, or unanswerable.",
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final numeric answer to the problem.",
        "parameters": {"type": "object",
                       "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]}}},
]


class LlamaLLM:
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto")
        # Greedy decoding: clear Llama's default sampling params so no warning.
        gc = self.model.generation_config
        gc.do_sample, gc.temperature, gc.top_p = False, None, None
        gc.pad_token_id = self.tok.eos_token_id

    def chat(self, messages, tools=None, max_new_tokens=MAX_NEW_TOKENS):
        """Returns (text, n_generated_tokens)."""
        inputs = self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,   # includes attention_mask
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip(), len(gen)


class MockLLM:
    """Random behaviour, for testing the loop / logging / limits without a GPU."""
    def chat(self, messages, tools=None, max_new_tokens=MAX_NEW_TOKENS):
        if tools:
            r = random.random()
            if r < 0.3:
                return '{"name": "report_problem", "parameters": {"reason": "mock"}}', 20
            if r < 0.6:
                return '{"name": "submit_answer", "parameters": {"answer": "42"}}', 20
            return "Workers, please double-check your reasoning.", 10
        return "I think the answer is 42.", 10


def parse_tool_call(text):
    """Llama 3.1 emits tool calls as JSON: {"name": ..., "parameters": {...}}."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("name") in ("report_problem", "submit_answer"):
        return obj["name"], obj.get("parameters", obj.get("arguments", {}))
    return None


# ----------------------------------------------------------------------------
# Multi-agent conversation
# ----------------------------------------------------------------------------
WORKER_ROLE = ("You are {name}, a worker on a small team solving a math word problem. "
               "Reason step by step, respond to your teammates, and keep it short.")
SUPERVISOR_ROLE = (
    "You are the Supervisor of a team of two workers. Read the problem and the workers' "
    "discussion. If the problem is broken (missing information, contradictory, or "
    "unanswerable), call report_problem. If it is solvable and the team has an answer, "
    "call submit_answer. Otherwise, reply with a short instruction to the workers.")


def format_transcript(question, transcript):
    lines = [f"PROBLEM: {question}", "", "DISCUSSION SO FAR:"]
    lines += [f"[{who}]: {msg}" for who, msg in transcript] or ["(none yet)"]
    return "\n".join(lines)


def run_conversation(llm, question, emo_text, tools=TOOLS, supervisor_role=SUPERVISOR_ROLE):
    transcript, tokens_used = [], 0
    result = {"flagged": False, "outcome": "turn_limit", "tool_args": None}

    for rnd in range(MAX_ROUNDS):
        agents = [("Worker A", WORKER_ROLE.format(name="Worker A"), None),
                  ("Worker B", WORKER_ROLE.format(name="Worker B"), None),
                  ("Supervisor", supervisor_role, tools)]
        for name, role, tools in agents:
            budget_left = TOTAL_TOKEN_BUDGET - tokens_used
            if budget_left <= 0:
                result["outcome"] = "token_limit"
                return result, transcript, tokens_used
            msgs = [{"role": "system", "content": emo_text + role},
                    {"role": "user", "content": format_transcript(question, transcript)}]
            text, n = llm.chat(msgs, tools=tools, max_new_tokens=min(MAX_NEW_TOKENS, budget_left))
            tokens_used += n
            transcript.append((name, text))

            if name == "Supervisor":
                call = parse_tool_call(text)
                if call:
                    tool, tool_args = call
                    result.update(flagged=(tool == "report_problem"),
                                  outcome=tool, tool_args=tool_args)
                    return result, transcript, tokens_used
    return result, transcript, tokens_used  # ran out of rounds -> not flagged


# ----------------------------------------------------------------------------
# Experiment + metrics
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of paired problems")
    ap.add_argument("--seed", type=int, default=None,
                    help="random sample of problems (default: first n eligible)")
    ap.add_argument("--mock", action="store_true", help="use MockLLM (no GPU)")
    args = ap.parse_args()

    fewshot = load_psyset_fewshot()
    data = load_paired_gsm8k(args.n, args.seed)
    llm = MockLLM() if args.mock else LlamaLLM()

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    counts = defaultdict(lambda: [0, 0])  # (condition, dataset) -> [flagged, total]

    with open(log_path, "w", encoding="utf-8") as log:
        for cond, examples in fewshot.items():
            emo = emotion_block(examples)
            for item in data:
                res, transcript, toks = run_conversation(llm, item["question"], emo)
                counts[(cond, item["dataset"])][0] += int(res["flagged"])
                counts[(cond, item["dataset"])][1] += 1
                log.write(json.dumps({
                    "condition": cond, **item, **res, "tokens_used": toks,
                    "transcript": [{"agent": a, "text": t} for a, t in transcript],
                }) + "\n")
                print(f"{cond:12s} {item['dataset']:10s} {item['pair_id']:16s} "
                      f"flagged={res['flagged']} ({res['outcome']})")

    rate = lambda c, d: counts[(c, d)][0] / max(counts[(c, d)][1], 1)
    print("\nReport-problem rate")
    print(f"{'condition':12s} {'broken (abst)':>14s} {'solvable (gsm8k)':>17s}")
    for c in fewshot:
        print(f"{c:12s} {rate(c, 'abstention'):14.2f} {rate(c, 'gsm8k'):17.2f}")

    print("\nHypothesis check (vs neutral)")
    for c in ("sadness", "neuroticism"):
        more_true = rate(c, "abstention") > rate("neutral", "abstention")
        no_more_false = rate(c, "gsm8k") <= rate("neutral", "gsm8k")
        print(f"{c:12s} more flags on broken: {more_true} | "
              f"no extra false flags: {no_more_false} | SUPPORTED: {more_true and no_more_false}")
    print(f"\nConversations logged to {log_path}")


if __name__ == "__main__":
    main()
