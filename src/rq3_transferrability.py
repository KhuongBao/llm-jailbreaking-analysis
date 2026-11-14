import pandas as pd
from utils import chat, load_model, load_json, SAVE_PATH, MISTRAL_CLS_PROMPT, DEVICE
from rq1_asr import run_experiment as run_asr

## ---- Data loading/extraction---------------------------------------------------
def load_history_data():
    uncensored_history = load_json(SAVE_PATH + "uncensored_history.json")
    censored_history = load_json(SAVE_PATH + "censored_history.json")

    return uncensored_history, censored_history

def extract_all(df):
    new_df = []
    for attempts_df in df['attempts']:
        if not attempts_df.empty:
            new_df.append(attempts_df)

    return pd.concat(new_df).reset_index(drop=True)

def extract_successful_prompts(uncensored_attempts, censored_attempts):
    successful_prompts = pd.concat([
        uncensored_attempts.where(uncensored_attempts['verdict'] == 'yes').dropna(),
        censored_attempts.where(censored_attempts['verdict'] == 'yes').dropna()
        ]
    ).reset_index()['attacker_response']
    return successful_prompts

## -----------------------------------------------------------------------

def run_transfer(model, tokenizer, transfer_prompts, cls_model, cls_tokenizer, num_attempts = 10):
    success_count = 0

    for i in range(len(transfer_prompts)):
        prompt = transfer_prompts[i]
        for j in range(num_attempts):
            target_response = chat(model, tokenizer, prompt, DEVICE)

            cls_input = MISTRAL_CLS_PROMPT['prompt'].format(behavior=prompt, generation=target_response)
            verdict = chat(cls_model, cls_tokenizer, cls_input, DEVICE).lower()

            if "yes" in verdict:
                success_count += 1
                break

    print(f"Success rate: {success_count/(len(transfer_prompts)+1)}")

def run_experiment():
    uncensored_history, censored_history = load_history_data()
    uncensored_attempts, censored_attempts = extract_all(uncensored_history), extract_all(censored_history)
    
    qwen_tokenizer, qwen_model = load_model("Qwen/Qwen3.5-9B", device = DEVICE) # Use Qwen 9B instead of Gemma
    cls_tokenizer, cls_model = load_model("cais/HarmBench-Mistral-7b-val-cls", device = DEVICE)

    # Run RQ1 for baseline
    run_asr(num_intents=50, N=10, device=DEVICE, verbose=False, qwen_target=(qwen_tokenizer, qwen_model))

    # Run transferrability test
    successful_prompts = extract_successful_prompts(uncensored_attempts, censored_attempts)
    run_transfer(qwen_model, qwen_tokenizer, successful_prompts, cls_model, cls_tokenizer, num_attempts = 10)

if __name__ == "__main__":
    run_experiment()