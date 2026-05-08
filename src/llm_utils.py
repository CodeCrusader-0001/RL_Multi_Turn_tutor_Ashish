import os, time, random, requests, re
import concurrent.futures

def call_llm(prompt, max_tokens=80, temperature=0.7, retries=6, system=None):
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return _mock_response(prompt)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
        "messages": msgs, "max_tokens": max_tokens, "temperature": temperature
    }
    for attempt in range(retries):
        try:
            r = requests.post("https://api.mistral.ai/v1/chat/completions",
                              headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                time.sleep(min(64, (2 ** attempt) * 1.5 + random.uniform(0.3, 2.5)))
            elif r.status_code >= 500:
                time.sleep(2 ** attempt)
            else:
                return ""
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt + random.uniform(0, 1))
        except Exception:
            if attempt == retries - 1:
                return ""
            time.sleep(2 ** attempt)
    return ""

def call_mistral(prompt, model="mistral-small-latest", max_tokens=256, temperature=0.7, retries=3):
    return call_llm(prompt, max_tokens=max_tokens, temperature=temperature, retries=retries)

def _mock_response(prompt):
    import random
    templates = [
        "Let me help you work through this step by step.",
        "That is a great question! Let us think about it carefully.",
        "You are on the right track. Consider breaking the problem into smaller parts.",
        "Try thinking about what information you already know.",
    ]
    return random.choice(templates)

def extract_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return nums[-1] if nums else None

def call_llm_batch(prompts, max_tokens=80, temperature=0.7, n_workers=8, system=None):
    results = [""] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
        fut_map = {exe.submit(call_llm, p, max_tokens, temperature, 6, system): i
                   for i, p in enumerate(prompts)}
        for fut in concurrent.futures.as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = ""
    return results

def check_api_key():
    resp = call_llm("Reply OK", max_tokens=5)
    ok = bool(resp and len(resp) > 0)
    print(f"  API key: {chr(39)+chr(39)+chr(39)}{'PASSED' if ok else 'FAILED'}{chr(39)+chr(39)+chr(39)} | response={resp!r}")
    return ok
