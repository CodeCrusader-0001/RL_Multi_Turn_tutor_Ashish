import re, random
import numpy as np
from llm_utils import call_llm

N_ACTIONS    = 4
ACTION_NAMES = ["Instruct", "Encourage", "Refocus", "Question"]

ACTION_PROMPTS = {
    0: "You are a math tutor. Give ONE short hint that moves the student forward WITHOUT revealing the answer. Max 2 sentences.",
    1: "You are a supportive math tutor. Briefly encourage the student and acknowledge what they did right. Max 2 sentences.",
    2: "You are a patient math tutor. The student is distracted - gently redirect them back to the problem. Max 2 sentences.",
    3: "You are a Socratic math tutor. Ask exactly ONE targeted question to check the student's understanding of the next step. Max 1 sentence.",
}

def tutor_response(dialogue, action):
    system = ACTION_PROMPTS[int(action)]
    prompt = f"Conversation so far:\n{dialogue}\n\nTutor:"
    resp   = call_llm(prompt, max_tokens=80, temperature=0.7, system=system)
    return resp if resp else "Let's think through this step by step."

def student_response(dialogue, question):
    prompt = (
        "You are a 6th-grade student who struggles with math. "
        "Keep your reply to 1-2 sentences. Give a NUMBER if you think you know the answer.\n\n"
        f"Problem: {question}\n\nConversation:\n{dialogue}\n\nStudent:"
    )
    resp = call_llm(prompt, max_tokens=50, temperature=0.9)
    return resp if resp else "I'm not sure how to do this."

def extract_number(text):
    nums = re.findall(r"\b\d+(?:[,.]\d+)?\b", str(text))
    return nums[-1].replace(",", "") if nums else ""

def shape_reward(reward, turn, max_turns, bonus=0.5):
    if reward > 0:
        return reward + bonus * (1.0 - turn / max(max_turns, 1))
    return reward

class PromptPolicy:
    def predict_from_dialogue(self, dialogue, question):
        prompt = (
            "You are a math tutor. Choose the BEST next action.\n"
            f"Problem: {question}\n\nConversation:\n{dialogue}\n\n"
            "Reply with ONLY one digit: 0=Instruct  1=Encourage  2=Refocus  3=Question\n"
            "Choice:"
        )
        resp = call_llm(prompt, max_tokens=5, temperature=0.2)
        m = re.search(r"[0-3]", resp)
        return int(m.group()) if m else random.randint(0, N_ACTIONS - 1)

    def predict(self, state):
        if isinstance(state, np.ndarray) and state.ndim > 1:
            return np.zeros(state.shape[0], dtype=np.int32)
        return 0

    def predict_value(self, state, action):
        return np.zeros(len(state) if hasattr(state, "__len__") else 1)

class D3RLPyPolicy:
    def __init__(self, model):
        self.model = model

    def predict(self, state):
        import numpy as np
        s = np.asarray(state, dtype=np.float32)
        if s.ndim == 1:
            s = s[np.newaxis, :]
        return int(self.model.predict(s)[0])

class SuperCQLPolicy:
    def __init__(self, models):
        self.models   = models
        self.policies = [D3RLPyPolicy(m) for m in models]

    def predict(self, state):
        import numpy as np
        votes  = [p.predict(state) for p in self.policies]
        counts = np.bincount(votes, minlength=N_ACTIONS)
        return int(np.argmax(counts))
