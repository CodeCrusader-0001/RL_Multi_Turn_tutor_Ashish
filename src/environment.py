import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import STATE_DIM, N_ACTIONS, DISCOUNT, MAX_TURNS
from llm_utils import call_mistral, extract_number

class TutorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, problems, max_turns=MAX_TURNS):
        super().__init__()
        self.problems   = problems
        self.max_turns  = max_turns
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(STATE_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)
        self._problem  = None
        self._state    = None
        self._turn     = 0
        self._history  = []
        self._dialogue = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._problem  = self.np_random.choice(self.problems) if hasattr(self, "np_random") \
                         else self.problems[np.random.randint(len(self.problems))]
        self._turn     = 0
        self._history  = []
        self._dialogue = f"Student: I need help with: {self._problem['question']}"
        self._state    = self._encode_state(False)
        return self._state.copy(), {}

    def step(self, action):
        tutor_resp   = self._tutor_response(action)
        self._dialogue += f"\nTutor: {tutor_resp}"
        student_resp = self._student_response()
        self._dialogue += f"\nStudent: {student_resp}"
        self._history.append({"action": int(action), "tutor": tutor_resp, "student": student_resp})
        self._turn += 1
        solved = self._check_solved(student_resp)
        if solved:
            reward = 1.0
        elif self._turn >= self.max_turns:
            reward = -1.0
        else:
            reward = 0.0
        terminated = solved or self._turn >= self.max_turns
        self._state = self._encode_state(solved)
        return self._state.copy(), reward, terminated, False, {"solved": solved}

    def _tutor_response(self, action):
        styles = [
            "Give ONE short hint about what concept to think about, without showing any calculations or steps",
            "Encourage the student warmly without giving any hints about the solution",
            "Gently redirect the student back to focus on the problem without giving hints",
            "Ask the student ONE short question to check their understanding, without revealing anything"
        ]
        hist = "\n".join([f"Tutor: {h['tutor']}\nStudent: {h['student']}"
                           for h in self._history[-2:]])
        q = self._problem["question"]
        prompt = (f"You are a math tutor helping a struggling 6th grade student.\n"
                  f"Your goal: {styles[action]}.\n"
                  f"Problem: {q}\n"
                  f"Recent history: {hist if hist else 'None'}\n"
                  f"IMPORTANT: Do NOT show any calculations, numbers, or solution steps. "
                  f"Do NOT reveal the answer or intermediate values. "
                  f"Keep response under 40 words.")
        resp = call_mistral(prompt, max_tokens=60, temperature=0.7)
        return resp if resp else "Let me help you think through this."

    def _student_response(self):
        q = self._problem["question"]
        prompt = (f"You are a 6th grade student who is NOT good at math and struggles with word problems.\n"
                  f"You often make arithmetic mistakes and get confused easily.\n"
                  f"You only give a final numerical answer if you are very confident after multiple hints.\n"
                  f"Problem: {q}\n"
                  f"Conversation so far:\n{self._dialogue[-1500:]}\n"
                  f"Respond in 1-2 sentences like a confused student. "
                  f"Only state a number if you are fully confident. Otherwise express confusion or ask for help.")
        resp = call_mistral(prompt, max_tokens=60, temperature=0.9)
        return resp if resp else "I am not sure how to do this."

    def _check_solved(self, student_resp):
        ans = extract_number(self._problem.get("answer", ""))
        got = extract_number(student_resp)
        if ans is None or got is None:
            return False
        try:
            return abs(float(ans) - float(got)) < 0.5
        except ValueError:
            return False

    def _encode_state(self, solved):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[0] = self._turn / max(self.max_turns, 1)
        s[1] = float(solved)
        s[2] = len(self._history) / max(self.max_turns, 1)
        if self._history:
            for i, h in enumerate(self._history[-4:]):
                s[3 + i] = h["action"] / 3.0
        tutor_q   = sum(1 for h in self._history if "?" in h.get("tutor", ""))
        student_q = sum(1 for h in self._history if "?" in h.get("student", ""))
        s[7] = min(tutor_q / 10.0, 1.0)
        s[8] = min(student_q / 10.0, 1.0)
        return s
