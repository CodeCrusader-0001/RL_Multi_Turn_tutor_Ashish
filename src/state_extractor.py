import re, threading
import numpy as np

STATE_DIM = 25
_encoder  = None
_enc_lock = threading.Lock()

_OFFTOPIC = ["anyway", "by the way", "never mind", "forget it", "random"]
_MATH_KW  = ["equation", "formula", "calculate", "solve", "multiply", "divide",
             "add", "subtract", "equals", "answer", "result", "total"]

def _enc():
    global _encoder
    if _encoder is None:
        with _enc_lock:
            if _encoder is None:
                from sentence_transformers import SentenceTransformer
                import torch
                m = SentenceTransformer("all-MiniLM-L6-v2")
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                _encoder = m.to(device)
    return _encoder

def extract_state(dialogue, turn, successes_so_far=0, max_turns=8,
                  tutor_q_count=0, student_q_count=0):
    emb = _enc().encode(dialogue, show_progress_bar=False)
    d   = dialogue.lower()
    nl  = chr(10)
    s_lines = [l for l in dialogue.split(nl) if l.startswith("Student:")]
    t_lines = [l for l in dialogue.split(nl) if l.startswith("Tutor:")]
    last_s  = s_lines[-1].replace("Student:", "").strip().lower() if s_lines else ""
    last_t  = t_lines[-1].replace("Tutor:", "").strip().lower()   if t_lines else ""
    f_math   = float(any(kw in last_s for kw in _MATH_KW)
                     or bool(re.search(r"\d+[\+\-\*/=]\s*\d", last_s)))
    f_solved = float("correct" in d[-200:] or "right answer" in d[-200:])
    f_reex   = float(any(p in last_s for p in
                         ["again", "re-explain", "clarify", "what do you mean", "i dont get"]))
    f_repeat = float(len(last_s) > 5 and last_s[:20] in last_t[:50]) if last_t else 0.0
    f_ooft   = float(any(w in last_s for w in _OFFTOPIC))
    f_unrel  = float(len(last_s) > 0 and not f_math
                     and not any(k in last_s for k in ["problem", "answer", "step", "think"]))
    f_sq     = float("?" in last_s)
    hand  = np.array([f_math, f_solved, f_reex, f_repeat,
                      f_ooft, f_unrel, f_sq], dtype=np.float32)
    state = np.concatenate([emb[:18].astype(np.float32), hand])
    assert state.shape[0] == STATE_DIM, f"State dim mismatch: {state.shape[0]} != {STATE_DIM}"
    return state
