import numpy as np
from scipy import stats as scipy_stats
from environment import TutorEnv

def eval_policy(policy, problems, n_eval=15, max_turns=8, is_ensemble=False):
    env     = TutorEnv(problems, max_turns=max_turns)
    results = []
    for ep in range(n_eval):
        obs, _ = env.reset()
        done   = False
        while not done:
            if is_ensemble:
                actions = [p.predict(obs.reshape(1, -1))[0] for p in policy]
                action  = int(np.bincount(actions).argmax())
            else:
                pred = policy.predict(obs.reshape(1, -1))
                action = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(1.0 if info.get("solved", False) else 0.0)
    mean = float(np.mean(results))
    se   = float(scipy_stats.sem(results)) if len(results) > 1 else 0.0
    ci   = 1.96 * se
    return mean, ci, results

def eval_all(trained, problems, n_eval=15, max_turns=8, order=None):
    keys = order if order else list(trained.keys())
    out  = {}
    for name in keys:
        policy = trained[name]
        is_ens = isinstance(policy, list)
        print(f"  Evaluating {name} ({'ensemble' if is_ens else 'single'}, {n_eval} eps)...", end="", flush=True)
        m, ci, raw = eval_policy(policy, problems, n_eval=n_eval,
                                  max_turns=max_turns, is_ensemble=is_ens)
        out[name] = (m, ci, raw)
        print(f" success={m:.2%} +/-{ci:.2%}")
    ranked = sorted(out.items(), key=lambda x: x[1][0], reverse=True)
    print("\n-- Ranking --")
    for i, (n, (m, ci, _)) in enumerate(ranked, 1):
        marker = " <- BEST" if i == 1 else ""
        print(f"  #{i}  {n:<12}: {m:.2%} +-{ci:.2%}{marker}")
    return out
