import os, json, random, concurrent.futures
import numpy as np
from tqdm import tqdm
from state_extractor import extract_state
from tutor_policy import (tutor_response, student_response,
                          extract_number, shape_reward, N_ACTIONS)

def dataset_stats(S, A, R, label="Dataset"):
    n_trans   = len(S)
    n_success = int((R > 0).sum())
    diversity = len(set(tuple([a.tolist()] if hasattr(a, "tolist") else [a]) for a in A)) / max(len(A), 1)
    sr = n_success / max(n_trans, 1)
    print(f"  {label}: {n_trans:,} transitions | success_rate={sr:.2%} | action_diversity={diversity:.2%}")
    return {"label": label, "n": n_trans, "success": sr, "diversity": diversity}

def run_episode(problem, max_turns=8, forced_first_action=None,
                reward_shaping=False, reward_shape_bonus=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    q, ans   = problem["question"], problem["answer"]
    nl       = chr(10)
    dialogue = f"Student: I need help with: {q}"
    states, actions, rewards, terminals = [], [], [], []
    successes = tq_cnt = sq_cnt = 0
    solved = False
    for t in range(max_turns):
        state  = extract_state(dialogue, t, successes, max_turns, tq_cnt, sq_cnt)
        action = (forced_first_action if (forced_first_action is not None and t == 0)
                  else random.randint(0, N_ACTIONS - 1))
        t_resp = tutor_response(dialogue, action)
        dialogue += nl + "Tutor: " + t_resp
        if "?" in t_resp: tq_cnt += 1
        s_resp = student_response(dialogue, q)
        dialogue += nl + "Student: " + s_resp
        if "?" in s_resp: sq_cnt += 1
        s_ans = extract_number(s_resp)
        c_ans = extract_number(ans)
        raw_r = 1.0 if (s_ans and c_ans and s_ans == c_ans) else 0.0
        reward = shape_reward(raw_r, t, max_turns, reward_shape_bonus) if reward_shaping else raw_r
        if raw_r > 0:
            successes += 1
            solved = True
        is_last = solved or (t == max_turns - 1)
        states.append(state); actions.append(action)
        rewards.append(reward); terminals.append(float(is_last))
        if solved: break
    return {"states": states, "actions": actions, "rewards": rewards,
            "terminals": terminals, "solved": solved, "n_turns": len(states)}

def _save_ckpt(path, states, actions, rewards, terminals, done):
    json.dump({"states": [s.tolist() for s in states], "actions": actions,
               "rewards": rewards, "terminals": terminals, "done_count": done},
              open(path, "w"))

def generate_dataset(problems, n_episodes, data_dir, n_workers=4, ckpt_every=200,
                     max_turns=8, reward_shaping=False, reward_shape_bonus=0.5,
                     suffix="", forced_actions=None):
    ckpt = os.path.join(data_dir, f"checkpoint{suffix}.json")
    if os.path.exists(ckpt):
        c     = json.load(open(ckpt))
        all_S = [np.array(s, dtype=np.float32) for s in c["states"]]
        all_A = list(c["actions"])
        all_R = list(c["rewards"])
        all_T = list(c["terminals"])
        done  = c.get("done_count", 0)
        print(f"  Resumed from checkpoint: {done}/{n_episodes}")
    else:
        all_S, all_A, all_R, all_T = [], [], [], []
        done = 0
    pool = (problems * (n_episodes // len(problems) + 1))[:n_episodes]
    rem  = pool[done:]
    fa   = (forced_actions[done:] if forced_actions else [None] * len(rem))
    n_sol = sum(1 for r in all_R if r > 0)
    pbar  = tqdm(total=n_episodes, initial=done, desc=f"Gen{suffix}", unit="ep")
    def _run(args):
        p, f = args
        return run_episode(p, max_turns=max_turns, forced_first_action=f,
                           reward_shaping=reward_shaping, reward_shape_bonus=reward_shape_bonus)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_run, a): i for i, a in enumerate(zip(rem, fa))}
        n_errors = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                ep = fut.result()
                all_S.extend(ep["states"]); all_A.extend(ep["actions"])
                all_R.extend(ep["rewards"]); all_T.extend(ep["terminals"])
                if ep["solved"]: n_sol += 1
                done += 1
            except Exception as exc:
                n_errors += 1
                if n_errors <= 5: print(f"  [warn] episode failed ({n_errors}): {exc}")
            pbar.update(1)
            pbar.set_postfix({"T": len(all_S), "sol%": f"{n_sol/max(done,1):.1%}", "err": n_errors})
            if done % ckpt_every == 0:
                _save_ckpt(ckpt, all_S, all_A, all_R, all_T, done)
    pbar.close()
    _save_ckpt(ckpt, all_S, all_A, all_R, all_T, done)
    if len(all_S) == 0:
        S = np.empty((0, 25), dtype=np.float32)
        A = np.empty((0,),    dtype=np.int32)
        R = np.empty((0,),    dtype=np.float32)
        T = np.empty((0,),    dtype=np.float32)
    else:
        S = np.array(all_S, dtype=np.float32)
        if S.ndim == 1: S = S.reshape(-1, 25)
        A = np.array(all_A, dtype=np.int32)
        R = np.array(all_R, dtype=np.float32)
        T = np.array(all_T, dtype=np.float32)
    for arr, name in [(S,"states"),(A,"actions"),(R,"rewards"),(T,"terminals")]:
        np.save(os.path.join(data_dir, f"{name}{suffix}.npy"), arr)
    if len(S) > 0:
        print(f"DONE D{suffix}: {len(S):,} transitions | mean_reward={R.mean():.2%}")
    else:
        print(f"WARN D{suffix}: 0 transitions - check API key / errors above")
    return {"S": S, "A": A, "R": R, "T": T}
