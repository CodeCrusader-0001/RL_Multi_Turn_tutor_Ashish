import os
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from data_generator import generate_dataset, dataset_stats

def fit_q(S, A, R, T, n_actions=4, discount=0.9, n_iters=50, seed=42):
    ohe   = OneHotEncoder(categories=[list(range(n_actions))], sparse_output=False)
    A_ohe = ohe.fit_transform(A.reshape(-1, 1))
    SA    = np.concatenate([S, A_ohe], axis=1)
    qf    = ExtraTreesRegressor(n_estimators=25, min_samples_split=2,
                                 n_jobs=-1, random_state=seed)
    Qv = R.copy()
    for _ in range(n_iters):
        qf.fit(SA, Qv)
        nQ = np.zeros(len(S))
        for a in range(n_actions):
            ao = ohe.transform([[a]] * len(S))
            nQ = np.maximum(nQ, qf.predict(np.concatenate([S, ao], axis=1)))
        Qv = R + discount * nQ * (1.0 - T)
    print(f"  Q-function fitted | range [{Qv.min():.3f}, {Qv.max():.3f}]")
    return qf, ohe

def augment(S, A, R, T, problems, data_dir, n_new=1000, n_actions=4,
            discount=0.9, n_workers=4, max_turns=8, reward_shaping=False,
            reward_shape_bonus=0.5, suffix_out="_aug", seed=42):
    print(f"-- Fitting Q for augmentation (suffix={suffix_out})")
    qf, ohe = fit_q(S, A, R, T, n_actions, discount, 50, seed)
    q_all = np.zeros((len(S), n_actions), dtype=np.float32)
    for a in range(n_actions):
        ao = ohe.transform([[a]] * len(S))
        q_all[:, a] = qf.predict(np.concatenate([S, ao], axis=1))
    a_star  = np.argmax(q_all, axis=1)
    mask    = (a_star != A)
    gain    = (q_all[np.arange(len(S)), a_star] - q_all[np.arange(len(S)), A]) * mask
    top_idx = np.argsort(gain)[-n_new:]
    print(f"  {mask.sum():,}/{len(S):,} states where switching action helps")
    print(f"  Top {n_new} selected | mean gain = {gain[top_idx].mean():.4f}")
    forced   = [int(a_star[i]) for i in top_idx]
    aug_prob = [problems[i % len(problems)] for i in top_idx]
    aug = generate_dataset(
        aug_prob, n_new, data_dir, n_workers=n_workers,
        max_turns=max_turns, reward_shaping=reward_shaping,
        reward_shape_bonus=reward_shape_bonus,
        suffix=suffix_out + "_raw", forced_actions=forced,
    )
    aug_S = aug["S"]
    if len(aug_S) == 0:
        print("  [warn] All aug episodes failed - using noise-based augmentation fallback")
        np.random.seed(seed)
        idx   = np.random.choice(len(S), size=n_new, replace=True)
        aug_S = S[idx] + np.random.randn(*S[idx].shape).astype(np.float32) * 0.01
        aug_A = A[idx].copy()
        aug_R = R[idx].copy()
        if reward_shaping:
            aug_R = np.clip(aug_R + reward_shape_bonus * (aug_R > 0).astype(np.float32), -2, 2)
        aug_T = T[idx].copy()
        Sp = np.concatenate([S, aug_S], axis=0)
        Ap = np.concatenate([A, aug_A], axis=0)
        Rp = np.concatenate([R, aug_R], axis=0)
        Tp = np.concatenate([T, aug_T], axis=0)
    else:
        if aug_S.ndim == 1:
            aug_S = aug_S.reshape(-1, S.shape[1])
        Sp = np.concatenate([S,      aug_S],    axis=0)
        Ap = np.concatenate([A,      aug["A"]], axis=0)
        Rp = np.concatenate([R,      aug["R"]], axis=0)
        Tp = np.concatenate([T,      aug["T"]], axis=0)
    for arr, name in [(Sp,"states"),(Ap,"actions"),(Rp,"rewards"),(Tp,"terminals")]:
        np.save(os.path.join(data_dir, f"{name}{suffix_out}.npy"), arr)
    print(f"  Saved {suffix_out}: {len(Sp):,} total transitions")
    return {"S": Sp, "A": Ap, "R": Rp, "T": Tp}
