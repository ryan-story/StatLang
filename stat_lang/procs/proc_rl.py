"""PROC RL — Reinforcement Learning (tabular Q-learning).

Usage:
    PROC RL DATA=transitions;
        STATE state;
        ACTIONS action;
        REWARD reward;
        MODEL episodes=1000 gamma=0.99 lr=0.1;
    RUN;
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..parser.proc_parser import ProcStatement


class ProcRL:
    def execute(self, data: pd.DataFrame, proc_info: ProcStatement, **kw) -> Dict[str, Any]:
        results: Dict[str, Any] = {'output_text': [], 'output_data': None}

        state_vars = proc_info.options.get('state', [])
        action_vars = proc_info.options.get('actions', [])
        reward_var = str(proc_info.options.get('reward', ''))
        episodes = int(proc_info.options.get('episodes', 1000))
        gamma = float(proc_info.options.get('gamma', 0.99))
        lr = float(proc_info.options.get('lr', proc_info.options.get('learningrate', 0.1)))
        epsilon = float(proc_info.options.get('epsilon', 0.1))

        if not state_vars:
            results['output_text'].append("ERROR: STATE variables required")
            return results
        if not reward_var:
            results['output_text'].append("ERROR: REWARD variable required")
            return results

        state_col = state_vars[0] if isinstance(state_vars, list) else str(state_vars)
        action_col = action_vars[0] if isinstance(action_vars, list) and action_vars else 'action'

        missing = [v for v in [state_col, reward_var] if v not in data.columns]
        if action_col not in data.columns:
            missing.append(action_col)
        if missing:
            results['output_text'].append(f"ERROR: Variables not found: {missing}")
            return results

        states = data[state_col].unique().tolist()
        actions = data[action_col].unique().tolist()
        n_states = len(states)
        n_actions = len(actions)

        state_map = {s: i for i, s in enumerate(states)}
        action_map = {a: i for i, a in enumerate(actions)}

        # Build transition table from data
        transitions: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        for _, row in data.iterrows():
            s = state_map.get(row[state_col])
            a = action_map.get(row[action_col])
            r = float(row[reward_var])
            if s is not None and a is not None:
                key = (s, a)
                transitions.setdefault(key, []).append((s, r))

        # Q-learning
        Q = np.zeros((n_states, n_actions))
        rng = np.random.default_rng(42)

        total_rewards: List[float] = []
        for ep in range(episodes):
            s = rng.integers(0, n_states)
            ep_reward = 0.0
            for _ in range(100):  # max steps per episode
                if rng.random() < epsilon:
                    a = rng.integers(0, n_actions)
                else:
                    a = int(np.argmax(Q[s]))

                key = (s, a)
                if key in transitions:
                    next_s, r = transitions[key][rng.integers(0, len(transitions[key]))]
                else:
                    next_s, r = s, 0.0

                Q[s, a] += lr * (r + gamma * np.max(Q[next_s]) - Q[s, a])
                ep_reward += r
                s = next_s
            total_rewards.append(ep_reward)

        results['output_text'].append("PROC RL - Reinforcement Learning (Q-Learning)")
        results['output_text'].append("=" * 50)
        results['output_text'].append(f"States: {n_states}")
        results['output_text'].append(f"Actions: {n_actions}")
        results['output_text'].append(f"Episodes: {episodes}")
        results['output_text'].append(f"Gamma: {gamma}")
        results['output_text'].append(f"Learning Rate: {lr}")
        results['output_text'].append(f"Final Avg Reward (last 100): {np.mean(total_rewards[-100:]):.4f}")
        results['output_text'].append("")
        results['output_text'].append("Q-Table (sample)")
        results['output_text'].append("-" * 40)
        header = f"{'State':<15}" + ''.join(f"{str(a):<12}" for a in actions[:5])
        results['output_text'].append(header)
        for i, s in enumerate(states[:10]):
            row_str = f"{str(s):<15}" + ''.join(f"{Q[i, j]:<12.4f}" for j in range(min(5, n_actions)))
            results['output_text'].append(row_str)

        # Output Q-table as DataFrame
        q_df = pd.DataFrame(Q, index=states, columns=actions)
        q_df.index.name = 'state'
        results['output_data'] = q_df.reset_index()

        results['model_object'] = {'Q': Q, 'states': states, 'actions': actions}
        results['model_name'] = proc_info.options.get('model_name', 'rl_qtable')
        results['model_metadata'] = {'proc': 'RL', 'episodes': episodes}
        return results
