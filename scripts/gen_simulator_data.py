"""
Generate real game-trajectory training data from Game-NN Simulator.

Runs NetworkWorld episodes and records each decision step as a
structured conversation for mulun training.

Usage:
    python scripts/gen_simulator_data.py --episodes 100 --output ../dataset
"""
import os, sys, json, random, argparse, math
from typing import List, Dict, Tuple
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
sys.path.insert(0, _ROOT)

# Simulator paths
_SIM_ROOT = os.path.join(_ROOT, '..', 'Game-nn-O', 'game_nn_simulator')
sys.path.insert(0, _SIM_ROOT)

# ─── Constants (match decision head) ───────────────────────────

ACTIONS = [
    ('BLOCK_IP', '封禁攻击源IP'),
    ('PATCH_VULN', '修补漏洞'),
    ('ISOLATE_HOST', '隔离受影响主机'),
    ('RESTORE_BACKUP', '从备份恢复系统'),
    ('DEEP_SCAN', '深度扫描分析攻击范围'),
    ('HUNT_THREATS', '主动威胁狩猎'),
    ('DEPLOY_HONEYPOT', '部署蜜罐诱捕'),
    ('ESCALATE_INCIDENT', '升级事件至应急响应团队'),
]

STRATEGY_ACTION_MAP = {
    0: [5, 6, 2, 0],  # aggressive
    1: [4, 7, 1, 2],  # balanced
    2: [0, 1, 2, 3],  # defensive
}

STRATEGY_NAMES = ['aggressive', 'balanced', 'defensive']
STATE_NAMES = [
    'threat_severity', 'threat_type_code', 'attack_surface',
    'lateral_movement_risk', 'data_exfil_risk', 'persistence_risk',
    'detection_level', 'attacker_sophistication',
    'critical_asset_count', 'patched_ratio', 'isolated_ratio',
    'compromised_ratio', 'alert_level', 'backup_status',
    'monitoring_coverage', 'incident_response_phase',
]

NODE_TYPE_NAMES = {0: '工作站', 1: '服务器', 2: '数据库', 3: '防火墙', 4: '路由器'}


def extract_state(env) -> List[float]:
    """Extract a 16-dim security state vector from the environment."""
    ns = env.node_state
    comp_ratio = float(ns[:, 1].mean())
    alert_mean = float(ns[:, 5].mean())
    patched_ratio = float(ns[:, 3].mean())
    isolated_ratio = float(ns[:, 6].mean())
    vuln_ratio = float(ns[:, 2].mean())

    return [
        min(1.0, abs(float(ns[:, 1].sum()) * 0.3 + 0.1)),    # threat_severity
        float(max(0, min(5, int(ns[:, 0].mean() * 5)))),      # threat_type_code
        min(1.0, alert_mean * 1.5),                            # attack_surface
        min(1.0, comp_ratio * 2.0),                             # lateral_movement_risk
        min(1.0, comp_ratio * 1.5),                             # data_exfil_risk
        min(1.0, float(ns[:, 7].mean()) * 2.0),                # persistence_risk
        float(env.intrusion_detected),                          # detection_level
        min(1.0, env.time_step / max(env.max_steps, 1)),        # attacker_sophistication
        min(1.0, float(ns[:, 0].mean())),                       # critical_asset_count
        patched_ratio,                                          # patched_ratio
        isolated_ratio,                                         # isolated_ratio
        comp_ratio,                                             # compromised_ratio
        min(1.0, alert_mean * 2.0),                              # alert_level
        max(0.1, 1.0 - comp_ratio),                             # backup_status
        min(1.0, alert_mean),                                   # monitoring_coverage
        min(1.0, env.time_step / max(env.max_steps, 1)),        # incident_response_phase
    ]


def state_to_description(env, state_vec: List[float]) -> str:
    """Translate raw state to a natural language situation report."""
    ns = env.node_state
    num_nodes = env.num_nodes
    comp_nodes = int(ns[:, 1].sum())
    vuln_nodes = int(ns[:, 2].sum())
    isolated_nodes = int(ns[:, 6].sum())
    alert_high = int((ns[:, 5] > 0.5).sum())

    parts = [f'网络共{num_nodes}个节点']
    if comp_nodes > 0:
        parts.append(f'{comp_nodes}个已被攻陷')
    if vuln_nodes > 0:
        parts.append(f'{vuln_nodes}个存在漏洞')
    if isolated_nodes > 0:
        parts.append(f'{isolated_nodes}个已隔离')
    if alert_high > 0:
        parts.append(f'{alert_high}个节点告警级别高')

    sev = state_vec[0]
    if sev < 0.3:
        parts.append('威胁等级较低')
    elif sev < 0.6:
        parts.append('威胁等级中等')
    else:
        parts.append('威胁等级严重')

    phase_idx = min(int(state_vec[15] * 4), 3)
    phase_names = ['检测分析阶段', '遏制处置阶段', '根除清理阶段', '恢复加固阶段']
    parts.append(f'处于{phase_names[phase_idx]}')

    return '，'.join(parts)


def build_think_block(strategy: int, action_name: str, state_vec: List[float],
                       reward: float) -> str:
    """Build a realistic <think> block from game state."""
    lines = []
    sev = state_vec[0]
    comp = state_vec[11]

    # Strategy reasoning
    if strategy == 0 and sev > 0.5:
        lines.append(f'威胁严重度{sev:.0%}，失陷{comp:.0%}，需主动反制')
    elif strategy == 1:
        lines.append(f'情况可控但需进一步分析（严重度{sev:.0%})')
    else:
        lines.append('优先止损防御')

    # Action reasoning
    lines.append(f'执行{action_name}')

    # Outcome reflection from reward
    if reward > 0:
        lines.append(f'效果正面（reward={reward:+.2f}）')
    elif reward < 0:
        lines.append(f'效果不佳（reward={reward:.2f}），需调整方案')
    else:
        lines.append('效果待观察')

    return '\n'.join(lines)


def episode_to_samples(env, episode_log: List[dict], ep_id: int) -> List[dict]:
    """Convert one episode's game log into training samples."""
    samples = []

    for step_idx, log in enumerate(episode_log):
        state_vec = log['state']
        action_idx = log['action']
        reward = log['reward']
        strategy = log['strategy']
        action_name, action_desc = ACTIONS[action_idx]

        # Build description
        situation = state_to_description(env, state_vec)
        state_desc = STATE_NAMES
        state_parts = [f'{state_desc[i]}={state_vec[i]:.2f}' for i in range(16)]
        state_str = ', '.join(state_parts)

        # User message
        if step_idx == 0:
            user_text = (
                f'检测到网络安全事件。{situation}。'
                f'当前状态特征：{state_str}。请做出安全决策。'
            )
        else:
            prev_log = episode_log[step_idx - 1]
            prev_action_name, _ = ACTIONS[prev_log['action']]
            user_text = (
                f'上一步执行{prev_action_name}后（reward={prev_log["reward"]:+.2f}），'
                f'当前情况：{situation}。请继续决策。'
            )

        # Assistant think + action
        think = build_think_block(strategy, action_name, state_vec, reward)

        outcome_str = f'预期遏制概率{max(0.1, min(0.95, 0.5 + reward * 0.3)):.0%}'
        assistant_text = f'<think>\n{think}\n</think>\n\n建议执行{action_name}：{action_desc}。{outcome_str}。'

        conversations = [
            {'role': 'system', 'content': '你是一个专业的网络安全专家。'},
            {'role': 'user', 'content': user_text},
            {'role': 'assistant', 'content': assistant_text},
        ]

        # Structured labels for decision head
        containment = max(0.1, min(0.95, 0.5 + reward * 0.3))
        structured = {
            'state': state_vec,
            'strategy': strategy,
            'strategy_name': STRATEGY_NAMES[strategy],
            'action': action_idx,
            'action_name': action_name,
            'value': containment,
        }

        samples.append({
            'conversations': conversations,
            'structured': structured,
            'metadata': {
                'episode': ep_id,
                'step': step_idx,
                'reward': reward,
                'compromised_ratio': float(env.node_state[:, 1].mean()),
            },
        })

    return samples


def run_one_episode(env, attacker, defender, max_steps: int) -> List[dict]:
    """Run one episode and log every defender decision step."""
    obs = env.reset()
    done = False
    step = 0
    episode_log = []

    while not done and step < max_steps * 2:
        if env.is_attacker_turn:
            action, node, node2 = attacker.act(obs, env.get_available_nodes())
            obs, reward, done, info = env.step(action, node, node2)
        else:
            # Defender's turn — record this
            action, node, _ = defender.act(obs, env.get_available_nodes())
            obs, reward, done, info = env.step(action, node)

            # Defender action is 4-7, map to 0-3
            def_action_idx = action - 4
            state_vec = extract_state(env)

            # Determine strategy from action
            strategy = 1  # default balanced
            for s, acts in STRATEGY_ACTION_MAP.items():
                if def_action_idx in acts:
                    strategy = s
                    break

            episode_log.append({
                'action': def_action_idx,
                'strategy': strategy,
                'reward': reward,
                'state': state_vec,
                'compromised': float(env.node_state[:, 1].mean()),
            })

        step += 1

    return episode_log


def generate_simulator_data(num_episodes: int = 100, max_steps: int = 30) -> List[dict]:
    """Main generator — runs episodes and returns training samples."""
    from network_world.environment import NetworkWorld, RandomAttacker, RandomDefender
    from config import Config

    cfg = Config()
    # Use smaller network for faster generation
    cfg.network.num_nodes = 8  # fewer nodes = faster
    cfg.network.max_steps = max_steps

    env = NetworkWorld(cfg.network)
    attacker = RandomAttacker()
    defender = RandomDefender()

    all_samples = []

    for ep in range(num_episodes):
        if ep % 10 == 0:
            print(f'  Episode {ep}/{num_episodes}...')

        episode_log = run_one_episode(env, attacker, defender, max_steps)
        samples = episode_to_samples(env, episode_log, ep)
        all_samples.extend(samples)

    print(f'Generated {len(all_samples)} decision steps from {num_episodes} episodes')
    return all_samples


def save_jsonl(samples: List[dict], path: str):
    """Save conversations as jsonl."""
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'Saved {len(samples)} samples to {path}')

    # Also save pure structured labels for decision head training
    labels_path = path.replace('.jsonl', '_labels.json')
    structures = [s['structured'] for s in samples]
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(structures, f, indent=2, ensure_ascii=False)
    print(f'Labels saved to {labels_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate simulator game data')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of game episodes')
    parser.add_argument('--max-steps', type=int, default=30,
                        help='Max steps per episode')
    parser.add_argument('--output', type=str,
                        default=os.path.join(_ROOT, '..', 'dataset'),
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print(f'Running {args.episodes} episodes (max {args.max_steps} steps)...')
    samples = generate_simulator_data(args.episodes, args.max_steps)

    # Stats
    strat_dist = {s: 0 for s in STRATEGY_NAMES}
    action_dist = {a[0]: 0 for a in ACTIONS}
    rewards = []
    for s in samples:
        st = s['structured']
        strat_dist[STRATEGY_NAMES[st['strategy']]] += 1
        action_dist[st['action_name']] += 1
        rewards.append(s['metadata']['reward'])

    print(f'\nStrategy: {strat_dist}')
    print(f'Actions: {dict(sorted(action_dist.items(), key=lambda x: -x[1]))}')
    print(f'Reward range: [{min(rewards):.2f}, {max(rewards):.2f}], avg={np.mean(rewards):.3f}')

    # Save
    base = f'simulator_data_{args.episodes}ep'
    save_jsonl(samples, os.path.join(args.output, base + '.jsonl'))

    print('\nDone.')
