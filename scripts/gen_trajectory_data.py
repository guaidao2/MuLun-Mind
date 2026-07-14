"""
Convert independent simulator samples to trajectory-level data.
Groups by episode -> creates mini-trajectories (3-5 consecutive steps)
-> each trajectory becomes a multi-turn conversation for RSSM training.

Usage:
    python scripts/gen_trajectory_data.py
"""
import os, sys, json, random
from collections import defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
sys.path.insert(0, _ROOT)

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

STATE_NAMES = [
    'threat_severity', 'threat_type_code', 'attack_surface',
    'lateral_movement_risk', 'data_exfil_risk', 'persistence_risk',
    'detection_level', 'attacker_sophistication',
    'critical_asset_count', 'patched_ratio', 'isolated_ratio',
    'compromised_ratio', 'alert_level', 'backup_status',
    'monitoring_coverage', 'incident_response_phase',
]


def state_to_situation(state_vec):
    """Brief state description for user message."""
    sev = state_vec[0]
    comp = state_vec[11]
    parts = []
    if comp > 0.3:
        parts.append(f'{comp:.0%}节点失陷')
    if sev > 0.6:
        parts.append('威胁等级高')
    elif sev < 0.3:
        parts.append('威胁等级较低')
    else:
        parts.append('威胁等级中等')
    return '，'.join(parts) if parts else '情况稳定'


def build_trajectory_conversation(steps):
    """
    Build a multi-turn conversation from N consecutive steps.
    Returns (conversations, per_step_targets)
    """
    conversations = [
        {'role': 'system', 'content': '你是一个专业的网络安全专家。'}
    ]
    targets = []

    for i, step in enumerate(steps):
        state_vec = step['structured']['state']
        strategy = step['structured']['strategy']
        action = step['structured']['action']
        value = step['structured']['value']
        reward = step['metadata']['reward']

        action_name, action_desc = ACTIONS[action]
        situation = state_to_situation(state_vec)

        if i == 0:
            user_msg = f'检测到安全事件：{situation}。请做出安全决策。'
        else:
            prev_action_name, _ = ACTIONS[steps[i-1]['structured']['action']]
            user_msg = f'上一步执行{prev_action_name}后（reward={reward:+.2f}），{situation}。请继续决策。'

        think = f'威胁评估：严重度{state_vec[0]:.0%}，失陷{state_vec[11]:.0%}。执行{action_name}：{action_desc}。'
        assistant_msg = f'<think>\n{think}\n</think>\n\n建议执行{action_name}：{action_desc}。预期遏制概率{value:.0%}。'

        conversations.append({'role': 'user', 'content': user_msg})
        conversations.append({'role': 'assistant', 'content': assistant_msg})
        targets.append({'strategy': strategy, 'action': action, 'value': value})

    return conversations, targets


def group_by_episode(input_path):
    """Load simulator data and group by episode."""
    episodes = defaultdict(list)
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ep = d['metadata']['episode']
            step = d['metadata']['step']
            episodes[ep].append((step, d))

    # Sort each episode by step
    for ep in episodes:
        episodes[ep].sort(key=lambda x: x[0])
        episodes[ep] = [d for _, d in episodes[ep]]

    return episodes


def create_trajectories(episodes, window=4, stride=2):
    """Slide a window over each episode to create mini-trajectories."""
    trajectories = []
    for ep_id, steps in episodes.items():
        if len(steps) < 2:
            continue
        for start in range(0, len(steps) - window + 1, stride):
            chunk = steps[start:start + window]
            convs, targets = build_trajectory_conversation(chunk)
            trajectories.append({
                'conversations': convs,
                'trajectory_steps': targets,
                'metadata': {'episode': ep_id, 'start_step': start, 'n_steps': len(chunk)},
            })
    return trajectories


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'Saved {len(data)} trajectories to {path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../dataset/simulator_data_200ep.jsonl')
    parser.add_argument('--output', default='../dataset/simulator_trajectories.jsonl')
    parser.add_argument('--window', type=int, default=4, help='Steps per trajectory')
    parser.add_argument('--stride', type=int, default=2, help='Slide stride')
    args = parser.parse_args()

    episodes = group_by_episode(args.input)
    print(f'Loaded {len(episodes)} episodes')

    trajectories = create_trajectories(episodes, args.window, args.stride)
    print(f'Example trajectory: {len(trajectories[0][\"conversations\"])} turns, '
          f'{len(trajectories[0][\"trajectory_steps\"])} decision steps')

    # Also convert synthetic data (each is a single step, wrap as 1-step trajectory)
    synthetic_trajs = []
    with open('../dataset/decision_chain_synthetic_5000.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            st = d['structured']
            synthetic_trajs.append({
                'conversations': d['conversations'],
                'trajectory_steps': [{'strategy': st['strategy'], 'action': st['action'], 'value': st['value']}],
                'metadata': {'from': 'synthetic'},
            })

    # Merge synthetic + simulator trajectories
    all_data = synthetic_trajs + trajectories
    random.shuffle(all_data)
    save_jsonl(all_data, args.output)
    print(f'Synthetic: {len(synthetic_trajs)} + Simulator trajectories: {len(trajectories)} = Total: {len(all_data)}')
