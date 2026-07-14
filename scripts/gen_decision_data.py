"""
Generate decision-chain training data for mulun model.

Two modes:
  1. simulator — runs Game-NN NetworkWorld to generate real game trajectories
  2. synthetic — standalone generation covering diverse security scenarios

Output: jsonl + parquet with conversations and structured labels.

Usage:
    python scripts/gen_decision_data.py --mode synthetic --count 5000
    python scripts/gen_decision_data.py --mode simulator --count 2000
"""
import os, sys, json, random, math, argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
sys.path.insert(0, _ROOT)

import numpy as np


# ═══════════════════════════════════════════════════════════
# Constants — match decision_head.py action/strategy spaces
# ═══════════════════════════════════════════════════════════

STRATEGIES = ['aggressive', 'balanced', 'defensive']
STRATEGY_DESC = {
    'aggressive': '主动反制：部署蜜罐、威胁狩猎、反向追踪',
    'balanced': '稳健应对：深度分析、评估风险、协调资源',
    'defensive': '防御优先：封禁、修补、隔离、恢复',
}

ACTIONS = [
    ('BLOCK_IP', '封禁攻击源IP', 'defensive'),
    ('PATCH_VULN', '修补漏洞', 'defensive'),
    ('ISOLATE_HOST', '隔离受影响主机', 'defensive'),
    ('RESTORE_BACKUP', '从备份恢复系统', 'defensive'),
    ('DEEP_SCAN', '深度扫描分析攻击范围', 'balanced'),
    ('HUNT_THREATS', '主动威胁狩猎', 'aggressive'),
    ('DEPLOY_HONEYPOT', '部署蜜罐诱捕', 'aggressive'),
    ('ESCALATE_INCIDENT', '升级事件至应急响应团队', 'balanced'),
]

STRATEGY_ACTION_MAP = {
    0: [5, 6, 2, 0],  # aggressive
    1: [4, 7, 1, 2],  # balanced
    2: [0, 1, 2, 3],  # defensive
}

STATE_DIM = 16
N_ACTIONS = 8
N_STRATEGIES = 3

# Security state dimension names
STATE_NAMES = [
    'threat_severity', 'threat_type_code', 'attack_surface',
    'lateral_movement_risk', 'data_exfil_risk', 'persistence_risk',
    'detection_level', 'attacker_sophistication',
    'critical_asset_count', 'patched_ratio', 'isolated_ratio',
    'compromised_ratio', 'alert_level', 'backup_status',
    'monitoring_coverage', 'incident_response_phase',
]


@dataclass
class SecurityState:
    """16-dim structured security state."""
    values: List[float] = field(default_factory=lambda: [0.0] * STATE_DIM)

    @property
    def threat_severity(self): return self.values[0]
    @property
    def compromised_ratio(self): return self.values[11]
    @property
    def lateral_movement_risk(self): return self.values[3]
    @property
    def alert_level(self): return self.values[12]
    @property
    def incident_response_phase(self): return self.values[15]

    def describe(self) -> str:
        """Generate human-readable state description."""
        parts = []
        sev = self.threat_severity
        if sev < 0.2:
            parts.append("威胁等级低")
        elif sev < 0.5:
            parts.append("威胁等级中等")
        elif sev < 0.8:
            parts.append("威胁等级高")
        else:
            parts.append("威胁等级严重")

        comp = self.compromised_ratio
        if comp < 0.1:
            parts.append("尚未发现资产被攻陷")
        elif comp < 0.3:
            parts.append(f"{comp:.0%}资产被攻陷")
        elif comp < 0.6:
            parts.append(f"{comp:.0%}资产被攻陷，情况严峻")
        else:
            parts.append(f"{comp:.0%}资产已失守，需紧急处置")

        lat = self.lateral_movement_risk
        if lat > 0.6:
            parts.append("横向移动风险极高，攻击者可能在扩散")

        det = self.alert_level
        if det < 0.2:
            parts.append("检测覆盖率低，可能遗漏告警")
        elif det > 0.7:
            parts.append("告警数量大，需过滤误报")

        phase = self.incident_response_phase
        phases = ['检测阶段', '分析阶段', '遏制阶段', '根除阶段', '恢复阶段']
        phase_name = phases[min(int(phase * 5), 4)]
        parts.append(f"当前处于{phase_name}")

        return "，".join(parts)

    def to_dict(self):
        return {STATE_NAMES[i]: self.values[i] for i in range(STATE_DIM)}


# ═══════════════════════════════════════════════════════════
# Scenario templates
# ═══════════════════════════════════════════════════════════

SCENARIOS = [
    # (state_template, user_text)
    # Each state_template is a dict of {dim_name: value_or_range}
    {
        'state': {
            'threat_severity': (0.1, 0.3),
            'threat_type_code': 0,
            'attack_surface': (0.2, 0.4),
            'lateral_movement_risk': (0.0, 0.2),
            'data_exfil_risk': (0.0, 0.1),
            'persistence_risk': (0.0, 0.1),
            'detection_level': (0.1, 0.3),
            'attacker_sophistication': (0.2, 0.4),
            'critical_asset_count': (0.0, 0.1),
            'patched_ratio': 0.3,
            'isolated_ratio': 0.0,
            'compromised_ratio': (0.0, 0.05),
            'alert_level': (0.1, 0.3),
            'backup_status': 0.8,
            'monitoring_coverage': 0.3,
            'incident_response_phase': 0.0,
        },
        'user_templates': [
            "Web服务器出现大量异常404请求，疑似扫描行为，请分析并给出处置建议",
            "防火墙告警：来自境外IP对办公网21端口进行暴力破解尝试",
            "IDS检测到针对Apache Struts的扫描流量，目前无明显成功迹象",
            "邮件网关拦截多封包含恶意附件的钓鱼邮件，收件人尚未点击",
        ],
        'preferred_strategies': [1, 2],  # balanced or defensive
    },
    {
        'state': {
            'threat_severity': (0.4, 0.6),
            'threat_type_code': (1, 3),
            'attack_surface': (0.4, 0.6),
            'lateral_movement_risk': (0.3, 0.5),
            'data_exfil_risk': (0.2, 0.4),
            'persistence_risk': (0.1, 0.3),
            'detection_level': 0.5,
            'attacker_sophistication': (0.4, 0.6),
            'critical_asset_count': (0.1, 0.3),
            'patched_ratio': 0.3,
            'isolated_ratio': 0.1,
            'compromised_ratio': (0.1, 0.25),
            'alert_level': (0.4, 0.6),
            'backup_status': 0.6,
            'monitoring_coverage': 0.5,
            'incident_response_phase': 0.25,
        },
        'user_templates': [
            "内网主机发现异常进程连接到外部C2服务器，疑似被控，已隔离该主机",
            "生产环境Web服务器被上传webshell，已经确认攻击者获得了低权限shell",
            "检测到内网一台服务器向多台主机发起SMB扫描，该主机权限可能已被窃取",
            "安全团队发现GitHub仓库泄露了AWS Access Key，已轮换但需评估影响范围",
        ],
        'preferred_strategies': [1],
    },
    {
        'state': {
            'threat_severity': (0.7, 0.9),
            'threat_type_code': (2, 5),
            'attack_surface': (0.6, 0.8),
            'lateral_movement_risk': (0.6, 0.8),
            'data_exfil_risk': (0.5, 0.7),
            'persistence_risk': (0.4, 0.6),
            'detection_level': 0.7,
            'attacker_sophistication': (0.6, 0.8),
            'critical_asset_count': (0.3, 0.5),
            'patched_ratio': 0.2,
            'isolated_ratio': 0.2,
            'compromised_ratio': (0.3, 0.5),
            'alert_level': (0.7, 0.9),
            'backup_status': 0.4,
            'monitoring_coverage': 0.6,
            'incident_response_phase': 0.5,
        },
        'user_templates': [
            "核心数据库服务器被勒索软件加密，攻击者留言索要比特币，数据库包含客户个人信息",
            "域控制器出现异常Kerberos请求，Golden Ticket攻击痕迹明显，攻击者可能已获取域管理员权限",
            "多台服务器被挖矿程序感染，CPU占用100%，攻击者通过SSH暴力破解进入",
            "ERP系统被植入后门，攻击者通过0day漏洞取得控制权，正在横向移动到财务系统",
        ],
        'preferred_strategies': [0, 1],
    },
    {
        'state': {
            'threat_severity': (0.8, 1.0),
            'threat_type_code': (3, 6),
            'attack_surface': 0.8,
            'lateral_movement_risk': (0.7, 0.9),
            'data_exfil_risk': (0.7, 0.9),
            'persistence_risk': (0.6, 0.8),
            'detection_level': 0.8,
            'attacker_sophistication': (0.8, 1.0),
            'critical_asset_count': (0.5, 0.8),
            'patched_ratio': 0.1,
            'isolated_ratio': 0.3,
            'compromised_ratio': (0.5, 0.7),
            'alert_level': (0.8, 1.0),
            'backup_status': 0.2,
            'monitoring_coverage': 0.7,
            'incident_response_phase': 0.5,
        },
        'user_templates': [
            "APT攻击：攻击者在内网活动已超过3个月，大量敏感数据被外传，C2通道隐蔽",
            "勒索软件在全网爆发，所有文件服务器被加密，备份系统也被破坏，业务完全瘫痪",
            "供应链攻击：第三方软件更新包被植入后门，已分发到所有客户环境，需紧急响应",
            "国家级攻击者利用多个0day组合攻击，已渗透到核心业务系统，无法确定失陷范围",
        ],
        'preferred_strategies': [0],
    },
    {
        'state': {
            'threat_severity': (0.2, 0.4),
            'threat_type_code': 0,
            'attack_surface': (0.2, 0.3),
            'lateral_movement_risk': (0.0, 0.1),
            'data_exfil_risk': (0.0, 0.1),
            'persistence_risk': (0.0, 0.1),
            'detection_level': 0.6,
            'attacker_sophistication': (0.2, 0.4),
            'critical_asset_count': 0.0,
            'patched_ratio': 0.6,
            'isolated_ratio': 0.1,
            'compromised_ratio': 0.0,
            'alert_level': (0.3, 0.5),
            'backup_status': 0.9,
            'monitoring_coverage': 0.6,
            'incident_response_phase': 0.75,
        },
        'user_templates': [
            "已成功隔离失陷主机并清除后门，请给出后续恢复和加固建议",
            "漏洞修补完成，攻击源已封禁，系统恢复正常运行，请输出安全总结报告",
            "应急响应结束，需要对整网进行安全加固，防止类似事件再次发生",
        ],
        'preferred_strategies': [2],
    },
]


def random_state(scenario: dict) -> SecurityState:
    """Generate a random state vector from a scenario template."""
    vals = []
    t = scenario['state']
    for i in range(STATE_DIM):
        name = STATE_NAMES[i]
        if name in t:
            v = t[name]
            if isinstance(v, tuple):
                vals.append(round(random.uniform(v[0], v[1]), 3))
            else:
                vals.append(float(v))
        else:
            vals.append(round(random.random() * 0.3, 3))
    return SecurityState(vals)


def state_to_nl(state: SecurityState) -> str:
    """Convert state to a one-paragraph security situation summary."""
    sev = state.threat_severity
    comp_ratio = state.compromised_ratio
    lat = state.lateral_movement_risk

    sev_label = '低' if sev < 0.3 else ('中' if sev < 0.6 else ('高' if sev < 0.8 else '严重'))

    parts = [f"当前威胁等级为{sev_label}"]
    if comp_ratio > 0:
        parts.append(f"约{comp_ratio:.0%}的资产已经失陷")
    if lat > 0.5:
        parts.append("攻击者正在尝试横向移动")

    phase_idx = min(int(state.incident_response_phase * 5), 4)
    phase_names = ['检测与分析', '分析评估', '遏制处置', '根除清理', '恢复加固']
    parts.append(f"处于{phase_names[phase_idx]}阶段")

    return '，'.join(parts)


# ═══════════════════════════════════════════════════════════
# Decision chain generation
# ═══════════════════════════════════════════════════════════

def pick_action(strategy: int) -> Tuple[int, str, str, float]:
    """Pick an action for the given strategy with realistic value."""
    action_ids = STRATEGY_ACTION_MAP[strategy]
    action_id = random.choice(action_ids)
    action_name, action_desc, action_type = ACTIONS[action_id]

    # Expected containment probability based on strategy and severity
    base_prob = {
        0: random.uniform(0.55, 0.85),  # aggressive: high variance
        1: random.uniform(0.60, 0.80),  # balanced: steady
        2: random.uniform(0.70, 0.90),  # defensive: most reliable
    }[strategy]

    return action_id, action_name, action_desc, round(base_prob, 3)


def generate_think_block(state: SecurityState, strategy: int, strategy_name: str,
                         action_name: str, action_desc: str, prob: float) -> str:
    """Generate the <think> block content from a decision."""
    sev = state.threat_severity
    comp = state.compromised_ratio
    lat = state.lateral_movement_risk

    reasoning_parts = []

    # Strategy reasoning
    if strategy == 0:
        reasoning_parts.append(f"威胁严重度{sev:.0%}，失陷比例{comp:.0%}，决定采取主动反制策略")
    elif strategy == 1:
        reasoning_parts.append(f"情况可控但需警惕（严重度{sev:.0%}），采取稳健策略进一步分析")
    else:
        reasoning_parts.append(f"优先止损防御，先控制局面防止进一步恶化")

    if lat > 0.5:
        reasoning_parts.append("横向移动风险高，必须立即阻断")

    # Action reasoning
    reasoning_parts.append(f"执行{action_name}：{action_desc}")

    # Reflection
    if prob > 0.8:
        reasoning_parts.append(f"预期成功率{prob:.0%}，方案可行")
    elif prob > 0.5:
        reasoning_parts.append(f"预期成功率{prob:.0%}，需监测效果")
    else:
        reasoning_parts.append(f"预期成功率仅{prob:.0%}，可能需要备选方案")

    return '\n'.join(reasoning_parts)


def generate_decision_conversation(
    scenario: dict,
    state: SecurityState,
    strategy: int,
    action_id: int,
    action_name: str,
    action_desc: str,
    prob: float,
) -> dict:
    """Generate a full conversation with decision chain."""
    strategy_name = STRATEGIES[strategy]
    user_text = random.choice(scenario['user_templates'])

    think = generate_think_block(state, strategy, strategy_name,
                                  action_name, action_desc, prob)
    recommendation = (
        f"建议执行{action_name}：{action_desc}。"
        f"预期遏制概率{prob:.0%}。"
    )

    assistant_text = f"<think>\n{think}\n</think>\n\n{recommendation}"

    conversations = [
        {'role': 'system', 'content': '你是一个专业的网络安全专家，精通渗透测试、漏洞分析和应急响应。'},
        {'role': 'user', 'content': user_text},
        {'role': 'assistant', 'content': assistant_text},
    ]

    # Structured labels for decision head training
    structured = {
        'state': state.values,
        'strategy': strategy,
        'strategy_name': strategy_name,
        'action': action_id,
        'action_name': action_name,
        'value': prob,
        'state_dict': state.to_dict(),
    }

    return {
        'conversations': conversations,
        'structured': structured,
    }


# ═══════════════════════════════════════════════════════════
# Simulator-based generation
# ═══════════════════════════════════════════════════════════

def generate_with_simulator(count: int) -> List[dict]:
    """Use NetworkWorld to generate real game trajectories."""
    from network_world.environment import NetworkWorld, RandomAttacker, RandomDefender
    from config import Config

    cfg = Config()
    env = NetworkWorld(cfg.network)
    attacker = RandomAttacker()
    defender = RandomDefender()
    samples = []
    n_per_episode = max(1, count // 50)

    for ep in range(max(1, count // n_per_episode)):
        obs = env.reset()
        done = False
        episode_states = []

        while not done and len(episode_states) < n_per_episode:
            if env.is_attacker_turn:
                action, node, node2 = attacker.act(obs, env.get_available_nodes())
                next_obs, reward, done, info = env.step(action, node, node2)
            else:
                action, node, _ = defender.act(obs, env.get_available_nodes())
                next_obs, reward, done, info = env.step(action, node)

            # Extract global features as state proxy
            obs_flat = obs.flatten()
            defense_actions = [4, 5, 6, 7]
            if action in defense_actions:
                comp_ratio = float(info.get('compromised_ratio', 0.0))
                state_vec = [
                    min(1.0, abs(reward) * 2),        # threat_severity
                    float(action - 4) / 4,             # threat_type
                    min(1.0, env.node_state[:, 5].mean() * 2),  # attack_surface
                    min(1.0, env.node_state[:, 1].mean() * 3),  # lateral
                    min(1.0, comp_ratio * 2),           # data_exfil
                    min(1.0, env.node_state[:, 7].mean() * 3),  # persistence
                    float(env.intrusion_detected),       # detection
                    min(1.0, env.time_step / env.max_steps),  # sophistication
                    min(1.0, env.node_state[:, 0].mean()),    # critical_asset
                    min(1.0, env.node_state[:, 3].mean()),    # patched
                    min(1.0, env.node_state[:, 6].mean()),    # isolated
                    comp_ratio,                                # compromised
                    min(1.0, env.node_state[:, 5].mean()),    # alert
                    max(0.2, 1.0 - comp_ratio * 2),           # backup
                    0.5,                                       # monitoring
                    min(1.0, env.time_step / env.max_steps),   # phase
                ]
                state = SecurityState([round(v, 3) for v in state_vec])

                # Map defender action to our action space
                action_idx = action - 4
                action_name, action_desc, _ = ACTIONS[action_idx]

                # Determine strategy from action
                for s, acts in STRATEGY_ACTION_MAP.items():
                    if action_idx in acts:
                        strategy = s
                        break
                else:
                    strategy = 1

                prob = round(min(0.95, max(0.1, 0.5 + reward * 0.3)), 3)

                sample = generate_decision_conversation(
                    SCENARIOS[1], state, strategy, action_idx,
                    action_name, action_desc, prob,
                )
                episode_states.append(sample)

            obs = next_obs

        samples.extend(episode_states)

    # Trim to exact count
    random.shuffle(samples)
    return samples[:count]


# ═══════════════════════════════════════════════════════════
# Synthetic generation (standalone)
# ═══════════════════════════════════════════════════════════

def generate_synthetic(count: int) -> List[dict]:
    """Generate decision-chain conversations without simulator."""
    samples = []

    for _ in range(count):
        # Pick a scenario
        scenario = random.choice(SCENARIOS)
        state = random_state(scenario)

        # Pick strategy (preferred or random for diversity)
        pref = scenario.get('preferred_strategies', [0, 1, 2])
        if random.random() < 0.7:
            strategy = random.choice(pref)
        else:
            strategy = random.choice([s for s in [0, 1, 2] if s not in pref] or pref)

        # Pick action with some randomness (20% suboptimal for training diversity)
        if random.random() < 0.2:
            # Suboptimal action (for training the model to distinguish)
            action_id = random.choice(
                [a for a in range(N_ACTIONS) if a not in STRATEGY_ACTION_MAP[strategy]]
            )
            action_name, action_desc, _ = ACTIONS[action_id]
            prob = round(random.uniform(0.2, 0.5), 3)
        else:
            action_id, action_name, action_desc, prob = pick_action(strategy)

        sample = generate_decision_conversation(
            scenario, state, strategy, action_id,
            action_name, action_desc, prob,
        )
        samples.append(sample)

    return samples


# ═══════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════

def save_jsonl(samples: List[dict], path: str):
    """Save as jsonl (one JSON object per line)."""
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Saved {len(samples)} samples to {path}")


def save_parquet(samples: List[dict], path: str):
    """Save as parquet (compatible with OmniDataset)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    conversations = []
    question_audios = []
    answer_audios = []
    image_bytes = []
    ref_audios = []
    spk_emb = []
    structures = []

    for s in samples:
        conversations.append(json.dumps(s['conversations'], ensure_ascii=False))
        question_audios.append([])
        answer_audios.append([])
        image_bytes.append([])
        ref_audios.append([])
        spk_emb.append([])
        structures.append(s['structured'])

    schema = pa.schema([
        ('conversations', pa.large_string()),
        ('question_audios', pa.list_(pa.string())),
        ('answer_audios', pa.list_(pa.string())),
        ('image_bytes', pa.list_(pa.binary())),
        ('ref_audios', pa.list_(pa.string())),
        ('spk_emb', pa.list_(pa.float32())),
    ])

    table = pa.table({
        'conversations': pa.array(conversations, type=pa.large_string()),
        'question_audios': pa.array(question_audios),
        'answer_audios': pa.array(answer_audios),
        'image_bytes': pa.array(image_bytes),
        'ref_audios': pa.array(ref_audios),
        'spk_emb': pa.array(spk_emb),
    }, schema=schema)

    pq.write_table(table, path, compression='snappy')

    # Save structured labels separately
    struct_path = path.replace('.parquet', '_labels.json')
    with open(struct_path, 'w', encoding='utf-8') as f:
        json.dump(structures, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(path) / 1024
    print(f"Saved {len(samples)} samples to {path} ({size_kb:.0f} KB)")
    print(f"Labels saved to {struct_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate decision-chain training data")
    parser.add_argument('--mode', choices=['synthetic', 'simulator'], default='synthetic',
                        help="Generation mode")
    parser.add_argument('--count', type=int, default=5000,
                        help="Number of samples to generate")
    parser.add_argument('--output', type=str, default=os.path.join(_ROOT, 'dataset'),
                        help="Output directory")
    parser.add_argument('--format', choices=['jsonl', 'parquet', 'both'], default='both',
                        help="Output format")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print(f"Generating {args.count} samples ({args.mode} mode)...")

    if args.mode == 'simulator':
        try:
            sys.path.insert(0, os.path.join(_ROOT, '..', 'Game-nn-O', 'game_nn_simulator'))
            samples = generate_with_simulator(args.count)
        except ImportError as e:
            print(f"Simulator not available ({e}), falling back to synthetic mode")
            samples = generate_synthetic(args.count)
    else:
        samples = generate_synthetic(args.count)

    # Stats
    strat_dist = {s: 0 for s in STRATEGIES}
    action_dist = {a[0]: 0 for a in ACTIONS}
    for s in samples:
        st = s['structured']
        strat_dist[STRATEGIES[st['strategy']]] += 1
        action_dist[st['action_name']] += 1

    print(f"\nStrategy distribution: {strat_dist}")
    print(f"Action distribution: {dict(sorted(action_dist.items(), key=lambda x: -x[1]))}")

    # Save
    base_name = f"decision_chain_{args.mode}_{args.count}"
    if args.format in ('jsonl', 'both'):
        save_jsonl(samples, os.path.join(args.output, base_name + '.jsonl'))
    if args.format in ('parquet', 'both'):
        save_parquet(samples, os.path.join(args.output, base_name + '.parquet'))

    print("\nDone.")
