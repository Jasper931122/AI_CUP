import math
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

required_cols = [
    'txn_total_count','txn_total_amount','txn_amount_avg','txn_amount_std',
    'from_total_count','from_txn_ratio','from_total_amount','from_amount_ratio',
    'from_avg_amount','from_amount_std',
    'from_amt_1_1000_count','from_amt_1001_5000_count','from_amt_5001_30000_count',
    'from_amt_30001_50000_count','from_amt_50001_plus_count',
    'from_amt_1_1000_ratio','from_amt_1001_5000_ratio','from_amt_5001_30000_ratio',
    'from_amt_30001_50000_ratio','from_amt_50001_plus_ratio',
    'to_total_count','to_txn_ratio','to_total_amount','to_amount_ratio',
    'to_avg_amount','to_amount_std',
    'to_amt_1_1000_count','to_amt_1001_5000_count','to_amt_5001_30000_count',
    'to_amt_30001_50000_count','to_amt_50001_plus_count',
    'to_amt_1_1000_ratio','to_amt_1001_5000_ratio','to_amt_5001_30000_ratio',
    'to_amt_30001_50000_ratio','to_amt_50001_plus_ratio',
    'txn_duration_days','txn_active_days','txn_active_ratio','txn_avg_per_day',
    'txn_00_06_count','txn_06_12_count','txn_12_18_count','txn_18_24_count',
    'txn_days_over_5_count','from_days_over_3_count','to_days_over_3_count',
    'acct_unique_count','acct_unique_to_count','acct_unique_from_count',
    'acct_txn_with_alert_count','acct_from_to_alert_count','acct_to_from_alert_count',
    'acct_from_to_alert_3','acct_to_from_alert_3',
    'self_txn_count','foreign_currency_count',
    'channel_99_count','channel_atm_count','channel_unk_count',
    'channel_mobile_count','channel_online_count','channel_counter_count',
    'txn_ratio_last_1d','txn_ratio_last_3d','txn_ratio_last_7d',
    'from_ratio_last_1d','from_ratio_last_3d','from_ratio_last_7d',
    'to_ratio_last_1d','to_ratio_last_3d','to_ratio_last_7d',
    'txn_amount_ratio_last_1d','txn_amount_ratio_last_3d','txn_amount_ratio_last_7d',
    'from_amount_ratio_last_1d','from_amount_ratio_last_3d','from_amount_ratio_last_7d',
    'to_amount_ratio_last_1d','to_amount_ratio_last_3d','to_amount_ratio_last_7d',
    'acct_overlap_ratio_last_1d','acct_overlap_ratio_last_3d','acct_overlap_ratio_last_7d',
    'from_overlap_ratio_last_1d','from_overlap_ratio_last_3d','from_overlap_ratio_last_7d',
    'to_overlap_ratio_last_1d','to_overlap_ratio_last_3d','to_overlap_ratio_last_7d',
    'from_to_amount_ratio_last_1d','from_to_amount_ratio_last_3d','from_to_amount_ratio_last_7d'
]

# ===== 可調參數 =====
SMALL_BINS = [(1,1000),(1001,5000),(5001,30000),(30001,50000),(50001, np.inf)]
TOP_DAY_RATIO   = 0.2
THRESH_TXN_PER_DAY  = 5
THRESH_SEND_PER_DAY = 3
THRESH_RECV_PER_DAY = 3

def _safe_div(n, d):
    # 0/0 -> 0；x/0 -> 0（避免 inf）
    return 0.0 if d == 0 else n / d

def _partners_set(sub: pd.DataFrame, kind: str):
    # kind: 'all' / 'send' / 'recv'
    if kind == 'send':
        g = sub[sub['id'] == sub['from_acct']].groupby('id')['to_acct'].agg(lambda x: set(x))
    elif kind == 'recv':
        g = sub[sub['id'] == sub['to_acct']].groupby('id')['from_acct'].agg(lambda x: set(x))
    else:
        s = sub[sub['id'] == sub['from_acct']].groupby('id')['to_acct'].agg(lambda x: set(x))
        r = sub[sub['id'] == sub['to_acct']].groupby('id')['from_acct'].agg(lambda x: set(x))
        g = s.combine(r, lambda a, b: (a if isinstance(a,set) else set()) |
                                   (b if isinstance(b,set) else set()))
    return g

def add_recent_window_features(df: pd.DataFrame,
                               feat: pd.DataFrame,
                               windows=(1,3,7)) -> pd.DataFrame:
    """
    依每個 id 的最後交易日為基準，計算最近 N 日的：
      - 交易/匯款/收款 次數比例
      - 交易/匯款/收款 金額比例
      - 匯款金額 / 收款金額 比例
      - 交易對手重疊比例（整體/匯款/收款）
    會直接把欄位 merge 進 feat 後回傳。
    """
    df = df.copy()
    df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')

    # 相對於各 id 的最後交易日的「距離天數」
    last_day = df.groupby('id')['txn_date'].transform('max')
    df['dfl'] = (last_day - df['txn_date']).dt.days  # 0 表示最後一天

    send_mask = (df['id'] == df['from_acct'])
    recv_mask = (df['id'] == df['to_acct'])

    # 全期間的唯一對手（做重疊比例的分母）
    total_all  = _partners_set(df, 'all')
    total_send = _partners_set(df, 'send')
    total_recv = _partners_set(df, 'recv')

    # 便利：把總分母拿到 feat 的索引順序
    idx = feat['id']
    denom_all  = total_all.reindex(idx).apply(lambda s: len(s) if isinstance(s,set) else 0).to_numpy()
    denom_send = total_send.reindex(idx).apply(lambda s: len(s) if isinstance(s,set) else 0).to_numpy()
    denom_recv = total_recv.reindex(idx).apply(lambda s: len(s) if isinstance(s,set) else 0).to_numpy()

    for N in windows:
        # 最近 N 日（含最後日）：dfl ∈ [0, N-1]
        recent = df[df['dfl'] <= N-1]
        prior  = df[df['dfl'] >= N]     # N 日之前

        # --- 1) 次數比例：整體 / 匯款 / 收款 ---
        # 整體
        cnt_recent = recent.groupby('id').size()
        feat[f'txn_ratio_last_{N}d'] = np.where(
            feat['txn_total_count']>0,
            cnt_recent.reindex(idx).fillna(0).to_numpy() / feat['txn_total_count'].to_numpy(),
            0
        )
        # 匯款
        cnt_recent_send = recent[send_mask].groupby('id').size()
        feat[f'from_ratio_last_{N}d'] = np.where(
            feat['from_total_count']>0,
            cnt_recent_send.reindex(idx).fillna(0).to_numpy() / np.where(feat['from_total_count']>0, feat['from_total_count'], 1),
            0
        )
        # 收款
        cnt_recent_recv = recent[recv_mask].groupby('id').size()
        feat[f'to_ratio_last_{N}d'] = np.where(
            feat['to_total_count']>0,
            cnt_recent_recv.reindex(idx).fillna(0).to_numpy() / np.where(feat['to_total_count']>0, feat['to_total_count'], 1),
            0
        )

        # --- 2) 金額比例：整體 / 匯款 / 收款 ---
        amt_recent = recent.groupby('id')['txn_amt'].sum()
        feat[f'txn_amount_ratio_last_{N}d'] = np.where(
            feat['txn_total_amount']>0,
            amt_recent.reindex(idx).fillna(0).to_numpy() / feat['txn_total_amount'].to_numpy(),
            0
        )

        amt_recent_send = recent[send_mask].groupby('id')['txn_amt'].sum()
        feat[f'from_amount_ratio_last_{N}d'] = np.where(
            feat['from_total_amount']>0,
            amt_recent_send.reindex(idx).fillna(0).to_numpy() / np.where(feat['from_total_amount']>0, feat['from_total_amount'], 1),
            0
        )

        amt_recent_recv = recent[recv_mask].groupby('id')['txn_amt'].sum()
        feat[f'to_amount_ratio_last_{N}d'] = np.where(
            feat['to_total_amount']>0,
            amt_recent_recv.reindex(idx).fillna(0).to_numpy() / np.where(feat['to_total_amount']>0, feat['to_total_amount'], 1),
            0
        )

        # --- 3) 匯款金額 / 收款金額（最近 N 日）
        s = amt_recent_send.reindex(idx).fillna(0).to_numpy()
        r = amt_recent_recv.reindex(idx).fillna(0).to_numpy()
        feat[f'from_to_amount_ratio_last_{N}d'] = np.where(r>0, s/r, 0.0)

        # --- 4) 交易對手重疊比例（最近 N 日 vs N 日前）---
        # recent/prior 的對手集合
        r_all  = _partners_set(recent, 'all').reindex(idx).apply(lambda v: v if isinstance(v,set) else set())
        p_all  = _partners_set(prior,  'all').reindex(idx).apply(lambda v: v if isinstance(v,set) else set())
        r_send = _partners_set(recent, 'send').reindex(idx).apply(lambda v: v if isinstance(v,set) else set())
        p_send = _partners_set(prior,  'send').reindex(idx).apply(lambda v: v if isinstance(v,set) else set())
        r_recv = _partners_set(recent, 'recv').reindex(idx).apply(lambda v: v if isinstance(v,set) else set())
        p_recv = _partners_set(prior,  'recv').reindex(idx).apply(lambda v: v if isinstance(v,set) else set())

        inter_all  = np.array([len(a & b) for a,b in zip(r_all,  p_all )], dtype=float)
        inter_send = np.array([len(a & b) for a,b in zip(r_send, p_send)], dtype=float)
        inter_recv = np.array([len(a & b) for a,b in zip(r_recv, p_recv)], dtype=float)

        # 分母使用「全期間的全部/匯款/收款對手數」
        feat[f'acct_overlap_ratio_last_{N}d'] = np.where(denom_all > 0,  inter_all /  denom_all,  0.0)
        feat[f'from_overlap_ratio_last_{N}d'] = np.where(denom_send > 0, inter_send / denom_send, 0.0)
        feat[f'to_overlap_ratio_last_{N}d']   = np.where(denom_recv > 0, inter_recv / denom_recv, 0.0)

    return feat

def _time_bucket(hour: int) -> str:
    if hour < 6:   return '00_06'
    if hour < 12:  return '06_12'
    if hour < 18:  return '12_18'
    return '18_24'


def _amount_bin_counts(s: pd.Series, bins):
    counts = []
    for lo, hi in bins:
        if np.isinf(hi):
            counts.append(((s >= lo)).sum())
        else:
            counts.append(((s >= lo) & (s <= hi)).sum())
    return counts


def _amount_bin_ratios(counts, denom):
    if denom <= 0:
        return [0.0]*len(counts)
    return [c/denom for c in counts]


def _build_graphs(df: pd.DataFrame):
    out_adj, in_adj = defaultdict(set), defaultdict(set)
    for f, t in zip(df['from_acct'], df['to_acct']):
        out_adj[f].add(t)
        in_adj[t].add(f)
    return out_adj, in_adj


def _bfs_has_alert(start, adj, alert_ids, max_depth=3):
    if start not in adj:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        u, d = q.popleft()
        if d == max_depth:
            continue
        for v in adj.get(u, ()):
            if v in alert_ids:
                return 1
            if v not in seen:
                seen.add(v)
                q.append((v, d+1))
    return 0


def build_features(
    mode: str,                         # 'alert' / 'normal' / 'predict'
    input_dir: str = ".",
    output_dir: str = ".",
    alert_ids: set | None = None,      # 警示帳戶集合（optional）
    compute_three_hop: bool = False,
):
    """根據模式提取特徵值並輸出 csv（不處理 event_date）"""

    mode = mode.lower()
    assert mode in {"alert","normal","predict"}, "mode 必須是 'alert' / 'normal' / 'predict'"

    in_path  = Path(input_dir)  / f"{mode}_preprocessing.csv"
    out_path = Path(output_dir) / f"{mode}_features.csv"

    df = pd.read_csv(in_path)
    df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')

    # txn_time → hour
    hh = df['txn_time'].astype(str).str.extract(r'(^\d{1,2})')[0].astype(float)
    df['txn_hour'] = hh.fillna(0).astype(int).clip(0, 23)
    df['time_bucket'] = df['txn_hour'].apply(_time_bucket)

    send_mask = df['id'] == df['from_acct']
    recv_mask = df['id'] == df['to_acct']

    # ===== 整體 =====
    grp_all = df.groupby('id', as_index=False).agg(
        txn_total_count=('txn_amt', 'count'),
        txn_total_amount=('txn_amt', 'sum'),
        txn_amount_avg=('txn_amt', 'mean'),
        txn_amount_std=('txn_amt', 'std')
    )

    # optional: label
    if 'label' in df.columns:
        grp_all = grp_all.merge(df.groupby('id')['label'].max().reset_index(), on='id', how='left')

    # ===== 匯款 =====
    send = df[send_mask]
    grp_send = send.groupby('id', as_index=False).agg(
        from_total_count=('txn_amt', 'count'),
        from_total_amount=('txn_amt', 'sum'),
        from_avg_amount=('txn_amt', 'mean'),
        from_amount_std=('txn_amt', 'std')
    )

    def safe_bin_counts(s):
        if s is None or len(s) == 0 or s.isna().all():
            return [0, 0, 0, 0, 0]
        return _amount_bin_counts(s, SMALL_BINS)

    send_bins_counts = (
        send.groupby('id')['txn_amt']
        .apply(safe_bin_counts)
        .reindex(grp_all['id'], fill_value=[0,0,0,0,0])
    )

    send_bin_cols = [
        'from_amt_1_1000_count',
        'from_amt_1001_5000_count',
        'from_amt_5001_30000_count',
        'from_amt_30001_50000_count',
        'from_amt_50001_plus_count',
    ]
    send_bins_df = pd.DataFrame(send_bins_counts.tolist(), index=grp_all['id'], columns=send_bin_cols).reset_index().fillna(0)

    # ===== 收款 =====
    recv = df[recv_mask]
    grp_recv = recv.groupby('id', as_index=False).agg(
        to_total_count=('txn_amt', 'count'),
        to_total_amount=('txn_amt', 'sum'),
        to_avg_amount=('txn_amt', 'mean'),
        to_amount_std=('txn_amt', 'std')
    )

    recv_bins_counts = (
        recv.groupby('id')['txn_amt']
        .apply(safe_bin_counts)
        .reindex(grp_all['id'], fill_value=[0,0,0,0,0])
    )

    recv_bin_cols = [
        'to_amt_1_1000_count',
        'to_amt_1001_5000_count',
        'to_amt_5001_30000_count',
        'to_amt_30001_50000_count',
        'to_amt_50001_plus_count',
    ]
    recv_bins_df = pd.DataFrame(recv_bins_counts.tolist(), index=grp_all['id'], columns=recv_bin_cols).reset_index().fillna(0)

    # ===== 合併並計比例 =====
    feat = (
        grp_all.merge(grp_send, on='id', how='left')
        .merge(send_bins_df, on='id', how='left')
        .merge(grp_recv, on='id', how='left')
        .merge(recv_bins_df, on='id', how='left')
    ).fillna(0)

    feat['from_txn_ratio'] = np.where(
        feat['txn_total_count'] > 0, feat['from_total_count'] / feat['txn_total_count'], 0
    )
    feat['from_amount_ratio'] = np.where(
        feat['txn_total_amount'] > 0, feat['from_total_amount'] / feat['txn_total_amount'], 0
    )
    feat['to_txn_ratio'] = np.where(
        feat['txn_total_count'] > 0, feat['to_total_count'] / feat['txn_total_count'], 0
    )
    feat['to_amount_ratio'] = np.where(
        feat['txn_total_amount'] > 0, feat['to_total_amount'] / feat['txn_total_amount'], 0
    )

    # ===== 時間特徵 =====
    life = df.groupby('id').agg(
        first_day=('txn_date', 'min'),
        last_day=('txn_date', 'max'),
        txn_active_days=('txn_date', lambda x: x.nunique())
    ).reset_index()
    life['txn_duration_days'] = (life['last_day'] - life['first_day']).dt.days + 1
    total_cnt = df.groupby('id')['txn_amt'].count().rename('tmp_total_cnt').reset_index()
    life = life.merge(total_cnt, on='id', how='left')
    life['txn_active_ratio'] = np.where(
        life['txn_duration_days'] > 0, life['txn_active_days'] / life['txn_duration_days'], 0
    )
    life['txn_avg_per_day'] = np.where(
        life['txn_active_days'] > 0, life['tmp_total_cnt'] / life['txn_active_days'], 0
    )
    life = life.drop(columns=['tmp_total_cnt'])
    feat = feat.merge(life, on='id', how='left')

    # ===== 時段分佈 =====
    time_counts = df.groupby(['id', 'time_bucket']).size().unstack(fill_value=0)
    for b in ['00_06', '06_12', '12_18', '18_24']:
        if b not in time_counts.columns:
            time_counts[b] = 0
    time_counts = (
        time_counts[['00_06', '06_12', '12_18', '18_24']]
        .reset_index()
        .rename(
            columns={
                '00_06': 'txn_00_06_count',
                '06_12': 'txn_06_12_count',
                '12_18': 'txn_12_18_count',
                '18_24': 'txn_18_24_count',
            }
        )
    )
    feat = feat.merge(time_counts, on='id', how='left').fillna(0)

    # ===== 唯一往來帳戶數 =====
    send_unique = df[send_mask].groupby('id')['to_acct'].nunique().rename('acct_unique_to_count')
    recv_unique = df[recv_mask].groupby('id')['from_acct'].nunique().rename('acct_unique_from_count')
    feat = feat.merge(send_unique.reset_index(), on='id', how='left')
    feat = feat.merge(recv_unique.reset_index(), on='id', how='left')
    feat[['acct_unique_to_count', 'acct_unique_from_count']] = feat[
        ['acct_unique_to_count', 'acct_unique_from_count']
    ].fillna(0)
    feat['acct_unique_count'] = feat['acct_unique_to_count'] + feat['acct_unique_from_count']

    # ===== 類別統計 =====
    feat = feat.merge(
        (df['is_self_txn'].eq(2)).groupby(df['id']).sum().rename('self_txn_count').reset_index(),
        on='id',
        how='left',
    ).fillna({'self_txn_count': 0})

    feat = feat.merge(
        (df['currency_type'].eq(1)).groupby(df['id']).sum().rename('foreign_currency_count').reset_index(),
        on='id',
        how='left',
    ).fillna({'foreign_currency_count': 0})

    def _count_channel(val):
        return (df['channel_type'].eq(val)).groupby(df['id']).sum()

    ch_map = {
        'channel_unk_count': 0,
        'channel_atm_count': 1,
        'channel_counter_count': 2,
        'channel_mobile_count': 3,
        'channel_online_count': 4,
        'channel_99_count': 99,
    }
    for col, code in ch_map.items():
        feat = feat.merge(_count_channel(code).rename(col).reset_index(), on='id', how='left').fillna({col: 0})

    # ===== 與警示帳戶互動（可選） =====
    if alert_ids is None:
        alert_ids = set()

    mask_to_alert = send_mask & df['to_acct'].isin(alert_ids)
    mask_from_alert = recv_mask & df['from_acct'].isin(alert_ids)

    acct_txn_with_alert = (mask_to_alert | mask_from_alert).groupby(df['id']).sum().rename('acct_txn_with_alert_count')
    acct_from_to_alert = mask_to_alert.groupby(df['id']).sum().rename('acct_from_to_alert_count')
    acct_to_from_alert = mask_from_alert.groupby(df['id']).sum().rename('acct_to_from_alert_count')

    for s in [acct_txn_with_alert, acct_from_to_alert, acct_to_from_alert]:
        feat = feat.merge(s.reset_index(), on='id', how='left').fillna(0)

    # ===== 三層上下游旗標（可選） =====
    if compute_three_hop and len(alert_ids) > 0:
        out_adj, in_adj = _build_graphs(df)
        feat['acct_from_to_alert_3'] = [
            _bfs_has_alert(i, out_adj, alert_ids, 3) for i in feat['id']
        ]
        feat['acct_to_from_alert_3'] = [
            _bfs_has_alert(i, in_adj, alert_ids, 3) for i in feat['id']
        ]
    else:
        feat['acct_from_to_alert_3'] = 0
        feat['acct_to_from_alert_3'] = 0

    # ===== 集中度（前20%天） =====
    def _concentration_ratio(sub):
        arr = sub['cnt'].values
        tot = arr.sum()
        if tot == 0:
            return 0.0
        k = max(1, math.ceil(len(arr) * TOP_DAY_RATIO))
        return np.sort(arr)[-k:].sum() / tot

    daily = df.groupby(['id', 'txn_date']).size().rename('cnt').reset_index()
    conc = daily.groupby('id').apply(_concentration_ratio).rename('txn_concentration_top20').reset_index()
    feat = feat.merge(conc, on='id', how='left').fillna({'txn_concentration_top20': 0.0})

    # …前面為你原本的特徵計算…

    # === 加上最近 1/3/7 天特徵 ===
    feat = add_recent_window_features(df, feat, windows=(1,3,7))

    # === 輸出 ===
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')


    # ===== 輸出 =====
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ {mode} features saved → {out_path} | rows={len(feat)}")



alert_ids = set(pd.read_csv("alert_preprocessing.csv")['id'])  # 若有 id 欄位
build_features("alert", alert_ids=alert_ids, compute_three_hop=True)
# build_features("normal", alert_ids=alert_ids, compute_three_hop=True)
# build_features("predict", alert_ids=alert_ids, compute_three_hop=True)