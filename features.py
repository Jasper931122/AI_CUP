import pandas as pd
import numpy as np

def get_features(f):

    df = pd.read_csv(f + '_preprocessing.csv', index_col=None)
    
    grp_all = df.groupby('id', as_index=False).agg(
        total_txn_count   = ('txn_amt', 'count'),
        total_txn_amount  = ('txn_amt', 'sum'),
        avg_txn_amount    = ('txn_amt', 'mean'),
        std_txn_amount    = ('txn_amt', 'std'),
        pct_small_txn     = ('txn_amt', lambda x: np.mean(x < 1000))
    )

    send_mask = df['id'] == df['from_acct']
    grp_send = df[send_mask].groupby('id', as_index=False).agg(
        send_count       = ('txn_amt', 'count'),
        send_sum         = ('txn_amt', 'sum'),
        send_avg         = ('txn_amt', 'mean'),
        send_std         = ('txn_amt', 'std'),
        send_pct_small   = ('txn_amt', lambda x: np.mean(x < 1000))
    )

    recv_mask = df['id'] == df['to_acct']
    grp_recv = df[recv_mask].groupby('id', as_index=False).agg(
        recv_count       = ('txn_amt', 'count'),
        recv_sum         = ('txn_amt', 'sum'),
        recv_avg         = ('txn_amt', 'mean'),
        recv_std         = ('txn_amt', 'std'),
        recv_pct_small   = ('txn_amt', lambda x: np.mean(x < 1000))
    )

    feat = (
        grp_all
        .merge(grp_send, on='id', how='left')
        .merge(grp_recv, on='id', how='left')
    )

    zero_cols = [
        'send_count','send_sum','send_avg','send_std','send_pct_small',
        'recv_count','recv_sum','recv_avg','recv_std','recv_pct_small'
    ]
    for col in zero_cols:
        feat[col] = feat[col].fillna(0)

    feat['send_count_ratio'] = np.where(
        feat['total_txn_count'] > 0,
        feat['send_count'] / feat['total_txn_count'],
        0
    )

    feat['recv_count_ratio'] = np.where(
        feat['total_txn_count'] > 0,
        feat['recv_count'] / feat['total_txn_count'],
        0
    )

    feat['send_sum_ratio'] = np.where(
        feat['total_txn_amount'] > 0,
        feat['send_sum'] / feat['total_txn_amount'],
        0
    )

    feat['recv_sum_ratio'] = np.where(
        feat['total_txn_amount'] > 0,
        feat['recv_sum'] / feat['total_txn_amount'],
        0
    )

    feat = feat[[
        'id',

        'total_txn_count',
        'total_txn_amount',
        'avg_txn_amount',
        'std_txn_amount',
        'pct_small_txn',

        'send_count',
        'send_count_ratio',
        'send_sum',
        'send_sum_ratio',
        'send_avg',
        'send_std',
        'send_pct_small',

        'recv_count',
        'recv_count_ratio',
        'recv_sum',
        'recv_sum_ratio',
        'recv_avg',
        'recv_std',
        'recv_pct_small',
    ]]

    df['txn_date'] = pd.to_datetime(df['txn_date'])

    life_stats = df.groupby('id').agg(
        first_day = ('txn_date', 'min'),
        last_day  = ('txn_date', 'max'),
        active_days = ('txn_date', lambda x: x.nunique()),
        total_txn_count = ('txn_amt', 'count'),
        total_txn_amount = ('txn_amt', 'sum')
    ).reset_index()

    # 存活日
    life_stats['lifetime_days'] = (life_stats['last_day'] - life_stats['first_day']).dt.days + 1

    # 活躍比例
    life_stats['active_ratio'] = life_stats['active_days'] / life_stats['lifetime_days']

    # 平均每個交易日的交易筆數
    life_stats['tx_per_active_day'] = life_stats['total_txn_count'] / life_stats['active_days']

    # 平均每個交易日的交易金額
    life_stats['amt_per_active_day'] = life_stats['total_txn_amount'] / life_stats['active_days']

    life_stats_feat = life_stats[['id',
        'lifetime_days',
        'active_days',
        'active_ratio',
        'tx_per_active_day',
        'amt_per_active_day'
    ]]

    feat = feat.merge(life_stats_feat, on='id', how='left')

    df['txn_hour'] = df['txn_time'].str.slice(0,2).astype(int)

    def bucket(hour):
        if   hour < 6:  return '00_06'
        elif hour < 12: return '06_12'
        elif hour < 18: return '12_18'
        else:           return '18_24'

    df['time_bucket'] = df['txn_hour'].apply(bucket)

    # 匯款視角
    send_time = df[df['id'] == df['from_acct']].groupby(['id','time_bucket']).size().unstack(fill_value=0)

    # 換成比例
    send_time = send_time.div(send_time.sum(axis=1), axis=0).fillna(0)

    # 欄位重新命名避免和收款撞名
    send_time = send_time.add_prefix('send_time_ratio_')  # e.g. send_time_ratio_00_06

    # 收款視角
    recv_time = df[df['id'] == df['to_acct']].groupby(['id','time_bucket']).size().unstack(fill_value=0)
    recv_time = recv_time.div(recv_time.sum(axis=1), axis=0).fillna(0)
    recv_time = recv_time.add_prefix('recv_time_ratio_')

    feat = feat.merge(send_time, on='id', how='left')
    feat = feat.merge(recv_time, on='id', how='left')

    time_ratio_cols = [c for c in feat.columns if c.startswith('send_time_ratio_') or c.startswith('recv_time_ratio_')]
    feat[time_ratio_cols] = feat[time_ratio_cols].fillna(0)

    def burst_metrics(subdf):

        subdf = subdf.copy()
        subdf['txn_date'] = pd.to_datetime(subdf['txn_date'])
        first_day = subdf['txn_date'].min()
        last_day  = subdf['txn_date'].max()
        lifetime_days = (last_day - first_day).days + 1

        total_cnt = len(subdf)
        total_amt = subdf['txn_amt'].sum()

        baseline_daily_tx  = total_cnt / lifetime_days
        baseline_daily_amt = total_amt / lifetime_days

        def window_stats(days):
            start = last_day - pd.Timedelta(days=days-1, unit='D')
            mask = (subdf['txn_date'] >= start) & (subdf['txn_date'] <= last_day)
            w_cnt = mask.sum()
            w_amt = subdf.loc[mask, 'txn_amt'].sum()
            w_days = days
            avg_daily_tx  = w_cnt / w_days
            avg_daily_amt = w_amt / w_days
            # ratio vs baseline
            burst_tx_ratio  = (avg_daily_tx  / baseline_daily_tx)  if baseline_daily_tx  > 0 else 0
            burst_amt_ratio = (avg_daily_amt / baseline_daily_amt) if baseline_daily_amt > 0 else 0
            return burst_tx_ratio, burst_amt_ratio

        tx1, amt1 = window_stats(1)
        tx3, amt3 = window_stats(3)
        tx7, amt7 = window_stats(7)

        return pd.Series({
            'burst_tx_ratio_1d': tx1,
            'burst_tx_ratio_3d': tx3,
            'burst_tx_ratio_7d': tx7,
            'burst_amt_ratio_1d': amt1,
            'burst_amt_ratio_3d': amt3,
            'burst_amt_ratio_7d': amt7,
        })

    burst_df = df.groupby('id').apply(burst_metrics).reset_index()

    feat = feat.merge(burst_df, on='id', how='left')

    def daily_peaks(subdf):
        subdf = subdf.copy()
        subdf['txn_date'] = pd.to_datetime(subdf['txn_date'])

        # 收款視角：id == to_acct
        recv = subdf[subdf['id'] == subdf['to_acct']]
        if len(recv):
            recv_daily = recv.groupby('txn_date').agg(
                recv_day_cnt = ('txn_amt','count'),
                recv_day_amt = ('txn_amt','sum'),
                recv_day_unique_from = ('from_acct', 'nunique')
            )
            recv_cnt_peak  = recv_daily['recv_day_cnt'].max()
            recv_amt_peak  = recv_daily['recv_day_amt'].max()
            recv_partners_peak = recv_daily['recv_day_unique_from'].max()
        else:
            recv_cnt_peak = recv_amt_peak = recv_partners_peak = 0

        # 匯款視角：id == from_acct
        send = subdf[subdf['id'] == subdf['from_acct']]
        if len(send):
            send_daily = send.groupby('txn_date').agg(
                send_day_cnt = ('txn_amt','count'),
                send_day_amt = ('txn_amt','sum'),
                send_day_unique_to = ('to_acct', 'nunique')
            )
            send_cnt_peak  = send_daily['send_day_cnt'].max()
            send_amt_peak  = send_daily['send_day_amt'].max()
            send_partners_peak = send_daily['send_day_unique_to'].max()
        else:
            send_cnt_peak = send_amt_peak = send_partners_peak = 0

        return pd.Series({
            'recv_peak_cnt_per_day': recv_cnt_peak,
            'recv_peak_amt_per_day': recv_amt_peak,
            'recv_peak_unique_senders': recv_partners_peak,
            'send_peak_cnt_per_day': send_cnt_peak,
            'send_peak_amt_per_day': send_amt_peak,
            'send_peak_unique_receivers': send_partners_peak,
        })

    peaks_df = df.groupby('id').apply(daily_peaks).reset_index()

    feat = feat.merge(peaks_df, on='id', how='left')

    feat.to_csv(f + '_features.csv', index=False)

get_features('alert')
get_features('normal')
get_features('training')
get_features('predict')