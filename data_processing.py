import pandas as pd

df_txn = pd.read_csv('acct_transaction.csv', index_col=None)
df_alert = pd.read_csv('acct_alert.csv', index_col=None)

alert_acct = df_alert['acct']

# 對預警帳號進行編號
alert_acct_id = {
    j:i+1  for i, j in enumerate(alert_acct)
}

alert_from_acct = df_txn[df_txn['from_acct'].isin(alert_acct)]
alert_to_acct = df_txn[df_txn['to_acct'].isin(alert_acct)]

alert_txn = pd.concat([alert_from_acct, alert_to_acct]).drop_duplicates().reset_index(drop=True)

mask = alert_txn['from_acct'].isin(alert_acct) | alert_txn['to_acct'].isin(alert_acct)
id_from = alert_txn['from_acct'].map(alert_acct_id)
id_to   = alert_txn['to_acct'].map(alert_acct_id)
id_col = id_from.combine_first(id_to).astype(int)

final_alert_df = alert_txn.loc[mask].copy()
final_alert_df.insert(0, 'id', id_col[mask])
final_alert_df = final_alert_df.sort_values(by=['id', 'txn_date', 'txn_time'])

final_alert_df.to_csv('警示帳戶交易項整理.csv', index=False)
print(final_alert_df)