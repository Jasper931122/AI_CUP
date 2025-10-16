import pandas as pd

df_txn = pd.read_csv('acct_transaction.csv', index_col=None)
df_alert = pd.read_csv('acct_alert.csv', index_col=None)
df_predict = pd.read_csv('acct_predict.csv', index_col=None)
all_acct = df_txn['from_acct']
alert_acct = df_alert['acct']
predict_acct = df_predict['acct']

# 對預警帳號進行編號
alert_acct_id = {
    j:f'A{i+1:0>5}'  for i, j in enumerate(alert_acct)
}

alert_from_acct = df_txn[all_acct.isin(alert_acct)]
alert_to_acct = df_txn[df_txn['to_acct'].isin(alert_acct)]

alert_txn = pd.concat([alert_from_acct, alert_to_acct]).drop_duplicates().reset_index(drop=True)
df = alert_txn[alert_txn['from_acct'].isin(alert_acct) | alert_txn['to_acct'].isin(alert_acct)].copy()

id_from = alert_txn['from_acct'].map(alert_acct_id)
id_to   = alert_txn['to_acct'].map(alert_acct_id)

df['id_list'] = pd.concat([id_from, id_to], axis=1).apply(
    lambda r: sorted(set(x for x in r if pd.notna(x))), axis=1
)

final_alert_df = df.explode('id_list', ignore_index=True).rename(columns={'id_list': 'id'})
cols = ['id'] + [c for c in final_alert_df.columns if c != 'id']
final_alert_df = final_alert_df[cols]
final_alert_df = final_alert_df.sort_values(by=['id', 'txn_date', 'txn_time'])

final_alert_df.to_csv('alert_preprocessing.csv', index=False)

all_acct = pd.concat([final_alert_df['from_acct'], final_alert_df['to_acct']])
warning_acct = pd.DataFrame({'acct': list(set(all_acct[~all_acct.isin(alert_acct)]))})
warning_acct_id = {
    j:f'W{i+1:0>5}'  for i, j in enumerate(warning_acct['acct'])
}
warning_acct_df = pd.DataFrame([warning_acct_id])
warning_acct_df = warning_acct_df.T.reset_index()
warning_acct_df.columns = ['acct', 'id']
# warning_acct_df.to_csv('與警示帳戶交易的非警示帳戶.csv', index=False)

exclude = pd.concat([alert_acct, warning_acct])
normal_acct = all_acct[~all_acct.isin(exclude)].drop_duplicates().head(1004)
normal_acct_id = {
    j:f'N{i+1:0>5}'  for i, j in enumerate(normal_acct)
}

mask = df_txn['from_acct'].isin(normal_acct) | df_txn['to_acct'].isin(normal_acct)
normal_txn = df_txn[mask]

id_from = normal_txn['from_acct'].map(normal_acct_id)
id_to   = normal_txn['to_acct'].map(normal_acct_id)
normal_txn['id_list'] = pd.concat([id_from, id_to], axis=1).apply(
    lambda r: sorted(set(x for x in r if pd.notna(x))), axis=1
)
final_normal_df = normal_txn.explode('id_list', ignore_index=True).rename(columns={'id_list': 'id'})
cols = ['id'] + [c for c in final_normal_df.columns if c != 'id']
final_normal_df = final_normal_df[cols]
final_normal_df = final_normal_df.sort_values(by=['id', 'txn_date', 'txn_time'])
final_normal_df.to_csv('normal_preprocessing.csv', index=False)

predict_acct_id = {
    j: f'P{i+1:0>5}' for i, j in enumerate(predict_acct)
}

mask = df_txn['from_acct'].isin(predict_acct) | df_txn['to_acct'].isin(predict_acct)
predict_txn = df_txn.loc[mask].copy()

id_from = predict_txn['from_acct'].map(predict_acct_id)
id_to   = predict_txn['to_acct'].map(predict_acct_id)

predict_txn['id_list'] = pd.concat([id_from, id_to], axis=1).apply(
    lambda r: sorted(set(x for x in r if pd.notna(x))), axis=1
)

final_predict_df = (
    predict_txn
    .explode('id_list', ignore_index=True)
    .rename(columns={'id_list': 'id'})
)

ols = ['id'] + [c for c in final_predict_df.columns if c != 'id']
final_predict_df = final_predict_df[cols]

final_predict_df = final_predict_df.sort_values(
    by=['id', 'txn_date', 'txn_time']
).reset_index(drop=True)

final_predict_df.to_csv('predict_preprocessing.csv', index=False)

df_train = pd.concat([final_alert_df, final_normal_df], ignore_index=True)
df_train.to_csv('training_preprocessing.csv', index=False)