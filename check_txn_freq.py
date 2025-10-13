import pandas as pd

main_df = pd.read_csv('警示帳戶交易項整理.csv', index_col=None)

# 確認某帳戶在特定時間範圍內之交易次數是否大於等於n
def check_txn_freq(id, d, freq):
    df = main_df[main_df['id']==id]
    for i in range(1, 122, d):
        if d == 1:
            tw = f'第{i}天'
        else:
            tw = f'第{i}~{i+d-1 if i+d-1<121 else 121}天'
        day = [x for x in range(i, i+d)]
        if len(df[df['txn_date'].isin(day)]) >= freq:
            print(f'在{tw}中交易數至少{freq}筆')

print('測試1:')
check_txn_freq(1, 1, 3)
print('測試2:')
check_txn_freq(1, 7, 3)