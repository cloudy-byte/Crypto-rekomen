import pandas as pd
import numpy as np

def load_data(file):
    return pd.read_excel(file)

def compute_indicators(group):
    group = group.copy()
    group['SMA_7'] = group['Close'].rolling(window=7).mean()
    group['SMA_14'] = group['Close'].rolling(window=14).mean()
    group['EMA_9'] = group['Close'].ewm(span=9, adjust=False).mean()
    group['EMA_12'] = group['Close'].ewm(span=12, adjust=False).mean()
    group['EMA_26'] = group['Close'].ewm(span=26, adjust=False).mean()
    group['MACD'] = group['EMA_12'] - group['EMA_26']

    delta = group['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    group['RSI_14'] = 100 - (100 / (1 + rs))

    obv = [0]
    for i in range(1, len(group)):
        if group['Close'].iloc[i] > group['Close'].iloc[i - 1]:
            obv.append(obv[-1] + group['Volume'].iloc[i])
        elif group['Close'].iloc[i] < group['Close'].iloc[i - 1]:
            obv.append(obv[-1] - group['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    group['OBV'] = obv

    group['Close_pct_change_7d'] = group['Close'].pct_change(periods=7)
    group['MarketCap_to_Volume'] = group['MarketCap'] / group['Volume']
    group['Volume_MA7'] = group['Volume'].rolling(window=7).mean()
    group['TVL_MA7'] = group['tvl'].rolling(window=7).mean()
    return group

def add_indicators(df):
    df_ind = df.groupby("Ticker").apply(compute_indicators).reset_index(drop=True)
    df_ind_clean = df_ind.dropna().reset_index(drop=True)
    return df_ind_clean

def scoring_and_recommendation(df_indicators_clean):
    mcap_volume_median = df_indicators_clean['MarketCap_to_Volume'].median()

    def full_recommendation(group):
        group = group.copy()
        group['Score'] = 0

        group.loc[group['SMA_7'] > group['SMA_14'], 'Score'] += 1
        group.loc[group['SMA_7'] < group['SMA_14'], 'Score'] -= 1

        group.loc[(group['EMA_9'] > group['EMA_12']) & (group['EMA_12'] > group['EMA_26']), 'Score'] += 1
        group.loc[(group['EMA_9'] < group['EMA_12']) & (group['EMA_12'] < group['EMA_26']), 'Score'] -= 1

        group.loc[group['MACD'] > 0, 'Score'] += 1
        group.loc[group['MACD'] < 0, 'Score'] -= 1

        group.loc[group['RSI_14'] < 30, 'Score'] += 1
        group.loc[group['RSI_14'] > 70, 'Score'] -= 1

        group['OBV_diff'] = group['OBV'].diff().fillna(0)
        group.loc[group['OBV_diff'] > 0, 'Score'] += 1
        group.loc[group['OBV_diff'] < 0, 'Score'] -= 1

        group.loc[group['MarketCap_to_Volume'] < mcap_volume_median, 'Score'] += 1
        group.loc[group['MarketCap_to_Volume'] > mcap_volume_median, 'Score'] -= 1

        group['Volume_MA7_diff'] = group['Volume_MA7'].diff().fillna(0)
        group.loc[group['Volume_MA7_diff'] > 0, 'Score'] += 1
        group.loc[group['Volume_MA7_diff'] < 0, 'Score'] -= 1

        group['TVL_MA7_diff'] = group['TVL_MA7'].diff().fillna(0)
        group.loc[group['TVL_MA7_diff'] > 0, 'Score'] += 1
        group.loc[group['TVL_MA7_diff'] < 0, 'Score'] -= 1

        group['Recommendation'] = group['Score'].apply(lambda x: 'Long' if x >= 3 else ('Short' if x <= -3 else 'Hold'))

        return group

    df_scored = df_indicators_clean.groupby("Ticker").apply(full_recommendation).reset_index(drop=True)
    return df_scored

def rule_based_prediction(row):
    score = 0
    if row['SMA_7'] > row['SMA_14']:
        score += 1
    else:
        score -= 1

    if row['EMA_9'] > row['EMA_26']:
        score += 1
    else:
        score -= 1

    if row['MACD'] > 0:
        score += 1
    else:
        score -= 1

    if row['RSI_14'] < 30:
        score += 1
    elif row['RSI_14'] > 70:
        score -= 1

    if row['Volume'] > row['Volume_MA7']:
        score += 1
    else:
        score -= 1

    if row['tvl'] > row['TVL_MA7']:
        score += 1
    else:
        score -= 1

    if score >= 3:
        return "Long"
    elif score <= -3:
        return "Short"
    else:
        return "Hold"

def get_score(prediction, pct_change):
    prediction = prediction.strip().lower()
    if prediction == 'long' and pct_change > 0:
        return 1
    elif prediction == 'short' and pct_change < 0:
        return 1
    elif prediction == 'hold':
        return 0
    else:
        return -1

def evaluate_asset(ticker, df_indicators_clean):
    asset = df_indicators_clean[
        (df_indicators_clean['Ticker'] == ticker) &
        (df_indicators_clean['Date'] >= pd.Timestamp('2025-01-01'))
    ].copy()

    asset.set_index('Date', inplace=True)
    weekly = asset.resample('7D').last().dropna().reset_index()

    results = []
    correct = 0
    total = 0

    for i in range(len(weekly) - 1):
        current = weekly.iloc[i]
        future = weekly.iloc[i + 1]

        current_price = current['Close']
        next_price = future['Close']
        pct_change = (next_price - current_price) / current_price

        prediction = rule_based_prediction(current)

        score = get_score(prediction, pct_change)

        if score == 1:
            correct += 1
        if score != 0:
            total += 1

        results.append({
            'Prediction Date': current['Date'],
            'Evaluation Date': future['Date'],
            'Price Now': current_price,
            'Price Next': next_price,
            'Pct Change': round(pct_change * 100, 2),
            'Recommendation': prediction,
            'Score': score
        })

    eval_df = pd.DataFrame(results)
    eval_df['Cumulative Score'] = eval_df['Score'].cumsum()

    accuracy = (correct / total) * 100 if total > 0 else 0

    summary = {
        'Ticker': ticker,
        'Total Weeks': len(eval_df),
        'Total Decisions': total,
        'Correct Decisions': correct,
        'Incorrect Decisions': total - correct,
        'Accuracy %': round(accuracy, 2),
        'Final Score (Cumulative)': eval_df['Cumulative Score'].iloc[-1] if not eval_df.empty else 0,
        'Total Positive Score': (eval_df['Score'] == 1).sum(),
        'Total Negative Score': (eval_df['Score'] == -1).sum(),
    }

    return eval_df, summary

def evaluate_all_assets(df_indicators_clean):
    tickers = df_indicators_clean['Ticker'].unique()
    results_list = []

    for t in tickers:
        eval_df, summary = evaluate_asset(t, df_indicators_clean)
        last_reco = eval_df.sort_values('Prediction Date').iloc[-1]['Recommendation'] if not eval_df.empty else None
        summary['Last Recommendation'] = last_reco
        results_list.append(summary)

    results_df = pd.DataFrame(results_list)
    results_df.sort_values(by='Accuracy %', ascending=False, inplace=True)
    return results_df

def main_pipeline(df, mode='full'):

    if isinstance(df, str):
        print("🔄 Memuat data dari file...")
        df = load_data(df)

    if mode == 'evaluate':
        print("🧪 Mengevaluasi semua aset...")
        evaluation_df = evaluate_all_assets(df)
        print("✅ Evaluasi selesai.")
        return evaluation_df

    print("📊 Menghitung indikator teknikal dan on-chain...")
    df_indicators = add_indicators(df)

    print("🧮 Memberikan skor dan rekomendasi...")
    df_scored = scoring_and_recommendation(df_indicators)

    if mode == 'rekomendasi':
        return df_scored

    print("🧪 Mengevaluasi semua aset...")
    evaluation_df = evaluate_all_assets(df_scored)
    print("✅ Evaluasi selesai.")

    print("📈 5 Aset dengan akurasi tertinggi:")
    print(evaluation_df[['Ticker', 'Accuracy %']].head())

    print("🔻 5 Aset dengan akurasi terendah:")
    print(evaluation_df[['Ticker', 'Accuracy %']].tail())

    return df_scored, evaluation_df


if __name__ == "__main__":
    df = pd.read_excel('crypto_data_tvl (4).xlsx')
    df_rekom, df_eval = main_pipeline(df, mode='full')
    print("Hasil evaluasi:")
    print(df_eval[['Ticker', 'Accuracy %', 'Total Decisions', 'Correct Decisions', 'Last Recommendation']])

    total_correct = df_eval['Correct Decisions'].sum()
    total_decisions = df_eval['Total Decisions'].sum()
    overall_accuracy = (total_correct / total_decisions) * 100 if total_decisions > 0 else 0
    print(f"Akurasi prediksi keseluruhan: {overall_accuracy:.2f}%")

    mean_accuracy = df_eval['Accuracy %'].mean()
    print(f"Rata-rata akurasi per ticker: {mean_accuracy:.2f}%")


