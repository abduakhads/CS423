import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load the Data from CSV
    file_path = 'final_results.csv' 
    df = pd.read_csv(file_path)

    # 2. Data Cleaning
    def clean_currency(x):
        if isinstance(x, str):
            return float(x.replace('$', '').replace(',', ''))
        return float(x)

    cols_to_clean = ['Ground_Truth', 'Classical_Pred', 'LSTM_Pred', 'EasyOCR_Pred']
    for col in cols_to_clean:
        df[col] = df[col].apply(clean_currency)

    # 3. Calculate Metrics
    def check_correct(row, pred_col, truth_col='Ground_Truth'):
        return round(row[pred_col], 2) == round(row[truth_col], 2)

    df['Classical_Correct_Strict'] = df.apply(lambda row: check_correct(row, 'Classical_Pred'), axis=1)
    df['LSTM_Correct_Strict'] = df.apply(lambda row: check_correct(row, 'LSTM_Pred'), axis=1)
    df['EasyOCR_Correct_Strict'] = df.apply(lambda row: check_correct(row, 'EasyOCR_Pred'), axis=1)

    # Aggregate the results
    models = ['Classical', 'LSTM', 'EasyOCR']
    correct_counts = [
        df['Classical_Correct_Strict'].sum(),
        df['LSTM_Correct_Strict'].sum(),
        df['EasyOCR_Correct_Strict'].sum()
    ]
    avg_times = [
        df['Classical_Time_ms'].mean(),
        df['LSTM_Time_ms'].mean(),
        df['EasyOCR_Time_ms'].mean()
    ]

    summary_df = pd.DataFrame({
        'Model': models,
        'Correct_Count': correct_counts,
        'Average_Time_ms': avg_times
    })

    print("Performance Summary:")
    print(summary_df)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1 FIX: Added hue='Model' and legend=False ---
    sns.barplot(
        x='Model', 
        y='Correct_Count', 
        hue='Model',  # Fix for FutureWarning
        data=summary_df, 
        ax=ax1, 
        palette='viridis', 
        legend=False  # Fix for FutureWarning
    )
    ax1.set_title('Number of Correct Predictions per Model')
    ax1.set_ylabel('Count of Correct Predictions')
    ax1.set_ylim(0, len(df) + 2) 
    
    for i, v in enumerate(correct_counts):
        ax1.text(i, v + 0.2, str(v), ha='center', fontweight='bold')

    # --- Plot 2 FIX: Added hue='Model' and legend=False ---
    sns.barplot(
        x='Model', 
        y='Average_Time_ms', 
        hue='Model',  # Fix for FutureWarning
        data=summary_df, 
        ax=ax2, 
        palette='magma', 
        legend=False  # Fix for FutureWarning
    )
    ax2.set_title('Average Inference Time per Model (ms)')
    ax2.set_ylabel('Time (ms)')
    
    for i, v in enumerate(avg_times):
        ax2.text(i, v + 100, f"{v:.0f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("experiment_summary.png")
    plt.show()

if __name__ == "__main__":
    main()