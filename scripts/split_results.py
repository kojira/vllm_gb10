import csv
import os
from collections import defaultdict

# 対象のCSVファイル
INPUT_CSV = "benchmarks/20251129_150915/result.csv"
OUTPUT_DIR = os.path.dirname(INPUT_CSV)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # データをモデルごとに分類
    data_by_model = defaultdict(list)
    fieldnames = []

    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictWriter(f, fieldnames=[]) # ダミーで初期化
        # ヘッダーを読み込むためにcsv.readerを使うか、DictReaderのfieldnamesを利用
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            model_name = row['model']
            data_by_model[model_name].append(row)

    # モデルごとにCSV書き出し
    print(f"Splitting results into {len(data_by_model)} files...")
    
    for model_name, rows in data_by_model.items():
        # ファイル名に使えない文字を置換（念のため）
        safe_name = model_name.replace("/", "_").replace(":", "_")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  Created: {output_path} ({len(rows)} rows)")

    print("Done.")

if __name__ == "__main__":
    main()

