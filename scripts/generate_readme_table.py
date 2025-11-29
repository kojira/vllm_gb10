import pandas as pd
import glob
import os

# 最新のベンチマークディレクトリを取得
benchmarks_base_dir = "benchmarks"
list_of_dirs = glob.glob(os.path.join(benchmarks_base_dir, "2025*"))
if not list_of_dirs:
    print(f"No benchmark directories found in {benchmarks_base_dir}")
    exit()

latest_dir = max(list_of_dirs, key=os.path.getmtime)
result_csv = os.path.join(latest_dir, "result.csv")

if not os.path.exists(result_csv):
    print(f"Result CSV not found: {result_csv}")
    exit()

print(f"Reading results from {result_csv}")
df = pd.read_csv(result_csv)

# モデルごとにグループ化
models = df['model'].unique()

print("\n## シングルストリーム ベンチマーク結果 (全トークン長)\n")
print("| Model | 64 tokens | 128 tokens | 256 tokens | 512 tokens | 1024 tokens |")
print("|:------|----------:|-----------:|-----------:|-----------:|------------:|")

for model in sorted(models):
    model_df = df[df['model'] == model]
    row = [f"**{model}**"]
    
    for target in [64, 128, 256, 512, 1024]:
        target_df = model_df[model_df['target_tokens'] == target]
        if len(target_df) > 0:
            tps = target_df.iloc[0]['tps']
            row.append(f"{tps:.2f}")
        else:
            row.append("N/A")
    
    print("| " + " | ".join(row) + " |")

print("\n*注: TPS (Tokens Per Second) = 生成トークン数 / 生成時間*")

