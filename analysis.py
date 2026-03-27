import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from scipy.stats import f_oneway

# 自动创建输出文件夹
os.makedirs("./figures", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ===================== 1. 读取数据 =====================
df = pd.read_sas("./data/c12diet.sas7bdat")
df.columns = df.columns.str.upper()

# 选择需要的列（T2 = 城乡）
df = df[['T2', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
df = df[(df['D3KCAL'] > 500) & (df['D3KCAL'] < 5000)]

# ===================== 2. 城乡分组（T2 = 1城市 / 2乡村）=====================
df['Group'] = df['T2'].map({1: 'Urban', 2: 'Rural'})

# ===================== 3. 4个供能比特征 =====================
df['fat_energy_ratio']    = (df['D3FAT'] * 9) / df['D3KCAL']
df['carbo_energy_ratio']  = (df['D3CARBO'] * 4) / df['D3KCAL']
df['protn_energy_ratio']  = (df['D3PROTN'] * 4) / df['D3KCAL']
df['fat_carbo_ratio']     = df['D3FAT'] / (df['D3CARBO'] + 1e-6)

# ===================== 4. 分组均值 =====================
group_mean = df.groupby('Group')[
    ['fat_energy_ratio',
     'carbo_energy_ratio',
     'protn_energy_ratio',
     'fat_carbo_ratio']
].mean().round(4)

# ===================== 5. ANOVA 显著性 =====================
def anova(col):
    u = df[df['T2'] == 1][col]
    r = df[df['T2'] == 2][col]
    return f_oneway(u, r)

f_fat, p_fat = anova('fat_energy_ratio')
f_car, p_car = anova('carbo_energy_ratio')
f_pro, p_pro = anova('protn_energy_ratio')
f_fc,  p_fc  = anova('fat_carbo_ratio')

# ===================== 6. 保存结果 =====================
with open("./results/urban_rural_analysis.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("📊 城乡膳食结构分析 (Urban vs Rural)\n")
    f.write("="*80 + "\n\n")

    f.write("✅ 组间均值（供能比）\n")
    f.write(group_mean.to_string())
    f.write("\n\n")

    f.write("✅ ANOVA 显著性检验\n")
    f.write(f"脂肪供能比       F={f_fat:.2f}    p={p_fat:.2e}\n")
    f.write(f"碳水供能比       F={f_car:.2f}    p={p_car:.2e}\n")
    f.write(f"蛋白质供能比     F={f_pro:.2f}    p={p_pro:.2e}\n")
    f.write(f"脂肪/碳水比      F={f_fc:.2f}    p={p_fc:.2e}\n")

# ===================== 7. 热力图 =====================
heat_data = group_mean.apply(lambda x: (x - x.mean()) / x.std())

plt.figure(figsize=(10, 4))
sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Urban vs Rural | Dietary Structure Ratios (Standardized)", fontsize=12)
plt.tight_layout()
plt.savefig("./figures/urban_rural_heatmap.png", dpi=300)

# ===================== 8. 柱状图 =====================
group_mean.plot(kind='bar', figsize=(12, 5))
plt.title("Urban vs Rural Dietary Comparison")
plt.ylabel("Mean Value")
plt.tight_layout()
plt.savefig("./figures/urban_rural_bar.png", dpi=300)

print("✅ 城乡分析完成！（T2 字段）")
print("📄 结果：results/urban_rural_analysis.txt")
print("🖼 图片：figures/urban_rural_heatmap.png")