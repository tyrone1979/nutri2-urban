import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from scipy.stats import f_oneway

# ===================== 省份编码映射（英文）=====================
PROVINCE_MAP = {
    11: "Beijing(2011+)",
    21: "Liaoning(1997miss)", 
    23: "Heilongjiang(1997+)",
    31: "Shanghai(2011+)",
    32: "Jiangsu",
    37: "Shandong", 
    41: "Henan",
    42: "Hubei",
    43: "Hunan",
    45: "Guangxi",
    52: "Guizhou",
    55: "Chongqing(2011+)"
}

# 简化版名称（用于图表）
PROVINCE_SHORT = {
    11: "Beijing",
    21: "Liaoning", 
    23: "Heilongjiang",
    31: "Shanghai",
    32: "Jiangsu",
    37: "Shandong", 
    41: "Henan",
    42: "Hubei",
    43: "Hunan",
    45: "Guangxi",
    52: "Guizhou",
    55: "Chongqing"
}

# 自动创建输出文件夹
os.makedirs("./figures", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ===================== 1. 读取数据 =====================
df = pd.read_sas("./data/c12diet.sas7bdat")
df.columns = df.columns.str.upper()

# 选择需要的列（T2 = 城乡, T1 = 省份, WAVE = 调查年份）
df = df[['T2', 'T1', 'WAVE', 'D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']].dropna()
df = df[(df['D3KCAL'] > 500) & (df['D3KCAL'] < 5000)]

# ===================== 2. 基础分组 =====================
df['Group'] = df['T2'].map({1: 'Urban', 2: 'Rural'})
df['Year'] = df['WAVE'].astype(int)
df['Province_Code'] = df['T1'].astype(int)
df['Province'] = df['Province_Code'].map(PROVINCE_SHORT)
df['Province_Full'] = df['Province_Code'].map(PROVINCE_MAP)

# 查看省份分布
province_counts = df['Province_Code'].value_counts().sort_index()
print("📍 Province Distribution:")
for code, count in province_counts.items():
    name = PROVINCE_SHORT.get(code, f"Unknown{code}")
    print(f"   {code}: {name:12s} n={count:5d}")

print(f"\n📅 Year Range: {df['Year'].min()} - {df['Year'].max()}")
print(f"📊 Total Sample: {len(df)}")

# 选择样本量充足的省份（至少1000个样本）
major_provinces = province_counts[province_counts >= 1000].index.tolist()
df_major = df[df['Province_Code'].isin(major_provinces)].copy()
print(f"\n✅ Major Provinces (≥1000 samples): {len(major_provinces)}")

# ===================== 3. 4个供能比特征 =====================
df['fat_energy_ratio']    = (df['D3FAT'] * 9) / df['D3KCAL']
df['carbo_energy_ratio']  = (df['D3CARBO'] * 4) / df['D3KCAL']
df['protn_energy_ratio']  = (df['D3PROTN'] * 4) / df['D3KCAL']
df['fat_carbo_ratio']     = df['D3FAT'] / (df['D3CARBO'] + 1e-6)

features = ['fat_energy_ratio', 'carbo_energy_ratio', 'protn_energy_ratio', 'fat_carbo_ratio']

# 应用到df_major
for col in features:
    df_major[col] = df[col]

# ===================== 4. 基础分析 =====================
group_mean = df.groupby('Group')[features].mean().round(4)

def anova_urban_rural(data, col):
    u = data[data['T2'] == 1][col]
    r = data[data['T2'] == 2][col]
    if len(u) > 0 and len(r) > 0:
        return f_oneway(u, r)
    return (np.nan, np.nan)

# ===================== 5. 省份分层分析 =====================
province_urban_rural = df_major.groupby(['Province', 'Group'])[features].mean().round(4)
province_overall = df_major.groupby('Province')[features].mean().round(4)

def anova_by_province(data, col):
    """每个省份内做城乡ANOVA"""
    results = {}
    for prov_code in data['Province_Code'].unique():
        subset = data[data['Province_Code'] == prov_code]
        u = subset[subset['T2'] == 1][col]
        r = subset[subset['T2'] == 2][col]
        if len(u) >= 30 and len(r) >= 30:
            f, p = f_oneway(u, r)
            prov_name = PROVINCE_SHORT.get(prov_code, str(prov_code))
            results[prov_name] = {
                'code': prov_code,
                'F': f, 
                'p': p, 
                'n_urban': len(u), 
                'n_rural': len(r)
            }
    return results

# ===================== 6. 时间趋势分析（Year）=====================
year_trend = df.groupby(['Year', 'Group'])[features].mean().round(4)
year_overall = df.groupby('Year')[features].mean().round(4)

# ===================== 7. 保存结果 =====================
with open("./results/urban_rural_analysis.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("Urban vs Rural Dietary Structure Analysis (CHNS 1991-2011)\n")
    f.write("="*80 + "\n")
    f.write(f"Data Source: China Health and Nutrition Survey\n")
    f.write(f"T1 = Province (n={len(province_counts)} regions)\n")
    f.write(f"WAVE = Survey Year (Range: {df['Year'].min()}-{df['Year'].max()})\n") 
    f.write(f"Major Provinces (≥1000 samples): {len(major_provinces)}\n\n")

    # 基础城乡对比
    f.write("Mean Values by Group (Energy Ratio)\n")
    f.write(group_mean.to_string())
    f.write("\n\n")

    f.write("ANOVA Significance Test (Urban vs Rural Overall)\n")
    for feat in features:
        f_stat, p_val = anova_urban_rural(df, feat)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        f.write(f"{feat:20s} F={f_stat:8.2f}    p={p_val:.2e} {sig}\n")

    # 省份分层
    f.write("\n" + "="*80 + "\n")
    f.write(f"Provincial Analysis (Urban × Rural)\n")
    f.write("="*80 + "\n\n")
    f.write("Mean by Province and Group:\n")
    f.write(province_urban_rural.to_string())
    f.write("\n\n")

    f.write("Overall Mean by Province:\n")
    f.write(province_overall.to_string())
    f.write("\n\n")

    f.write("Provincial ANOVA (p<0.05 significant results only)\n")
    for feat in features:
        prov_results = anova_by_province(df_major, feat)
        sig_provs = {k: v for k, v in prov_results.items() if v['p'] < 0.05}
        if sig_provs:
            f.write(f"\n{feat} (Significant: {len(sig_provs)}/{len(prov_results)}):\n")
            for prov_name, res in sorted(sig_provs.items(), key=lambda x: x[1]['p']):
                sig = "***" if res['p'] < 0.001 else "**" if res['p'] < 0.01 else "*"
                f.write(f"  {prov_name:12s}(Code{res['code']:2d}): F={res['F']:7.2f}  p={res['p']:.2e} {sig}  (Urban n={res['n_urban']}, Rural n={res['n_rural']})\n")
        else:
            f.write(f"\n{feat}: No significant provincial differences\n")

    # 时间趋势
    f.write("\n" + "="*80 + "\n")
    f.write("Temporal Trends by Survey Year\n")
    f.write("="*80 + "\n\n")
    f.write("Overall Trend:\n")
    f.write(year_overall.to_string())
    f.write("\n\nTrend by Group:\n")
    f.write(year_trend.to_string())

    # 各年份内的城乡差异
    f.write("\n\nYearly ANOVA (Urban vs Rural by Year)\n")
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        f.write(f"\nYear {year} (n={len(year_data)}):\n")
        for feat in features:
            f_stat, p_val = anova_urban_rural(year_data, feat)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            f.write(f"  {feat:20s} F={f_stat:8.2f}  p={p_val:.2e} {sig}\n")

print("\n✅ Results saved to: results/urban_rural_analysis.txt")

# ===================== 8. 可视化 =====================

# 图1：基础城乡热力图
plt.figure(figsize=(10, 4))
heat_data = group_mean.apply(lambda x: (x - x.mean()) / x.std())
sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Urban vs Rural | Dietary Structure (Standardized)", fontsize=12)
plt.tight_layout()
plt.savefig("./figures/01_urban_rural_heatmap.png", dpi=300)
plt.close()

# 图2：城乡柱状图
fig, ax = plt.subplots(figsize=(12, 5))
group_mean.plot(kind='bar', ax=ax)
plt.title("Urban vs Rural Dietary Comparison")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("./figures/02_urban_rural_bar.png", dpi=300)
plt.close()

# 图3：各省份城乡对比（脂肪供能比）- 使用英文省份名
fig, ax = plt.subplots(figsize=(14, 6))
prov_pivot = province_urban_rural.reset_index().pivot(index='Province', columns='Group', values='fat_energy_ratio')
prov_pivot.plot(kind='bar', ax=ax, width=0.8)
plt.title("Fat Energy Ratio by Province and Urban/Rural")
plt.ylabel("Fat Energy Ratio")
plt.xlabel("Province")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Group')
plt.tight_layout()
plt.savefig("./figures/03_province_fat_ratio.png", dpi=300)
plt.close()

# 图4：时间趋势折线图（分城乡）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features):
    ax = axes[idx]
    for group in ['Urban', 'Rural']:
        data = year_trend.loc[(slice(None), group), feat].reset_index(level=1, drop=True)
        ax.plot(data.index, data.values, marker='o', label=group, linewidth=2, markersize=8)
    ax.set_title(f"{feat} by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel(feat)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./figures/04_year_trends.png", dpi=300)
plt.close()

# 图5：省份×城乡热力图（4个子图）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feat in enumerate(features):
    ax = axes[idx]
    pivot_data = df_major.groupby(['Province', 'Group'])[feat].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
               center=pivot_data.values.mean(), ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(f"{feat} by Province and Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Province")

plt.tight_layout()
plt.savefig("./figures/05_province_urban_heatmaps.png", dpi=300)
plt.close()

# 图6：各年份城乡差异变化（看差异是否随时间缩小）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features):
    ax = axes[idx]
    diff_by_year = []
    years = sorted(df['Year'].unique())

    for year in years:
        year_data = df[df['Year'] == year]
        urban_mean = year_data[year_data['Group'] == 'Urban'][feat].mean()
        rural_mean = year_data[year_data['Group'] == 'Rural'][feat].mean()
        diff_by_year.append(urban_mean - rural_mean)

    ax.plot(years, diff_by_year, marker='o', linewidth=2, markersize=8, color='darkred')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f"Urban-Rural Gap in {feat} by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Urban - Rural Difference")
    ax.grid(True, alpha=0.3)

    # 标注趋势方向
    if diff_by_year[-1] > diff_by_year[0]:
        trend = "↑ Gap Widening"
        color = 'red'
    else:
        trend = "↓ Gap Narrowing" 
        color = 'green'
    ax.text(0.02, 0.98, trend, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', color=color, fontweight='bold')

plt.tight_layout()
plt.savefig("./figures/06_urban_rural_gap_trends.png", dpi=300)
plt.close()

print("🖼  Figures saved to figures/:")
print("   - 01_urban_rural_heatmap.png")
print("   - 02_urban_rural_bar.png")
print("   - 03_province_fat_ratio.png (English province names)")
print("   - 04_year_trends.png")
print("   - 05_province_urban_heatmaps.png")
print("   - 06_urban_rural_gap_trends.png")
