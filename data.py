"""
CHNS膳食数据清洗与多任务标签生成类
用于机器学习多任务分类分析的数据预处理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class CHNSDietDataCleaner:
    """
    CHNS膳食数据清洗类

    功能：
    - 支持多种格式读取（SAS7BDAT、CSV等）
    - 自动识别膳食变量、目标变量和特征变量
    - 省份代码T1转换为实际地名
    - 缺失值处理（按变量类型删除）
    - 异常值处理（基于分位数的截断或删除）
    - 派生变量计算（营养素供能比等）
    - 多任务标签生成（膳食模式聚类、城乡、高能量）
    - 数据质量报告生成
    """

    # 默认变量配置 - 根据PDF文档修正变量名
    DEFAULT_DIET_COLS = ['D3KCAL', 'D3CARBO', 'D3FAT', 'D3PROTN']
    DEFAULT_TARGET_COLS = ['T2']  # 城乡类别: 1=城市, 2=农村
    DEFAULT_FEATURE_COLS = ['WAVE', 'T1', 'T5']  # 年份、省份、家庭编号

    # CHNS省份代码映射（根据官方文档）
    PROVINCE_MAP = {
        11: '北京',
        21: '辽宁',
        23: '黑龙江',
        31: '上海',
        32: '江苏',
        37: '山东',
        41: '河南',
        42: '湖北',
        43: '湖南',
        45: '广西',
        52: '贵州',
        55: '重庆'
    }

    def __init__(
        self,
        diet_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        outlier_method: str = 'remove',  # 'remove' 或 'winsorize'
        outlier_bounds: Tuple[float, float] = (0.01, 0.99),
        n_diet_patterns: int = 3,  # 膳食模式聚类数
        verbose: bool = True
    ):
        """
        初始化清洗器

        Args:
            diet_cols: 膳食变量列表
            target_cols: 目标变量列表
            feature_cols: 特征变量列表
            outlier_method: 异常值处理方法
            outlier_bounds: 异常值边界分位数
            n_diet_patterns: 膳食模式聚类数量
            verbose: 是否打印详细日志
        """
        self.diet_cols = diet_cols or self.DEFAULT_DIET_COLS.copy()
        self.target_cols = target_cols or self.DEFAULT_TARGET_COLS.copy()
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS.copy()
        self.outlier_method = outlier_method
        self.outlier_bounds = outlier_bounds
        self.n_diet_patterns = n_diet_patterns
        self.verbose = verbose

        # 运行时状态
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.processing_log: List[Dict] = []
        self.variable_mapping: Dict[str, str] = {}
        self.scaler = StandardScaler()

    def _log(self, message: str, level: str = 'info') -> None:
        """记录处理日志"""
        if self.verbose:
            prefix = {'info': '✓', 'warning': '⚠', 'error': '✗', 'step': '='}.get(level, '•')
            if level == 'step':
                print(f"\n{prefix * 60}")
                print(message)
                print(f"{prefix * 60}")
            else:
                print(f"{prefix} {message}")
        self.processing_log.append({'level': level, 'message': message})

    def load_data(self, file_path: Union[str, Path], file_format: Optional[str] = None) -> pd.DataFrame:
        """加载数据文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if file_format is None:
            file_format = file_path.suffix.lower().replace('.', '')

        self._log(f"步骤 1: 读取数据 [{file_format.upper()}格式]", 'step')

        try:
            if file_format in ['sas7bdat', 'sas']:
                df = self._load_sas(file_path)
            elif file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format in ['pkl', 'pickle']:
                df = pd.read_pickle(file_path)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")

            self.raw_data = df.copy()
            self._log(f"成功读取数据，样本量: {len(df):,}，变量数: {len(df.columns)}")
            self._log(f"变量列表: {list(df.columns)}")
            return df

        except Exception as e:
            self._log(f"读取失败: {e}", 'error')
            raise

    def _load_sas(self, file_path: Path) -> pd.DataFrame:
        """加载SAS7BDAT格式文件"""
        try:
            from sas7bdat import SAS7BDAT
            with SAS7BDAT(file_path) as reader:
                return reader.to_data_frame()
        except ImportError:
            pass

        try:
            import pyreadstat
            df, meta = pyreadstat.read_sas7bdat(file_path)
            return df
        except ImportError:
            pass

        try:
            return pd.read_sas(file_path, format='sas7bdat')
        except Exception as e:
            raise ImportError(f"无法读取SAS文件。请安装: pip install sas7bdat 或 pip install pyreadstat\n错误: {e}")

    def identify_variables(self, df: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
        """识别并验证关键变量"""
        if df is None:
            df = self.raw_data

        self._log("步骤 2: 识别关键变量", 'step')

        # 检查变量存在性（大小写不敏感）
        df_cols_upper = [c.upper() for c in df.columns]

        available_diet = []
        for col in self.diet_cols:
            if col.upper() in df_cols_upper:
                actual_col = df.columns[df_cols_upper.index(col.upper())]
                available_diet.append(actual_col)
                if actual_col != col:
                    self.variable_mapping[col] = actual_col

        available_target = []
        for col in self.target_cols:
            if col.upper() in df_cols_upper:
                actual_col = df.columns[df_cols_upper.index(col.upper())]
                available_target.append(actual_col)

        available_features = []
        for col in self.feature_cols:
            if col.upper() in df_cols_upper:
                actual_col = df.columns[df_cols_upper.index(col.upper())]
                available_features.append(actual_col)

        self.available_vars = {
            'diet': available_diet,
            'target': available_target,
            'features': available_features
        }

        self._log(f"膳食变量: {available_diet}")
        self._log(f"目标变量: {available_target} (T2: 1=城市, 2=农村)")
        self._log(f"特征变量: {available_features}")

        if not available_diet:
            self._log("警告: 未找到膳食变量，请检查变量名", 'error')

        return self.available_vars

    def map_province_names(self, df: Optional[pd.DataFrame] = None,
                          province_col: str = 'T1') -> pd.DataFrame:
        """
        将省份代码T1转换为实际省份名称

        Args:
            df: 输入数据框
            province_col: 省份代码列名

        Returns:
            添加省份名称列的数据框
        """
        if df is None:
            df = self.raw_data.copy() if self.raw_data is not None else self.cleaned_data.copy()

        # 查找实际的省份列名（大小写不敏感）
        actual_col = None
        for col in df.columns:
            if col.upper() == province_col.upper():
                actual_col = col
                break

        if actual_col is None:
            self._log(f"未找到省份列 {province_col}", 'warning')
            return df

        self._log("步骤 2.5: 省份代码转换为地名", 'step')

        # 创建省份名称列
        df['province_name'] = df[actual_col].map(self.PROVINCE_MAP)

        # 检查未匹配的代码
        unmatched = df[df['province_name'].isna() & df[actual_col].notna()][actual_col].unique()
        if len(unmatched) > 0:
            self._log(f"警告: 以下省份代码未匹配: {unmatched}", 'warning')

        self._log(f"省份分布:")
        prov_dist = df['province_name'].value_counts()
        for prov, count in prov_dist.items():
            self._log(f"  {prov}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def handle_missing_values(self, df: Optional[pd.DataFrame] = None,
                            drop_diet_missing: bool = True,
                            drop_target_missing: bool = True) -> pd.DataFrame:
        """处理缺失值"""
        if df is None:
            df = self.raw_data.copy()

        self._log("步骤 3: 缺失值处理", 'step')

        all_key_vars = (self.available_vars['diet'] +
                       self.available_vars['target'] +
                       self.available_vars['features'])

        missing_stats = df[all_key_vars].isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]

        if len(missing_cols) > 0:
            self._log("缺失值统计:")
            for col, count in missing_cols.items():
                self._log(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            self._log("关键变量无缺失值")

        # 删除膳食变量缺失
        if drop_diet_missing and self.available_vars['diet']:
            before = len(df)
            df = df.dropna(subset=self.available_vars['diet'])
            after = len(df)
            if before != after:
                self._log(f"删除膳食数据缺失样本: {before:,} → {after:,} (删除 {before-after:,} 条)")

        # 删除目标变量缺失
        if drop_target_missing and self.available_vars['target']:
            before = len(df)
            df = df.dropna(subset=self.available_vars['target'])
            after = len(df)
            if before != after:
                self._log(f"删除目标变量缺失样本: {before:,} → {after:,} (删除 {before-after:,} 条)")

        return df

    def handle_outliers(self, df: Optional[pd.DataFrame] = None,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """处理异常值"""
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.raw_data.copy()

        if columns is None:
            columns = self.available_vars.get('diet', [])

        if not columns:
            self._log("步骤 4: 跳过异常值处理（无膳食变量）", 'step')
            return df

        self._log("步骤 4: 异常值处理", 'step')

        # 显示描述性统计
        self._log("膳食变量描述性统计:")
        desc = df[columns].describe()
        print(desc)

        lower_q, upper_q = self.outlier_bounds

        for col in columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(lower_q)
            Q99 = df[col].quantile(upper_q)
            before_count = len(df)

            outliers = (df[col] < Q1) | (df[col] > Q99)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                if self.outlier_method == 'remove':
                    df = df[~outliers]
                    action = "删除"
                else:
                    df[col] = df[col].clip(lower=Q1, upper=Q99)
                    action = "缩尾处理"

                self._log(f"  {col}: {action} {outlier_count:,} 个极端值 "
                         f"({outlier_count/before_count*100:.2f}%)")

        self._log(f"异常值处理后样本量: {len(df):,}")

        # 检查负值和零值
        self._log("检查负值和零值:")
        for col in columns:
            negative = (df[col] < 0).sum()
            zero = (df[col] == 0).sum()
            if negative > 0:
                self._log(f"  警告: {col} 有 {negative} 个负值", 'warning')
            if zero > 0:
                self._log(f"  注意: {col} 有 {zero} 个零值")

        return df

    def create_derived_variables(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """创建派生变量"""
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.raw_data.copy()

        self._log("步骤 5: 创建派生变量", 'step')

        # 获取实际列名（考虑大小写）
        col_map = {c.upper(): c for c in df.columns}

        kcal_col = col_map.get('D3KCAL')
        carbo_col = col_map.get('D3CARBO')
        fat_col = col_map.get('D3FAT')
        protn_col = col_map.get('D3PROTN')

        required = [kcal_col, carbo_col, fat_col, protn_col]

        if all(col is not None for col in required):
            # 避免除以零
            df['carb_ratio'] = np.where(df[kcal_col] > 0,
                                        (df[carbo_col] * 4) / df[kcal_col] * 100, 0)
            df['fat_ratio'] = np.where(df[kcal_col] > 0,
                                       (df[fat_col] * 9) / df[kcal_col] * 100, 0)
            df['prot_ratio'] = np.where(df[kcal_col] > 0,
                                        (df[protn_col] * 4) / df[kcal_col] * 100, 0)

            self._log("已创建营养素供能比变量:")
            self._log(f"  - carb_ratio (碳水供能比): 均值 {df['carb_ratio'].mean():.1f}%")
            self._log(f"  - fat_ratio (脂肪供能比): 均值 {df['fat_ratio'].mean():.1f}%")
            self._log(f"  - prot_ratio (蛋白供能比): 均值 {df['prot_ratio'].mean():.1f}%")
        else:
            missing = [col for col in required if col is None]
            self._log(f"无法计算供能比，缺少变量: {missing}", 'warning')

        return df

    def generate_multitask_labels(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        生成多任务学习标签

        Task 1: 膳食模式类型（基于D3FAT, D3CARBO, D3PROTN聚类）
        Task 2: 城乡类别（直接使用T2: 1=城市, 2=农村）
        Task 3: 高能量摄入（D3KCAL按三分位数划分: 高vs非高）
        """
        if df is None:
            df = self.cleaned_data.copy()

        self._log("步骤 5.5: 生成多任务标签", 'step')

        col_map = {c.upper(): c for c in df.columns}

        # ========== Task 1: 膳食模式聚类 ==========
        self._log("Task 1: 生成膳食模式标签（基于营养素聚类）...")

        # 使用三大营养素进行聚类
        nutrient_cols = [col_map.get(c) for c in ['D3FAT', 'D3CARBO', 'D3PROTN']]
        nutrient_cols = [c for c in nutrient_cols if c is not None]

        if len(nutrient_cols) == 3:
            # 标准化
            nutrients_scaled = self.scaler.fit_transform(df[nutrient_cols])

            # K-Means聚类
            kmeans = KMeans(n_clusters=self.n_diet_patterns, random_state=42, n_init=10)
            df['diet_pattern'] = kmeans.fit_predict(nutrients_scaled)

            # 分析各簇特征
            self._log(f"  聚类完成，生成 {self.n_diet_patterns} 类膳食模式:")
            for i in range(self.n_diet_patterns):
                cluster_data = df[df['diet_pattern'] == i]
                self._log(f"    模式 {i}: n={len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)")
                self._log(f"      脂肪: {cluster_data[nutrient_cols[0]].mean():.1f}g, "
                         f"碳水: {cluster_data[nutrient_cols[1]].mean():.1f}g, "
                         f"蛋白: {cluster_data[nutrient_cols[2]].mean():.1f}g")
        else:
            self._log(f"  警告: 缺少营养素变量，跳过膳食模式聚类", 'warning')

        # ========== Task 2: 城乡类别 ==========
        self._log("Task 2: 城乡类别标签...")
        t2_col = col_map.get('T2')
        if t2_col:
            df['urban_rural'] = df[t2_col].map({1: '城市', 2: '农村'})
            urban_dist = df['urban_rural'].value_counts()
            self._log(f"  城乡分布: {urban_dist.to_dict()}")

        # ========== Task 3: 高能量摄入 ==========
        self._log("Task 3: 高能量摄入标签（三分位数划分）...")
        kcal_col = col_map.get('D3KCAL')
        if kcal_col:
            # 计算三分位数
            q33 = df[kcal_col].quantile(0.33)
            q67 = df[kcal_col].quantile(0.67)

            # 高能量 = 上三分之一
            df['high_energy'] = (df[kcal_col] > q67).astype(int)
            df['energy_level'] = pd.cut(df[kcal_col],
                                        bins=[-np.inf, q33, q67, np.inf],
                                        labels=['低能量', '中能量', '高能量'])

            self._log(f"  能量三分位数: 低(<{q33:.0f}), 中({q33:.0f}-{q67:.0f}), 高(>{q67:.0f})")
            energy_dist = df['energy_level'].value_counts().sort_index()
            self._log(f"  能量分布: {energy_dist.to_dict()}")

        return df

    def run_pipeline(self, file_path: Union[str, Path],
                    output_dir: Optional[Union[str, Path]] = None,
                    save_format: List[str] = ['csv', 'pkl']) -> pd.DataFrame:
        """执行完整清洗流程"""
        # 1. 加载数据
        df = self.load_data(file_path)

        # 2. 识别变量
        self.identify_variables(df)

        # 2.5. 省份名称映射
        df = self.map_province_names(df)

        # 3. 处理缺失值
        df = self.handle_missing_values(df)

        # 4. 处理异常值
        df = self.handle_outliers(df)

        # 5. 创建派生变量
        df = self.create_derived_variables(df)

        # 5.5. 生成多任务标签
        df = self.generate_multitask_labels(df)

        self.cleaned_data = df

        # 6. 保存数据
        if output_dir:
            self.save_data(df, output_dir, save_format)

        # 7. 生成报告
        self.generate_report()

        return df

    def save_data(self, df: Optional[pd.DataFrame] = None,
                 output_dir: Union[str, Path] = './data',
                 formats: List[str] = ['csv', 'pkl'],
                 filename_prefix: str = 'c12diet_cleaned') -> Dict[str, Path]:
        """保存清洗后的数据"""
        if df is None:
            df = self.cleaned_data

        self._log("步骤 6: 保存清洗后数据", 'step')

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 选择需要保存的变量（包括新生成的标签）
        base_vars = (self.available_vars.get('diet', []) +
                    self.available_vars.get('target', []) +
                    self.available_vars.get('features', []))

        extra_vars = ['province_name', 'carb_ratio', 'fat_ratio', 'prot_ratio',
                     'diet_pattern', 'urban_rural', 'high_energy', 'energy_level']

        vars_to_save = [v for v in base_vars + extra_vars if v in df.columns]

        for fmt in formats:
            if fmt == 'csv':
                path = output_dir / f"{filename_prefix}.csv"
                df[vars_to_save].to_csv(path, index=False, encoding='utf-8-sig')
                saved_files['csv'] = path
                self._log(f"清洗后数据已保存至: {path}")

            elif fmt in ['pkl', 'pickle']:
                path = output_dir / f"{filename_prefix}.pkl"
                df.to_pickle(path)
                saved_files['pkl'] = path
                self._log(f"完整数据已保存至: {path}")

        self._log(f"最终样本量: {len(df):,}，保存变量数: {len(vars_to_save)}")
        return saved_files

    def generate_report(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """生成数据质量报告"""
        if df is None:
            df = self.cleaned_data

        self._log("步骤 7: 数据质量报告", 'step')

        print("\n最终数据描述性统计:")
        diet_vars = self.available_vars.get('diet', [])
        if diet_vars:
            print(df[diet_vars].describe())

        # 多任务标签分布
        print("\n多任务标签分布:")
        if 'diet_pattern' in df.columns:
            print(f"膳食模式: {df['diet_pattern'].value_counts().to_dict()}")
        if 'urban_rural' in df.columns:
            print(f"城乡类别: {df['urban_rural'].value_counts().to_dict()}")
        if 'high_energy' in df.columns:
            print(f"高能量摄入: {df['high_energy'].value_counts().to_dict()}")

        print(f"\n数据完整性:")
        all_vars = (self.available_vars.get('diet', []) +
                   self.available_vars.get('target', []) +
                   self.available_vars.get('features', []))
        completeness = {}
        for col in all_vars:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df) * 100
                completeness[col] = missing_pct
                print(f"  {col}: {missing_pct:.2f}% 缺失")

        report = {
            'sample_size': len(df),
            'variables': len(df.columns),
            'completeness': completeness,
            'processing_log': self.processing_log
        }

        self._log("数据清洗完成！", 'step')
        return report


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    '''
    # 初始化清洗器（3类膳食模式）
    cleaner = CHNSDietDataCleaner(
        n_diet_patterns=3,
        outlier_method='remove',
        verbose=True
    )

    # 运行完整流程
    df_clean = cleaner.run_pipeline(
        './data/c12diet.sas7bdat',
        output_dir='./data'
    )
    '''

    import pandas as pd

    # 读取清洗后的数据
    df = pd.read_csv('./data/c12diet_cleaned.csv')

    # ========== Task 1: 膳食模式 ==========
    print("=" * 50)
    print("Task 1: 膳食模式分布")
    print(df['diet_pattern'].value_counts().sort_index())

    # 查看各模式的营养素特征
    print("\n各膳食模式营养素均值:")
    print(df.groupby('diet_pattern')[['d3fat', 'd3carbo', 'd3protn']].mean())

    # ========== Task 2: 城乡类别 ==========
    print("\n" + "=" * 50)
    print("Task 2: 城乡分布")
    print(df['urban_rural'].value_counts())

    # 交叉分析：城乡 vs 膳食模式
    print("\n城乡 vs 膳食模式 交叉表:")
    print(pd.crosstab(df['urban_rural'], df['diet_pattern']))

    # ========== Task 3: 高能量摄入 ==========
    print("\n" + "=" * 50)
    print("Task 3: 能量摄入分布")
    print(df['energy_level'].value_counts())
    print("\n高能量摄入比例:")
    print(df['high_energy'].value_counts(normalize=True))

    # 交叉分析：高能量 vs 膳食模式
    print("\n高能量 vs 膳食模式 交叉表:")
    print(pd.crosstab(df['high_energy'], df['diet_pattern']))