# Data Analysis & Machine Learning Portfolio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange.svg" alt="Jupyter">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Pandas-1.3+-purple.svg" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0+-red.svg" alt="Scikit-learn">
</p>

## 项目简介

本项目是一个综合性的数据分析与机器学习实战仓库，涵盖数据探索性分析（EDA）、数据可视化、统计建模、机器学习算法应用等多个维度。项目采用 `Python` 生态体系，结合 `Jupyter Notebook` 交互式开发环境，对真实场景数据集进行系统性分析与建模实践。

**核心技术栈：**
- **数据处理：** Pandas, NumPy
- **数据可视化：** Matplotlib, Seaborn, PyEcharts
- **机器学习：** Scikit-learn
- **网络分析：** NetworkX
- **科学计算：** SciPy

---

## 目录结构

```
data-analysis/
├── README.md                                    # 项目说明文档
├── WineDatasets/                                # 红酒数据集综合分析
│   ├── 红酒数据集.ipynb                          # 完整分析 notebook
│   ├── 红酒数据集.py                             # Python 脚本版本
│   └── 红酒统计信息.md                           # 特征字段释义文档
├── 一线城市租房情况分析/                          # 租房市场数据分析
│   ├── 上海.ipynb                               # 上海租房数据分析
│   ├── 上海.html                                # 上海分析结果（静态页面）
│   ├── 北京.ipynb                               # 北京租房数据分析
│   ├── 北京.html                                # 北京分析结果（静态页面）
│   ├── SH_data.csv                              # 上海数据集
│   ├── 链家上海租房数据.csv                       # 链家上海数据源
│   ├── 链家北京租房数据.csv                       # 链家北京数据源
│   └── *.png                                    # 分析可视化图表
├── 基于逻辑回归实现客户流失率预测分析/              # 客户流失预测项目
│   ├── code/客户流失预测模型.ipynb                # 建模 notebook
│   └── data/churn.csv                           # 客户流失数据集
└── 机器学习/                                    # 机器学习理论学习资料
    ├── 机器学习(科学计算库1).pdf
    ├── 机器学习(科学计算库2).pdf
    ├── 机器学习(算法篇1).pdf
    └── 机器学习(算法篇2).pdf
```

---

## 模块详解

### 1. WineDatasets — 红酒数据集综合分析

基于 UCI Machine Learning Repository 的经典红酒数据集（`sklearn.datasets.load_wine`），对意大利同一产区三种不同品种的红酒进行多维度理化特征分析与建模。

**数据集特征：**

| 特征名 | 说明 | 单位 |
|--------|------|------|
| `alcohol` | 酒精含量 | % |
| `malic_acid` | 苹果酸含量 | g/L |
| `ash` | 灰分（矿物质残留） | g/L |
| `alcalinity_of_ash` | 灰分碱度 | — |
| `magnesium` | 镁含量 | mg/L |
| `total_phenols` | 总酚含量 | — |
| `flavanoids` | 类黄酮含量 | — |
| `nonflavanoid_phenols` | 非类黄酮酚类 | — |
| `proanthocyanins` | 原花青素 | — |
| `color_intensity` | 颜色强度 | — |
| `hue` | 色调 | — |
| `od280/od315` | 稀释葡萄酒吸光度比值 | — |
| `proline` | 脯氨酸含量 | mg/L |
| `target` | 类别标签（0/1/2） | — |

**分析内容覆盖：**

- **描述性统计：** 数据概览、基本统计量计算
- **基础可视化：** 散点图、柱状图（横向/纵向）、饼图、极坐标图、3D 线性图/散点图/柱状图
- **高级可视化：** 热力图（特征相关性）、箱线图、小提琴图、Pairplot 特征组合图、密度图、雷达图、漏斗图、环状图
- **网络图分析：** 基于 NetworkX 的特征相关性网络图（Spring 布局、环形布局、圆盘布局）
- **监督学习模型：**
  - 逻辑回归（Logistic Regression）— 二分类决策边界可视化
  - 线性回归（Linear Regression）— 特征间线性关系拟合
  - 决策树（Decision Tree）— 可解释性分类模型
  - 随机森林（Random Forest）— 集成学习分类
  - 支持向量机（SVM）— 最大间隔分类器
  - 神经网络（MLP）— 多层感知机分类
- **无监督学习模型：**
  - K-Means 聚类 — 无标签数据分群

---

### 2. 一线城市租房情况分析

针对上海、北京两大一线城市的租房市场数据进行系统性分析，数据来源为链家网公开租房数据。

**分析维度：**
- 各行政区房源数量分布
- 租金水平统计与区域对比
- 房型结构分析（数量与租金关联）
- 租金分布直方图
- 房屋面积与租金关系散点图

**技术亮点：**
- 基于 PyEcharts 生成交互式 HTML 可视化报告
- 地理区域维度下的多指标对比分析
- 数据清洗与预处理流程

---

### 3. 基于逻辑回归实现客户流失率预测分析

电信/订阅类业务场景下的客户流失预测实战项目，通过历史客户数据构建二分类模型，识别高风险流失客户。

**项目流程：**
1. **数据加载与探索：** 理解字段含义、缺失值检测
2. **特征工程：** 类别变量编码、特征缩放
3. **模型构建：** 逻辑回归分类器训练
4. **模型评估：** 准确率、精确率、召回率、F1-Score、ROC-AUC
5. **结果解释：** 系数分析、特征重要性排序

**数据集字段：** 客户 demographics 信息、订阅服务、账户信息、消费行为等

---

### 4. 机器学习 — 理论学习资料

收录机器学习系统性学习文档，覆盖两大核心板块：

| 资料名称 | 内容说明 |
|---------|---------|
| `机器学习(科学计算库1).pdf` | NumPy 数组运算、Pandas 数据处理基础 |
| `机器学习(科学计算库2).pdf` | Matplotlib/Seaborn 可视化进阶、数据清洗技巧 |
| `机器学习(算法篇1).pdf` | 监督学习算法：线性回归、逻辑回归、决策树、SVM、KNN |
| `机器学习(算法篇2).pdf` | 集成学习、聚类算法、降维、模型评估与调优 |

---

## 环境依赖

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pyecharts networkx scipy
```

**推荐 Python 版本：** 3.8+

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/xuanz54/data-analysis.git
cd data-analysis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Jupyter Notebook
jupyter notebook

# 4. 打开对应模块的 .ipynb 文件即可运行
```

---

## 贡献与反馈

欢迎通过 Issue 或 Pull Request 提出改进建议。如有疑问，请在 Issues 区留言讨论。

---

## 许可协议

本项目采用 MIT License 开源协议。
