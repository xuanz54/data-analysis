# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from mpl_toolkits.mplot3d import Axes3D
# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载红酒数据集
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df.head()
# 数据导出到本地excel文件
df.to_excel('data.xlsx', index=False)
# 基本统计信息
df.describe()

# 绘制酒精含量与灰分的散点图
plt.figure(figsize=(8, 6))
plt.scatter(df['alcohol'], df['ash'], alpha=0.7, color='blue')
plt.title('酒精含量 vs 灰分')
plt.xlabel('酒精含量')
plt.ylabel('灰分')
plt.show()

# 绘制酒精含量、灰分、颜色强度的泡状图
plt.figure(figsize=(8, 6))
plt.scatter(df['alcohol'], df['ash'], s=df['color_intensity']*100, alpha=0.6, color='green')
plt.title('酒精含量 vs 灰分（泡状图）')
plt.xlabel('酒精含量')
plt.ylabel('灰分')
plt.show()

# 绘制酒精含量的竖向柱状图
plt.figure(figsize=(8, 6))
df.groupby('target')['alcohol'].mean().plot(kind='bar', color='skyblue')
plt.title('不同类别的酒精含量平均值')
plt.xlabel('酒的类别')
plt.ylabel('酒精含量')
plt.xticks(rotation=0)
plt.show()

# 绘制酒精含量的横向柱状图
plt.figure(figsize=(8, 6))
df.groupby('target')['alcohol'].mean().plot(kind='barh', color='purple')
plt.title('不同类别的酒精含量平均值')
plt.xlabel('酒精含量')
plt.ylabel('酒的类别')
plt.show()

# 准备数据
values = df['alcohol'].values
categories = df['target'].values
# 计算每个类别的平均值
mean_values = df.groupby('target')['alcohol'].mean()
# 设置极坐标图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
# 创建角度
theta = np.linspace(0.0, 2 * np.pi, len(mean_values), endpoint=False)
# 将数据添加到图中，并使柱状图闭合
bars = ax.bar(theta, mean_values, width=0.3, bottom=0.0, color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
# 添加标签
ax.set_xticks(theta)
ax.set_xticklabels(['类别 0', '类别 1', '类别 2'])
ax.set_yticklabels([])
# 添加图例
plt.legend(bars, ['类别 0', '类别 1', '类别 2'], loc='upper right')
# 设置标题
ax.set_title('酒精含量的极坐标柱状图')
plt.show()

# 饼图 - 不同类别酒精含量总和的分布（带有数据信息）
df_grouped_sum = df.groupby('target').agg(total_alcohol=('alcohol', 'sum'))
plt.figure(figsize=(8, 6))
plt.pie(df_grouped_sum['total_alcohol'], labels=df_grouped_sum.index, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
plt.title('不同类别酒精含量总和的分布')
plt.show()

# 计算不同类型的酒在酒精含量上的平均值
alcohol_means = df.groupby('target')['alcohol'].mean()
# 设置标签
labels = ['酒类型 0', '酒类型 1', '酒类型 2']
# 设置explode参数，让其中一个部分突出显示
explode = (0.1, 0, 0)
# 绘制饼状图
plt.figure(figsize=(8, 6))
plt.pie(
    alcohol_means,
    explode=explode,  # 突出显示酒类型 0
    labels=labels,
    autopct='%1.1f%%',  # 百分比显示
    shadow=True,
    startangle=90,  # 开始角度
    colors=['#ff9999', '#66b3ff', '#99ff99']  # 配色
)
# 设置标题
plt.title('不同酒类型在酒精含量上的占比')
# 显示图形
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
# 选取数据进行3D线性图
x = df.index
y = df['alcohol']
z = df['malic_acid']
ax.plot3D(x, y, z, color='green')
ax.set_xlabel('样本编号')
ax.set_ylabel('酒精含量')
ax.set_zlabel('苹果酸', labelpad=0)  # 调整z轴标签与轴线的距离
plt.title('立体线形图：酒精含量与苹果酸')
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
# 绘制散点图，并使用颜色映射来表示'target'列的值
scatter = ax.scatter(df['alcohol'], df['malic_acid'], df['color_intensity'], c=df['target'], cmap='viridis')
# 设置坐标轴标签
ax.set_xlabel('酒精含量')
ax.set_ylabel('苹果酸')
ax.set_zlabel('颜色强度')
# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('目标类别')
plt.title('3D 散点图：酒精含量、苹果酸和颜色强度')
plt.show()

# 立体柱状图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(df['alcohol'].count())
y = df['alcohol']
z = np.zeros_like(x)
dx = dy = np.ones_like(x)
dz = df['alcohol']
ax.bar3d(x, y, z, dx, dy, dz, color='cyan')
ax.set_xlabel('样本编号')
ax.set_ylabel('酒精含量',labelpad=10)
ax.set_zlabel('值')
plt.title('立体柱状图：酒精含量分布')
plt.show()

# 热力图
plt.figure(figsize=(10, 8))
corr = df.drop(columns='target').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热力图')
plt.show()

plt.figure(figsize=(8, 6))
# 绘制箱线图，并将'target'列作为hue参数传入
sns.boxplot(x='target', y='alcohol', data=df, hue='target', palette='Set2', legend=False)
# 添加自定义图例
categories = df['target'].unique()  # 获取唯一的类别
colors = sns.color_palette('Set2', len(categories))  # 获取颜色
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=f'{cat}') for cat, c in zip(categories, colors)]
plt.legend(handles=legend_patches, title='酒的类别')
plt.title('每类酒的酒精含量分布（箱线图）')
plt.xlabel('酒的类别')
plt.ylabel('酒精含量')
plt.show()

plt.figure(figsize=(8, 6))
# 绘制小提琴图，并将'target'列作为hue参数传入
sns.violinplot(x='target', y='alcohol', data=df, hue='target', palette='Set2', legend=False)
# 添加自定义图例
categories = df['target'].unique()  # 获取唯一的类别
colors = sns.color_palette('Set2', len(categories))  # 获取颜色
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=f'{cat}') for cat, c in zip(categories, colors)]
plt.legend(handles=legend_patches, title='酒的类别')
plt.title('每类酒的酒精含量分布（小提琴图）')
plt.xlabel('酒的类别')
plt.ylabel('酒精含量')
plt.show()

plt.figure(figsize=(15, 15))
pairplot = sns.pairplot(df, vars=['alcohol', 'ash', 'color_intensity'], hue='target', palette='coolwarm')
# 调整图表布局，为标题留出空间
plt.subplots_adjust(top=0.9)  # 调整顶部空间，数值越小，留出的空间越大
# 设置标题，并通过pad参数增加与图表的间距
plt.suptitle('酒精含量、灰分和颜色强度的特征组合图', y=1.05, fontsize=16)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 准备数据
X = df[['alcohol', 'ash']]
y = df['target']
# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# 训练逻辑回归模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# 绘制散点图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
# 创建网格用于显示决策边界
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 200),
                     np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 200))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
# 添加颜色条
plt.colorbar(scatter, label='目标类别')
plt.title('逻辑回归模型的散点分布图')
plt.xlabel('酒精含量（标准化）')
plt.ylabel('灰分（标准化）')
plt.show()

from sklearn.linear_model import LinearRegression
# 准备数据
X = df[['alcohol']]
y = df['malic_acid']
# 训练线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# 绘制散点图和回归线
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.6)
plt.plot(X, lin_reg.predict(X), color='red', linewidth=2)
plt.title('酒精含量与苹果酸的线性回归模型')
plt.xlabel('酒精含量')
plt.ylabel('苹果酸')
plt.show()

import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
# 准备数据
X = df[['alcohol', 'malic_acid']]
# 训练 K-Means 聚类模型
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
# 绘制聚类图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['alcohol'], df['malic_acid'], c=df['cluster'], cmap='rainbow', alpha=0.7)
# 创建图例
n_clusters = len(kmeans.cluster_centers_)
# 使用新的matplotlib颜色映射方法
cmap = mcolors.ListedColormap(plt.cm.rainbow(np.linspace(0, 1, n_clusters)))
colors = cmap(np.arange(n_clusters))  # 获取每个聚类的颜色
labels = [f'Cluster {i}' for i in range(n_clusters)]  # 创建聚类标签
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(n_clusters)]
plt.legend(handles=legend_handles, labels=labels, title='Cluster')
plt.title('酒精含量与苹果酸的聚类模型')
plt.xlabel('酒精含量')
plt.ylabel('苹果酸')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
# 准备数据
X = df[['alcohol', 'malic_acid']]
y = df['target']
# 训练决策树模型
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)
# 绘制决策树
plt.figure(figsize=(12, 8))
plot_tree(tree_clf, filled=True, feature_names=['alcohol', 'malic_acid'], class_names=wine.target_names)
plt.title('决策树模型')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
# 准备数据
X = df[['alcohol', 'malic_acid']]
y = df['target']
# 训练随机森林模型，包含两棵树
rf_clf = RandomForestClassifier(n_estimators=2, random_state=42)  
rf_clf.fit(X, y)
# 绘制两棵决策树
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15), dpi=300)  # 设置图形大小和分辨率
# 选择2棵树进行绘制
for i in range(2):
    plot_tree(rf_clf.estimators_[i], feature_names=['alcohol', 'malic_acid'], class_names=wine.target_names, filled=True, ax=axes[i])
    axes[i].set_title(f'决策树 {i+1}')
plt.show()

import networkx as nx
# 计算相关性矩阵
corr_matrix = df.drop(columns='target').corr()
# 创建网络图
G = nx.Graph()
# 添加节点
for feature in corr_matrix.columns:
    G.add_node(feature)
# 添加边（相关性绝对值大于0.5的）
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
# 绘制网络图
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42, k=0.4)  # 调整布局参数k
weights = [G[u][v]['weight'] for u, v in G.edges()]
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
# 绘制边
nx.draw_networkx_edges(G, pos, width=[abs(w) * 5 for w in weights], alpha=0.5, edge_color='grey')
# 绘制标签
labels = {node: node if len(node) < 10 else node[:10] + '...' for node in G.nodes()}  # 缩短标签
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
plt.title('特征相关性网络图')
plt.show()

# 绘制圆盘状网络图
plt.figure(figsize=(12, 8))
pos = nx.circular_layout(G)  # 使用圆盘状布局
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
# 绘制边
nx.draw_networkx_edges(G, pos, width=[abs(G[u][v]['weight']) * 5 for u, v in G.edges()], alpha=0.5, edge_color='grey')
# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.title('圆盘状网络图')
plt.show()

# 计算相关性矩阵
corr_matrix = df.drop(columns='target').corr()
# 创建网络图
G = nx.Graph()
# 添加节点及其属性
for feature in corr_matrix.columns:
    G.add_node(feature, value=corr_matrix[feature].mean())  # 使用平均相关性作为节点属性
# 添加边（相关性绝对值大于0.5的）
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
# 获取节点属性
node_values = nx.get_node_attributes(G, 'value')
# 绘制网络图
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42, k=0.4)
# 绘制节点，颜色基于属性值
node_colors = [node_values[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, cmap='viridis', alpha=0.9)
# 绘制边，宽度基于权重
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=[abs(w) * 5 for w in edge_weights], alpha=0.5, edge_color='grey')
# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
# 添加色条
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
fig = plt.gcf()  # 获取当前图形
cbar = fig.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('节点属性值')
plt.title('含属性及边的颜色及粗度表达的网络图')
plt.show()

# 计算相关性矩阵
corr_matrix = df.drop(columns='target').corr()
# 创建网络图
G = nx.Graph()
# 添加节点及其属性
for feature in corr_matrix.columns:
    G.add_node(feature, value=corr_matrix[feature].mean())  # 使用平均相关性作为节点属性
# 添加边（相关性绝对值大于0.5的）
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
# 获取节点属性
node_values = nx.get_node_attributes(G, 'value')
# 绘制网络图
plt.figure(figsize=(12, 10))
pos = nx.circular_layout(G)  # 使用圆形布局
# 绘制节点，颜色基于属性值
node_colors = [node_values[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, cmap='viridis', alpha=0.9)
# 绘制边，宽度基于权重
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=[abs(w) * 5 for w in edge_weights], alpha=0.5, edge_color='grey')
# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
# 添加色条
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
fig = plt.gcf()  # 获取当前图形
# 创建一个新的轴来放置色条
cbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('节点属性值')
plt.title('环形布局的网络图')
plt.show()

from pyecharts.charts import Bar
from pyecharts import options as opts
# 数据准备
df_stacked = df.groupby('target').agg({
    'alcohol': 'sum',
    'malic_acid': 'sum',
    'color_intensity': 'sum'
}).T
# 初始化图表
bar = Bar()
for feature in df_stacked.index:
    bar.add_xaxis(df_stacked.columns.tolist())
    bar.add_yaxis(feature, df_stacked.loc[feature].tolist(), stack='stack1')
# 配置选项并渲染
bar.set_global_opts(
    title_opts=opts.TitleOpts(title="堆积柱状图：各类别特征总和"),
    xaxis_opts=opts.AxisOpts(name="特征"),
    yaxis_opts=opts.AxisOpts(name="总和"),
    legend_opts=opts.LegendOpts(pos_top="5%")
)
bar.load_javascript()
bar.render_notebook()

from scipy import stats
# 创建密度图
x = np.linspace(df['alcohol'].min(), df['alcohol'].max(), 100)
kde_alcohol = stats.gaussian_kde(df['alcohol'])
kde_malic_acid = stats.gaussian_kde(df['malic_acid'])
plt.figure(figsize=(10, 6))
plt.plot(x, kde_alcohol(x), label="酒精含量", color='blue')
plt.plot(x, kde_malic_acid(x), label="苹果酸", color='green')
plt.title("密度图：酒精含量与苹果酸")
plt.xlabel("值")
plt.ylabel("密度")
plt.legend()
plt.show()

from pyecharts import options as opts
from pyecharts.charts import Radar
# 准备数据
features = df.columns.tolist()
values = df.loc[0, features].tolist()
# 创建雷达图
radar = Radar()
radar.add_schema(
    schema=[opts.RadarIndicatorItem(name=feature, max_=df[feature].max()) for feature in features],
    shape='circle'
)
radar.add(
    series_name='样本 0',
    data=[values]
)
radar.set_global_opts(
    title_opts=opts.TitleOpts(title='样本 0 的特征雷达图')
)
# 渲染图表
radar.render_notebook()


from pyecharts import options as opts
from pyecharts.charts import Funnel
# 创建酒精含量的范围并计算每个范围的数量
bins = [10, 12, 14, 16, 18]  # 酒精含量的范围
labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
df['alcohol_bins'] = pd.cut(df['alcohol'], bins=bins, labels=labels, include_lowest=True)
funnel_data = df['alcohol_bins'].value_counts().sort_index()
# 创建漏斗图
funnel = Funnel()
funnel.add(
    series_name="酒精含量分布",
    data_pair=[(label, count) for label, count in funnel_data.items()],
    sort_='descending'
)
funnel.set_global_opts(
    title_opts=opts.TitleOpts(title="酒精含量漏斗图"),
    tooltip_opts=opts.TooltipOpts(trigger='item', formatter='{b}: {c}')
)
# 渲染图表
funnel.render_notebook()

from pyecharts import options as opts
from pyecharts.charts import Pie
# 创建酒精含量的范围并计算每个范围的数量
bins = [10, 12, 14, 16, 18]  # 酒精含量的范围
labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
df['alcohol_bins'] = pd.cut(df['alcohol'], bins=bins, labels=labels, include_lowest=True)
pie_data = df['alcohol_bins'].value_counts()
# 创建饼图
pie = Pie()
pie.add(
    series_name="酒精含量分布",
    data_pair=[(label, count) for label, count in pie_data.items()],
    radius=["30%", "75%"],  # 饼图的半径
    label_opts=opts.LabelOpts(formatter="{b}: {d}%")  # 标签显示为百分比
)
pie.set_global_opts(
    title_opts=opts.TitleOpts(title="酒精含量环状图"),
    tooltip_opts=opts.TooltipOpts(trigger='item', formatter='{b}: {c} ({d}%)')
)
# 渲染图表
pie.render_notebook()

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = wine.data[:, :2]  # 选择前两个特征
y = wine.target
# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 训练神经网络模型
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
# 创建网格数据
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
# 计算预测
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 绘制分类边界
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
# 绘制数据点
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
# 添加图例
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
ax.legend(handles, wine.target_names, title="类别", loc="best")
# 设置标签和标题
ax.set_xlabel('特征 1')
ax.set_ylabel('特征 2')
ax.set_title('神经网络分类边界')
# 添加颜色条
plt.colorbar(contour, ax=ax, label='预测类别')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# 选择前两个特征（酒精含量和苹果酸）
X = df[['alcohol', 'malic_acid']].values
y = df['target'].values
# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# 创建SVM模型
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
# 预测测试集
y_pred = svm_model.predict(X_test)
print(f"SVM 准确率: {accuracy_score(y_test, y_pred):.2f}")
# 绘制决策边界
def plot_decision_boundary(X, y, model):
    h = .02  # 网格中的步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('酒精含量 (标准化)')
    plt.ylabel('苹果酸 (标准化)')
    plt.title('支持向量机分类边界')
    plt.show()
# 绘制SVM决策边界
plot_decision_boundary(X_scaled, y, svm_model)