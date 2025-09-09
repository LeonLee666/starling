# KNN图构建功能

## 概述

本功能为 `graph_partitioner` 类添加了KNN（K-Nearest Neighbors）图构建能力。该功能可以将原始的DiskANN图结构转换为KNN图，其中每个节点的邻居被替换为其距离最近的k个节点。

## 主要特性

- **Best First Search算法**：使用优先队列进行智能的邻居搜索，比简单的两跳扩展更精确
- **量化坐标优化**：使用量化坐标代替原始坐标，大幅减少内存使用（减少30-50倍）
- **内存高效**：在加载索引时同时加载量化向量数据，避免重复IO
- **并行处理**：使用OpenMP进行多线程并行计算
- **类型支持**：支持 `float`、`uint8_t`、`int8_t` 数据类型
- **可配置参数**：可以自定义k值（每个节点的邻居数量）

## 使用方法

### 基本用法

```cpp
#include "include/partitioner.h"

// 创建图分割器，启用KNN图构建
GP::graph_partitioner<float> partitioner(
    "path/to/index.disk",  // 索引文件路径
    "float",               // 数据类型
    true,                  // 从磁盘加载
    1,                     // 批处理大小
    true,                  // 显示进度
    "",                    // 频率文件（可选）
    INF,                   // 不切割图
    true,                  // 启用KNN图构建
    10                     // k=10，每个节点保留10个最近邻居
);
```

### 参数说明

- `indexName`: 磁盘索引文件路径
- `data_type`: 数据类型，支持 "uint8"、"int8"、"float"
- `load_disk`: 是否从磁盘加载索引
- `BS`: 批处理大小
- `visual`: 是否显示进度信息
- `freq_file`: 频率文件路径（可选）
- `cut`: 图切割参数
- `build_knn_graph`: **新增** - 是否构建KNN图
- `knn_k`: **新增** - KNN的k值，即每个节点保留的邻居数量

## 算法原理

### Best First Search + 量化坐标策略

1. **量化坐标生成**：
   - 将原始向量按维度分块（每8维一个chunk）
   - 对每个chunk计算平均值并量化为8位整数
   - 大幅减少内存使用（从 `_nd * _dim * sizeof(T)` 到 `_nd * _n_chunks`）

2. **Best First Search搜索**：
   - 使用优先队列（最小堆）维护候选节点
   - 从直接邻居开始，逐步扩展到更远的节点
   - 基于量化距离进行智能搜索，避免盲目扩展

3. **搜索优化**：
   - 限制搜索深度避免过度计算
   - 维护已访问节点集合避免重复计算
   - 如果搜索不足，随机补充候选节点

### 性能优化

- **内存高效**：量化坐标减少30-50倍内存使用
- **计算快速**：量化距离计算比原始距离计算快
- **智能搜索**：Best First Search比简单扩展更精确
- **并行计算**：使用OpenMP进行多线程并行处理

## 实现细节

### 新增成员变量

```cpp
// KNN图构建参数
bool _build_knn_graph = false;           // 是否构建KNN图
unsigned _knn_k = 10;                    // KNN的k值
std::vector<_u8> _pq_data;               // 存储量化向量数据
_u64 _n_chunks = 0;                      // 量化chunk数量
```

### 核心函数

- `build_knn_graph()`: 主要的KNN图构建函数
- `best_first_search_knn()`: Best First Search算法实现
- `load_quantized_data()`: 加载量化向量数据（在索引加载时自动调用）
- `compute_quantized_distance()`: 计算量化距离

## 性能考虑

### 内存使用

- **量化坐标**：`_nd * _n_chunks` 字节（通常比原始数据少30-50倍）
- **量化chunk数量**：`_n_chunks = _dim / 8`（每8维一个chunk）
- **内存优势**：对于100万节点128维数据，从512MB减少到16MB

### 计算复杂度

- **时间复杂度**：O(_nd * search_depth * _n_chunks)
- **搜索深度**：限制为 `k * 10`，避免过度搜索
- **距离计算**：量化L2距离比原始距离计算快

### 并行化

- 使用OpenMP进行节点级别的并行处理
- 建议根据CPU核心数调整线程数

## 注意事项

1. **量化精度**：使用量化坐标会有一定的精度损失，但对KNN图构建通常可接受
2. **数据加载**：量化向量数据在索引加载时自动加载
3. **类型支持**：确保数据类型与索引文件匹配
4. **k值选择**：k值过小可能影响图连通性，过大则增加计算开销
5. **搜索深度**：Best First Search的深度限制为 `k * 10`，可根据需要调整

## 示例代码

参见 `knn_graph_example.cpp` 文件中的完整示例。

## 未来改进

1. **PQ量化加速**：集成Product Quantization进行更快的距离计算
2. **增量更新**：支持图的增量更新
3. **更多距离度量**：支持余弦距离、内积等
4. **自适应k值**：根据节点密度自动调整k值
