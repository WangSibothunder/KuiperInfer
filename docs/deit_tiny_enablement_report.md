# KuiperInfer 支持 DeiT-Tiny（pnnx）端到端推理：工程复盘与面试包装手册

> 文档定位：面向 **推理引擎/系统岗** 的“可讲述、可复核、可追问”版本。  
> 使用方式：你可以直接把本文当成项目复盘稿、简历素材库、面试答题库。  
> 包装边界：**真实增强**（只基于已完成事实，不虚构未做项）。

---

## 0. 读者与用途（先讲清楚你为什么写这份文档）

### 0.1 目标读者
- 面试官：关心你是否真的做过“模型推理框架落地”，不是只会调 API。
- 你自己：需要在 1 分钟、3 分钟、10 分钟三个粒度都能稳定讲清楚。

### 0.2 你要证明的三件事
1. 你不是“跑了个 demo”，而是把 **Transformer Attention 全链路** 从不可用修到可用。
2. 你不仅修正确性，还做了 **性能工程**（热点识别 + 算子重写）。
3. 你有工程方法论：能用“现象 -> 根因 -> 修复 -> 回归”闭环推进复杂系统问题。

### 0.3 真实增强原则（面试不要踩雷）
- 可说：你修了 runtime 数据传播、attention 关键算子、形状语义问题，最终 E2E 通过且对齐。
- 不要说：你做了完整通用 Transformer 引擎（这超出当前事实）。
- 可说：你做了阶段性性能优化（约 9.7x 改善），但仍慢于 PyTorch CPU。
- 不要说：已经全面超过 PyTorch（与当前结果不符）。

---

## 1. 项目目标与验收标准

### 1.1 项目目标
在 KuiperInfer 上支持 `deit_tiny`（由 pnnx 导出）端到端推理，做到：
- 不崩溃（无 fatal / assert）
- 输出数值和 PyTorch 对齐
- 支持 Attention 全链路执行

### 1.2 验收命令（事实证据）
- 构建：
```bash
cmake --build build --parallel 4
```
- E2E：
```bash
./build/test/test_kuiper --gtest_filter=test_model.deit_tiny_e2e
```

### 1.3 验收口径
- 功能：测试完整跑完，不在 `transpose/cat/attention` 等处崩溃。
- 精度：`cos_sim > 0.99`（当前实测为 `1`）。

---

## 2. 系统背景：KuiperInfer 是如何执行一个 pnnx 模型的

这一节很重要：你要让面试官相信你理解的是“系统”，不是“某个 cpp 文件”。

### 2.1 运行时三层结构
1. **模型加载层（IR）**：读取 `param/bin`，将 pnnx 节点转成 `RuntimeOperator`。  
2. **图构建层（Graph Build）**：建立依赖关系、拓扑顺序、输入输出张量空间。  
3. **算子执行层（Layer Forward）**：逐算子前向，并通过传播函数把输出传给下游。

### 2.2 关键执行路径（你可以背）
- `RuntimeGraph::Init`：读图 + 初始化 operator/params/attrs
- `RuntimeGraph::Build`：建边 + topo 排序 + 初始化 operand datas + 填充 `pnnx.Attribute`
- `RuntimeGraph::Forward`：按拓扑执行 + `PropagateLayerOutputs` 传播
- `Layer::Forward`：收集 `input_operands_seq` 输入，空输入直接失败

### 2.3 为什么 DeiT 会比 CNN 更容易把框架打穿
- DeiT 的 attention 路径有大量形状变换与分支拆分：
  - `qkv linear -> reshape -> permute -> unbind -> SDPA -> transpose -> reshape -> proj`
- 任意一个环节语义不一致，都会表现为：
  - 空输入、维度错位、越界、数值异常、后续残差链路中断

---

## 3. Attention 从“不可支持”到“可支持”的工程闭环（重点章节）

> 这是你在面试中最该重点讲的章节。

### 3.1 失败现象与触发路径

以第一个 block 为例，理想路径应为：

```text
ln_0
 -> blocks.0.attn.qkv
 -> Tensor.reshape_38
 -> Tensor.permute_26
 -> torch.unbind_77
 -> F.scaled_dot_product_attention_116
 -> torch.transpose_65
 -> Tensor.reshape_39
 -> blocks.0.attn.proj
```

早期失败现象集中在：
- `torch.transpose` 维度异常（越界/fatal）
- `torch.cat_62` 输入来自 `pnnx_fold_38` 为空
- `torch.unbind -> SDPA` 只传进一个分支，q/k/v 不完整
- 线性层和 attention 太慢导致 E2E 速度极差

### 3.2 根因分层（你可以用这张“故障树”回答追问）

1. **语义层**：pnnx 的维度编号与 runtime Tensor 维度解释存在偏差。  
2. **数据流层**：operand key 查找失败导致传播 miss。  
3. **结构层**：one-to-many（`unbind` 一出三）传播机制不完善。  
4. **常量层**：Attribute 虽已填充，但未稳定传入下游输入槽。  
5. **性能层**：Linear/SDPA 使用低效循环实现，成为热点瓶颈。

### 3.3 算子注册与实例化（Attention 核心算子是怎么“被框架认识”的）

你做的不是“临时调用函数”，而是走框架标准路径：

1. **注册**：通过 `LayerRegistererWrapper` 将类型字符串与创建函数绑定。  
2. **实例化**：`CreateInstance` 从 `RuntimeOperator` 读取参数并创建具体层对象。  
3. **挂接到图**：`CreateNodeRelation` 阶段创建 layer 并写入 `RuntimeOperator::layer`。  
4. **执行**：`RuntimeGraph::Forward` 统一调用 `ExecuteLayer`。

Attention 对应关键注册类型：
- `F.scaled_dot_product_attention`

关键点：type 字符串必须和 pnnx 导出保持一致，否则工厂无法命中。

### 3.4 Runtime 数据流修复（Attention 能跑起来的关键）

#### 问题 1：`Attribute -> cat` 传播断链
- 现象：`pnnx_fold_38` Build 时有值，但 `torch.cat_62` 输入为空。
- 根因：`next_input_operands.find(current_op->name)` 在部分边上 miss。

#### 修复策略
在 `PropagateLayerOutputs` 中保留主路径，同时增加两个 fallback：
1. **按 `input_operands_seq` 名称扫描**（补 key miss）  
2. **形状匹配兜底**（含去 batch 维兼容）

结果：常量链路打通，`cat` 不再空输入。

#### 问题 2：`unbind` 一出三，SDPA 只收到一支
- 根因：map 结构天然偏“一 key 一入口”，难表达同 producer 多输入位。

#### 修复策略
- 在 key 命中后，补充 one-to-many 分发逻辑：
  - 将 `layer_output_datas` 按顺序分发到 `input_operands_seq` 中同名但不同槽位的 operand。

结果：`q/k/v` 三路输入完整进入 SDPA。

### 3.5 SDPA 正确性与性能实现（从可跑到可用）

#### 正确性约束
- 输入必须是 3 个 tensor：Q/K/V。
- 输入形状语义固定为 `(B, H, S, D)`。
- scale 缺省按 `1/sqrt(D)` 自动计算。
- 输出形状回到 Q 的形状。

#### 算法路径
- `logits = Q * K^T`  
- row-wise softmax（减 max 保持数值稳定）  
- `out = probs * V`

#### 性能实现
- 将 `QK^T` 和 `PV` 两段核心计算改为 BLAS GEMM（`cblas_sgemm`）。
- softmax 保留逐行稳定实现。

结果：attention 层耗时显著下降，E2E 总时长同步下降。

### 3.6 多 block 验证与残差回接

你不是只修了 `blocks.0`：
- 通过 E2E 路径验证 attention 在多个 block 连续执行
- 确认 attention 输出经后处理后能正确回到残差表达式（`pnnx.Expression`）链路
- 最终能跑到 `head` 并通过输出对齐校验

---

## 4. 从“跑不通”到“跑通”的关键修复（统一按六段法）

> 本节每个子项都按：`现象 -> 证据 -> 根因 -> 修复策略 -> 风险 -> 回归测试`。

### 4.1 `torch.transpose` 维度语义修复

**现象**
- transpose 处出现维度越界或错误交换，导致后续 attention 输入布局错乱。

**证据**
- 调试日志可见 rank 与 dim 不一致（如 rank=2 但请求 dim=2）。

**根因**
- pnnx 导出维度语义包含逻辑 batch，而 runtime raw shape 解释方式不同。

**修复策略**
- 引入逻辑维映射；
- 对明显越界与 batch 参与交换场景采用兼容 no-op，优先保稳定。

**风险**
- 兼容逻辑若过宽可能掩盖新问题。

**回归测试**
- 运行 `deit_tiny_e2e`，并观察 attention 前后 `transpose` 节点可持续执行。

### 4.2 `Attribute` 常量传播修复（`pnnx_fold_38 -> torch.cat_62`）

**现象**
- Build 阶段 Attribute 已填充，Forward 阶段下游仍空。

**证据**
- 构图日志显示填充成功；执行日志显示 `cat` 某输入为空。

**根因**
- key 查找 miss，导致传播函数没写进目标输入 datas。

**修复策略**
- `PropagateLayerOutputs` 增加名称扫描 fallback + 形状匹配 fallback。

**风险**
- fallback 若缺边界可能误匹配，需加可观测日志与一次性 warning。

**回归测试**
- `cat` 相关测试 + DeiT E2E；确认不再出现该空输入故障。

### 4.3 `torch.unbind -> SDPA` one-to-many 分发修复

**现象**
- SDPA 理应接收 q/k/v 三路，实际仅一条被填充。

**证据**
- `PropagateMismatch` 可见 `layer_output_datas=3` 但单槽位被命中。

**根因**
- map 型输入结构不能天然覆盖一 producer 到多输入槽的语义。

**修复策略**
- 在 key 命中后追加对 `input_operands_seq` 同名槽位的游标分发。

**风险**
- 顺序依赖错误会导致 q/k/v 对错位。

**回归测试**
- 关注 attention 层持续通过；最终 `cos_sim` 对齐通过。

### 4.4 `nn.Linear` 正确性与性能改造

**现象**
- 线性层是 attention 与 MLP 主热点；早期实现稳定性/性能都不足。

**证据**
- Layer timing 显示多处 linear 占时明显。

**根因**
- 大量三重循环 + 内存访问局部性差。

**修复策略**
- 明确 `feature_dims`/`in_features_` 推导规则；
- 输出形状稳定分配；
- bias 在输出上正确叠加；
- row-major 主路径改为 GEMM。

**风险**
- 布局设定错误会造成“看似可跑、数值悄悄偏离”。

**回归测试**
- E2E `cos_sim` 校验 + 观察 top5 类别一致性。

### 4.5 SDPA 算子重写

**现象**
- 注意力层前向耗时高，E2E 不可接受。

**证据**
- 优化前 profiling 中 SDPA 层多次高耗时。

**根因**
- 核心矩阵计算未使用高性能数值库。

**修复策略**
- `QK^T` 与 `PV` 均改 GEMM；softmax 做数值稳定版本。

**风险**
- 形状与 leading dimension 参数容易写错。

**回归测试**
- E2E 通过 + attention 层耗时显著下降。

### 4.6 运行时线程与日志策略收敛

**现象**
- 强制单线程限制压制性能；热路径日志噪声影响观察与速度。

**修复策略**
- 去掉 Forward 中强制设置线程为 1 的逻辑；
- 高频 debug 改环境变量开关（默认关）。

**回归测试**
- 比较优化前后 benchmark；同时保持 E2E 对齐。

---

## 5. 工程取舍：为什么这么做，而不是那样做

### 5.1 为什么先“稳定跑通”，再“追性能”
- 原因：attention 链路涉及多算子联动，先把语义和数据流对齐，否则性能优化会放大错误。
- 价值：先保证可回归，再做热点优化，迭代风险可控。

### 5.2 为什么在传播层做 fallback
- 原因：pnnx 图与 runtime operand 映射存在现实差异，不是每条边都能靠单一 key 精准命中。
- 价值：提高鲁棒性，避免因命名差异导致链路断裂。

### 5.3 为什么 Linear/SDPA 优先 GEMM
- 原因：这是 DeiT 最重计算路径，收益最大。
- 价值：少改动换大收益，且借助成熟 BLAS 降低自写内核风险。

### 5.4 为什么保留“可开关”的调试探针
- 原因：复杂图故障常复现难、定位慢；完全删掉探针会降低后续维护效率。
- 做法：默认关闭，定位时打开，兼顾性能和可维护性。

---

## 6. 性能结果与基线一致性说明（必须可复核）

### 6.1 测试条件（统一口径）
- 输入：`test_data/deit_e2e/input.csv`
- 模型：`deit_tiny.pnnx.param/bin`（Kuiper）与 `deit_tiny.pt`（PyTorch）
- 统计方式：warmup=5，iters=20，报告 `ms/iter`
- 线程：显式设置（示例：`OMP_NUM_THREADS=1`）
- 计时不包含 build 阶段

### 6.2 阶段性结果（已有事实）
- Kuiper（优化前，1线程）：约 `2105 ms/iter`
- Kuiper（优化后，1线程）：约 `216 ms/iter`
- 阶段加速：约 **9.7x**
- PyTorch CPU（1线程）：约 `24.6 ms/iter`
- 当前差距：Kuiper 仍慢于 PyTorch（约 8.8x）

### 6.3 当前主要瓶颈（诚实但专业）
- `permute/reshape` 的内存重排成本
- LayerNorm/GELU 的向量化程度
- 全链路线程并行策略仍可优化

---

## 7. 证据索引（主张-证据对照表）

| 主张 | 证据类型 | 命令/位置 |
|---|---|---|
| DeiT E2E 可完整跑通 | 测试通过 | `./build/test/test_kuiper --gtest_filter=test_model.deit_tiny_e2e` |
| 数值对齐通过 | 测试断言 | `cos_sim > 0.99`（当前日志显示 `cos_sim=1`） |
| Attention 链路已执行 | Forward 日志链路 | `blocks.*.attn.qkv -> ... -> F.scaled_dot_product_attention_*` |
| 传播断链问题被修复 | 运行时逻辑变更 | `source/runtime/runtime_ir.cpp` 中 `PropagateLayerOutputs` fallback + one-to-many |
| SDPA 已从慢循环改为 GEMM | 代码实现 | `source/layer/details/scaled_dot_product_attention.cpp` |
| Linear 热路径已优化 | 代码实现 | `source/layer/details/linear.cpp` |
| 性能已阶段提升 | benchmark 输出 | `kuiper_ms_per_iter` 优化前后对比 |

> 面试技巧：每说一个结果，都能落回“命令、日志、文件”三者之一。

---

## 8. 可复现实验附录（让第三方可核验）

### 8.1 构建与 E2E 回归
```bash
cmake --build build --parallel 4
./build/test/test_kuiper --gtest_filter=test_model.deit_tiny_e2e
```

### 8.2 Kuiper 侧时延测量（示例）
```bash
GLOG_minloglevel=2 OMP_NUM_THREADS=1 /tmp/bench_deit_kuiper
```
关注输出：`kuiper_ms_per_iter`

### 8.3 PyTorch CPU 侧对照（示例）
```bash
python3 - <<'PY'
import time, numpy as np, torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = torch.jit.load('deit_tiny.pt', map_location='cpu').eval()
x = torch.from_numpy(np.loadtxt('test_data/deit_e2e/input.csv', delimiter=',', dtype=np.float32).reshape(1,3,224,224))

warmup, iters = 5, 20
with torch.no_grad():
    for _ in range(warmup): model(x)
    t0 = time.perf_counter()
    for _ in range(iters): model(x)
    t1 = time.perf_counter()

print('pytorch_ms_per_iter=', (t1 - t0) * 1000 / iters)
PY
```

### 8.4 最终判定
- 功能：E2E 全程通过
- 精度：`cos_sim > 0.99`
- 性能：记录 Kuiper 与 PyTorch 的 `ms/iter`，按同口径对比

---

## 9. 简历可直接复用（四套模板）

### 9.1 1 句项目描述（简历条目）
我在 KuiperInfer 中完成了 DeiT-Tiny（pnnx）端到端推理支持，修复 attention 全链路（qkv 重排/分发/SDPA）与 runtime 传播断链问题，实现 `cos_sim=1`，并通过 Linear/SDPA 热点优化将推理时延降低约 9.7x（阶段性）。

### 9.2 3 句项目开场（面试口述）
1. 这个项目的核心不是“加一个算子”，而是把 Transformer attention 从不可执行修到可执行，涉及 runtime 图传播、形状语义和算子实现三层联动。  
2. 我们重点解决了 `Attribute->cat` 断链和 `unbind->SDPA` one-to-many 分发问题，并修正了 transpose/reshape/permute 在 DeiT 路径的语义偏差。  
3. 跑通后我又做了热点优化：Linear 与 SDPA 改 GEMM，E2E 在同口径下约 9.7x 提升，同时保持 `cos_sim=1`。

### 9.3 STAR 版本（可深挖）
- **S（情境）**：DeiT 在 KuiperInfer 上无法稳定推理，attention 链路会在重排/拼接/分发处断裂。  
- **T（任务）**：打通 DeiT E2E，保证数值对齐，并进行阶段性性能优化。  
- **A（行动）**：
  - 建立运行时探针定位空输入边；
  - 在 `PropagateLayerOutputs` 增加 fallback 与 one-to-many 分发；
  - 修复 transpose/reshape/permute/select 语义对齐；
  - 完成 SDPA 算子稳定实现并 GEMM 化；
  - 优化 Linear 热路径并做回归验证。  
- **R（结果）**：`test_model.deit_tiny_e2e` 通过，`cos_sim=1`；Kuiper 时延从约 2105ms 降到约 216ms（1线程，warmup=5/iters=20）。

### 9.4 包装边界（可说/不要说）
- 可说：我负责了 attention 使能链路和 runtime 传播鲁棒性。
- 不要说：我重写了整个推理框架。
- 可说：我做了阶段性加速，仍有进一步优化空间。
- 不要说：我已经全面超过 PyTorch。

---

## 10. 面试追问 Q/A（15 题，建议重点背前 8 题）

### Q1：你怎么定位到是传播问题，不是算子本身错？
A：我在 Forward 输入拼装处和传播函数上加了定向探针，先确认 `Attribute` 在 Build 阶段已填充，再看下游 operand datas 是否被赋值。结果证明是边命中失败，而不是常量本体缺失。

### Q2：为什么 `pnnx_fold_38` 有值但 `torch.cat_62` 还是空？
A：因为原传播依赖 `input_operands` 的 key 查找，命名/映射差异会 miss。我们加了 `input_operands_seq` 名称扫描和形状匹配 fallback，恢复了这条边。

### Q3：`unbind -> SDPA` 的难点是什么？
A：`unbind` 是一进三出，map 结构天然偏一对一。我们补了 one-to-many 分发逻辑，把 q/k/v 依序放进对应槽位，避免只填一支。

### Q4：为什么先修正确性再做性能？
A：attention 链路是强耦合结构，语义不对时性能优化只会放大错误。先跑通并建立可回归，再做热点优化，风险更可控。

### Q5：你如何保证不是“碰巧对齐”？
A：不是只看单点输出。我用 E2E 全链路 + `cos_sim` 阈值 + top5 观察，并验证多 block 连续 attention 执行，确保不是局部偶然。

### Q6：SDPA 你做了什么？
A：先保证输入约束 `(B,H,S,D)` 和 scale 规则正确，然后把 `QK^T`、`PV` 两段改为 GEMM，softmax 用逐行稳定实现，兼顾正确性和性能。

### Q7：Linear 为什么是性能重点？
A：DeiT 每个 block 都有 attention/MLP linear，调用密度高。把主路径从三重循环切到 GEMM，收益很直接。

### Q8：你如何定义“支持 attention”而不是“测试刚好过了”？
A：我定义为五层都成立：算子可注册、可实例化、q/k/v 可正确传播、SDPA 数值正确、后处理与残差可持续回接，且能跨 12 层 block 连续执行。

### Q9：fallback 会不会误匹配？
A：有风险，所以加了匹配顺序和 warning，并优先主路径命中。fallback 仅在 key miss 时启用，且通过 E2E 回归验证。

### Q10：为什么要保留 debug 探针？
A：复杂图问题复发时定位成本很高。我们用环境变量 gate 默认关闭，定位时打开，兼顾性能与可维护性。

### Q11：你做过哪些“最小改动”决策？
A：优先在 runtime 传播层补鲁棒逻辑，而不是大规模重构 IR 数据结构；先修关键路径，再做系统性清理。

### Q12：如何验证改动没有破坏已有功能？
A：除了 DeiT E2E，还回归了相关算子测试（如 cat 路径），并保持 기존调用接口不变。

### Q13：你对当前性能差距怎么看？
A：阶段性成果显著，但与 PyTorch 仍有差距。主要瓶颈在内存重排、向量化程度和并行策略，后续路径明确。

### Q14：如果再给你两周你做什么？
A：先做算子级 profiling 固化瓶颈，再优化 permute/reshape 与 LayerNorm/GELU 向量化，最后做线程策略与 cache 友好性改造。

### Q15：这个项目最大的工程价值是什么？
A：不是“某个 kernel 快了”，而是建立了复杂模型在自研推理框架中的可落地方法：可定位、可修复、可回归、可持续优化。

---

## 11. 30 秒 / 3 分钟 / 10 分钟讲述提纲

### 11.1 30 秒版本（电梯）
我把 KuiperInfer 上原本跑不通的 DeiT-Tiny attention 链路打通了，核心是修了 runtime 传播断链和 qkv 分发机制，同时补齐 SDPA 实现并做 GEMM 优化，最终 E2E 通过且 `cos_sim=1`，并把推理时延阶段性降了约 9.7x。

### 11.2 3 分钟版本（常规面试）
- 背景：DeiT 在重排、拼接和 attention 分发处容易断。  
- 方法：先加探针定位，再修传播与语义，再做热点优化。  
- 关键修复：`Attribute->cat` fallback、`unbind` one-to-many、transpose/reshape/permute 对齐、SDPA/Linear GEMM 化。  
- 结果：E2E + 精度对齐通过，性能显著改善，但仍有进一步优化空间。

### 11.3 10 分钟版本（技术深挖）
按本文第 3、4、5、6、7 节顺序讲：
- 先讲故障树
- 再讲 attention 闭环
- 然后讲工程取舍
- 最后讲证据与边界

---

## 12. 当前状态与下一步（你主动说，会显得成熟）

### 已完成
- DeiT-Tiny batch=1 E2E 跑通
- Attention 全链路可执行
- 精度对齐通过（`cos_sim=1`）
- 阶段性性能优化（约 9.7x）

### 下一步（建议路线）
1. 系统化 profiling（算子耗时 + 内存行为）。
2. `permute/reshape` 路径优化（减少重排开销）。
3. LayerNorm/GELU 向量化与并行策略。
4. 多 batch 泛化与鲁棒性验证。

---

## 13. 核心文件索引（面试官问“改了哪些地方”时直接用）

- Runtime/Graph：`source/runtime/runtime_ir.cpp`
- Attention 算子：`source/layer/details/scaled_dot_product_attention.cpp`
- 线性层：`source/layer/details/linear.cpp`
- 重排与张量操作：
  - `source/layer/details/transpose.cpp`
  - `source/layer/details/reshape.cpp`
  - `source/layer/details/permute.cpp`
  - `source/layer/details/unbind.cpp`
  - `source/layer/details/tensor_select.cpp`
  - `source/layer/details/cat.cpp`
- E2E 测试：`test/test_net/test_deit.cpp`

---

如果你要，我下一步可以在这个文件后面再追加一个“**逐题模拟面试（你问我答）**”版本：
- 偏系统实现（runtime/算子）
- 偏性能优化（profiling/算子改写）
- 偏项目 ownership（你如何推动迭代）
