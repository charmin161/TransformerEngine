先对齐一下概念：你说的 **PLE**，在 Hugging Face 这版源码里不是官方命名；它对应的是 `per_layer_inputs / hidden_size_per_layer_input` 这条“每层额外输入”分支。你贴的这些参数都属于 `Gemma4Model.language_model` 里的 **文本塔第 0 层**。对 `gemma-4-E2B` 来说，文本侧真实配置是：`hidden_size=1536`、`intermediate_size=6144`、`num_hidden_layers=35`、`num_attention_heads=8`、`num_key_value_heads=1`、`head_dim=256`、`global_head_dim=512`、`hidden_size_per_layer_input=256`、`num_kv_shared_layers=20`、`use_double_wide_mlp=true`；layer type 是“4 个 sliding + 1 个 full”循环，所以 full attention 层在 `4/9/14/19/24/29/34`。([Hugging Face][1])

### 1）从一个 `token_id` 开始，实际上会生成两条输入流

对一个纯文本 token，主流很直接：

`input_ids[B,S] -> embed_tokens -> inputs_embeds[B,S,1536]`

这里 `embed_tokens` 是带缩放的 embedding，缩放因子是 `sqrt(hidden_size)`，对 E2B 就是 `sqrt(1536)`。与此同时，PLE 这条 side-channel 也会为**同一个 token**生成一份“每层专属”的 256 维向量：
`embed_tokens_per_layer(input_ids)` 先产出 `[B,S,35*256]`，再 reshape 成 `[B,S,35,256]`。此外，主 embedding `inputs_embeds[B,S,1536]` 还会经过一个 `per_layer_model_projection: 1536 -> 35*256`，reshape 成另一份 `[B,S,35,256]`，再做 RMSNorm；最后两份相加并乘 `1/sqrt(2)`，得到最终的 `per_layer_inputs[B,S,35,256]`。所以 Gemma4 不是只有一条 1536 维 residual stream，而是还有一条“每层各自一份 256 维额外输入”的 side stream。([Hugging Face][1])

把 shape 写死成 E2B 就是：

* 主 token embedding：`[B,S,1536]`
* PLE token embedding：按源码构造应为 `embed_tokens_per_layer.weight [262144, 35*256] = [262144, 8960]`
* PLE 投影：`per_layer_model_projection.weight [8960, 1536]`
* 最终 PLE 输入：`per_layer_inputs [B,S,35,256]`。([Hugging Face][1])

### 2）第 0 层（你贴出来这一层）的真实前向

第 0 层是 `sliding_attention`，所以它走的是 **局部滑窗注意力**，`head_dim=256`，不是 full 层的 512。设当前输入是 `x0[B,S,1536]`。先走 attention block：

`x0 -> input_layernorm -> a0[B,S,1536]`

然后进入 self-attention：

* `q_proj.weight [2048,1536]`，因为 `8 * 256 = 2048`
  `Q = q_proj(a0) -> [B,S,2048] -> view [B,S,8,256]`
* `q_norm.weight [256]` 不是 bug，它是**沿单头最后一维**做 norm，所以参数只要 256，而不是 2048
* `k_proj.weight [256,1536]`，因为 E2B 是 `num_key_value_heads=1`
  `K = k_proj(a0) -> [B,S,256] -> view [B,S,1,256]`
* `v_proj.weight [256,1536]`
  `V = v_proj(a0) -> [B,S,256] -> view [B,S,1,256]`
* `k_norm.weight [256]` 同理也是 per-head norm
* 代码里其实还有 `v_norm`，但它 `with_scale=False`，所以**没有单独的 checkpoint 参数**，这也解释了你列表里为什么没有 `v_norm.weight`
* Q/K 做 RoPE，随后转成 attention layout：
  `Q -> [B,8,S,256]`，`K/V -> [B,1,S_or_T,256]`

因为 `num_key_value_heads=1`，这是 **MQA**：backend 会把 K/V repeat 成 8 个 query groups，再算注意力分数，所以 score 张量是 `[B,8,S,T]`；这里 `T` 是当前可见的 KV 长度，首 token 时 `T=1`，增量解码时是 `past + 1`。layer 0 是 sliding 层，所以 mask 只允许看最后一个窗口（E2B 的 `sliding_window=512`）。attention 输出先得到 `[B,8,S,256]`，再 reshape 成 `[B,S,2048]`，最后过 `o_proj.weight [1536,2048]` 回到 `[B,S,1536]`。之后再过 `post_attention_layernorm[1536]`，加回 residual。([Hugging Face][1])

### 3）第 0 层的 FFN 和 PLE 是怎么接上的

attention 之后进入 FFN：

* `pre_feedforward_layernorm.weight [1536]`
* `gate_proj.weight [6144,1536]`
* `up_proj.weight [6144,1536]`
* `down_proj.weight [1536,6144]`

这说明 layer 0 的 FFN 宽度是 `6144 = 4 * 1536`。其公式不是 SwiGLU，而是：

`FFN(x) = down_proj( act(gate_proj(x)) * up_proj(x) )`

其中 `act` 对 E2B 文本塔来说是 `gelu_pytorch_tanh`。所以 layer 0 的 FFN 中间张量形状是：

* `gate = [B,S,6144]`
* `up = [B,S,6144]`
* `gate_act * up = [B,S,6144]`
* `down = [B,S,1536]`

E2B 的 `enable_moe_block=false`，所以代码里的 router / experts 分支在这个模型上**根本不走**。FFN 之后 `post_feedforward_layernorm[1536]`，再做一次 residual add。([Hugging Face][1])

然后才轮到你关心的 PLE 分支。第 0 层会取出这一个 token 对应的 `per_layer_inputs[:,:,0,:]`，shape 是 `[B,S,256]`。层内做的是：

* `per_layer_input_gate.weight [256,1536]`：把主流 `1536 -> 256`
* 过同一个 `gelu_pytorch_tanh`
* 与该层专属的 `per_layer_input[B,S,256]` 做逐元素乘法
* `per_layer_projection.weight [1536,256]`：再从 `256 -> 1536`
* `post_per_layer_input_norm.weight [1536]`
* residual add
* 最后整层输出乘 `layer_scalar [1]`

所以第 0 层完整输出还是 `[B,S,1536]`，只是内部多了一条：

`1536 -> 256 -> (和 per_layer_input[256] 相乘) -> 1536`

的门控 side branch。你列的 `layer_scalar / per_layer_input_gate / per_layer_projection / post_per_layer_input_norm` 正是在这里用的。([Hugging Face][1])

### 4）为什么后面层会和 layer 0 不一样

有三个关键差异。

第一，**full attention 层**（4/9/14/19/24/29/34）会把 `head_dim` 从 256 切到 512，所以按 config + 构造逻辑推导，这些层的投影 shape 应该是：

* `q_proj [4096,1536]`，因为 `8 * 512 = 4096`
* `k_proj [512,1536]`
* `v_proj [512,1536]`
* `o_proj [1536,4096]`

但最终层输出仍然回到 `[B,S,1536]`。同时 full 层用的 RoPE 配方也和 sliding 层不同。([Hugging Face][1])

第二，`num_kv_shared_layers=20`，35 层里前 15 层是非 shared 区，后 20 层是 shared-KV 区。按当前层型排布，**shared sliding 层**会复用第 13 层缓存下来的 K/V，**shared full 层**会复用第 14 层缓存下来的 K/V。也就是：

* `15,16,17,18,20,21,...,33 -> 共享 layer 13 的 K/V`
* `19,24,29,34 -> 共享 layer 14 的 K/V`

这里要注意一个细节：这条复用路径是在 **cache 存在时**走的；而 Gemma4 文本模型默认 `use_cache=True`，推理时通常会自动建 `DynamicCache`，所以实际推理里这套 shared-KV 会生效。训练或显式 `use_cache=False` 时，它们会各自重新算自己的 K/V。([Hugging Face][1])

第三，`use_double_wide_mlp=true` 只在这些 shared-KV 层上生效，所以第 `15~34` 层的 FFN 宽度会从 `6144` 翻倍成 `12288`。也就是说，后 20 层按源码推导应当是：

* `gate_proj [12288,1536]`
* `up_proj [12288,1536]`
* `down_proj [1536,12288]`

而你贴的 layer 0 还是普通 `6144`，因为它不在 shared-KV 区里。([Hugging Face][1])

### 5）多模态场景下，PLE 这个“额外输入”到底怎么喂

如果走的是顶层 `Gemma4Model`（而不是单独 `language_model`），那前向不是“先把 image/audio token 当普通 token embedding”，而是：

1. 先从 `input_ids` 生成文本主 embedding
2. 再把 image / video / audio placeholder 位置替换成对应 tower 输出的 soft tokens
3. 把替换后的 `inputs_embeds[B,S,1536]` 喂给 `language_model`

而 `per_layer_inputs` 这条 side input 会在**soft token merge 之前**就预先从原始 `input_ids` 计算好；源码 docstring 也明确写了，这是 multimodal wrapper 的主要用途：因为 soft token 一旦 scatter 进去，原始 token id 已经不可逆了。于是文本塔每一层拿到的实际上是两份输入：

* 主流：`hidden_states[B,S,1536]`
* side input：`per_layer_inputs[B,S,35,256]`

第 `i` 层只消费其中的 `per_layer_inputs[:,:,i,:]`。([GitHub][2])

### 6）把它压成一个最简前向

```python
# input_ids: [B, S]

E = embed_tokens(input_ids)                      # [B, S, 1536]

P_tok = embed_tokens_per_layer(input_ids)        # [B, S, 35*256]
P_tok = P_tok.view(B, S, 35, 256)

P_mod = per_layer_model_projection(E)            # [B, S, 35*256]
P_mod = P_mod * (1536 ** -0.5)
P_mod = P_mod.view(B, S, 35, 256)
P_mod = RMSNorm_256(P_mod)

P = (P_tok + P_mod) * (2 ** -0.5)                # [B, S, 35, 256]

x = E                                            # [B, S, 1536]

for i in range(35):
    # attention
    h = input_ln_i(x)
    a = self_attn_i(h)                           # -> [B, S, 1536]
    x = x + post_attn_ln_i(a)

    # ffn
    f = pre_ffn_ln_i(x)
    f = down_i(act(gate_i(f)) * up_i(f))         # -> [B, S, 1536]
    x = x + post_ffn_ln_i(f)

    # ple
    p = per_layer_input_gate_i(x)                # [B, S, 256]
    p = act(p) * P[:, :, i, :]                   # [B, S, 256]
    p = per_layer_projection_i(p)                # [B, S, 1536]
    x = x + post_per_layer_input_norm_i(p)

    # layer scalar
    x = x * layer_scalar_i

x = final_norm(x)                                # [B, S, 1536]
logits = lm_head(x)                              # [B, S, 262144]
```

对你贴的 **layer 0** 来说，把上面代进去就是：`Q=2048`、`KV=256`、`FFN=6144`、`PLE bottleneck=256`；对 full 层则是 `Q=4096`、`KV=512`；对后 20 个 shared-KV 层，FFN 会扩成 `12288`。最后如果接 `CausalLM` 头，还会走 tied `lm_head` 并做 `final_logit_softcapping=30.0`。([Hugging Face][1])

下一条可继续把这套逻辑改写成一版完全对齐 HF 命名的 PyTorch 伪代码。

[1]: https://huggingface.co/google/gemma-4-E2B/blob/main/config.json "https://huggingface.co/google/gemma-4-E2B/blob/main/config.json"
[2]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py"

我查了一圈后，结论很明确：**公开资料里没有一篇把 Rubin 的 adaptive compression 直接画成完整 datapath**。NVIDIA 官方只公开到结果层：第三代 Transformer Engine 有硬件加速的 adaptive compression，保持现有编程模型，并自动提升 arithmetic density、execution efficiency 和前端/decode efficiency；而公开 PTX 9.2 里的 `tcgen05.mma.sp` 仍然只暴露 **结构稀疏**，例如 `tf32` 是 1:2，`f16/f8/i8` 是 2:4，`mxf4/mxf4nvf4` 是 4:8（按 pair）。PTX 还明确写了 sparsity selector 是从 TMEM（Tensor Memory，张量内存）里选子列形成 metadata matrix，再与 `A` 一起形成 multiplicand。换句话说，你说的“先小窗口识别/编码，再跨更长 K 重打包成更少的 `g×64` packet”如果 Rubin 真有，今天大概率还是藏在 Transformer Engine/TMEM/collector 前端里，而不在公开 MMA contract 里。([NVIDIA Developer][1])

沿着你这条 two-stage 路径去找，**最像的不是某一篇单独文献，而是一组 NVIDIA 专利/论文刚好把它需要的 primitive 拼齐了**。按“额外运算/额外硬件”从低到高，大致可以分成四档。

## 1）最低开销：静态预打包 + 编译/驱动携带 sparsity information

最贴近你现在 `weights-only` 场景的是 NVIDIA 的 **US20220366007A1《Performing matrix value indication》**。它明确写到：API/编译器可以压缩一个或多个矩阵，既可以压缩 **行**，也可以压缩 **列**；把 non-zero values 和它们的 index values 存到 GPU 可访问的数据结构里；编译器接收带 sparsity information 的指令后，再生成 GPU 执行的矩阵乘指令。它还描述了 gather 得到 non-zero indices、compress 只存 non-zero values+indices、以及由这些信息生成 scatter instruction 的流程。同一组申请里，还能看到兄弟申请 **“Application programming interface to decompress data”** 和 **“Matrix multiplication and accumulation operations on compressed matrices”**，这很像是把“标注非零—解压/散布—压缩矩阵 MMA”当成一整套栈来申请。对你的 `[4096,512]` 固定权重零值场景，这条路的价值很直接：**把找零、找索引、做压缩这件事前移到模型加载/plan 阶段，运行时就不再扫描整块 W。** ([谷歌专利][2])

## 2）低开销：把“小窗口识别/编码”变成 metadata + gather，而不是 CSR/COO

最像你说的“先在小窗口里识别/编码”的，是 NVIDIA 的 **US11489541B2《Compression techniques for data structures suitable for artificial neural networks》**。它给出的 primitive 非常直接：先从一个 `N` 元素数据结构里**选出 `M` 个元素**，记录这些元素的位置作为 metadata，再把这 `M` 个值 gather 成压缩后的数据结构；专利里还有 `GENMETADATA`、`GATHER`、`GATHERPLUS` 这组指令级 primitive，并且明确展示了 **级联（cascaded）** 地把它们串起来做更高阶的压缩，比如 2:8。更重要的是，这件专利明确把 **CSR/COO** 作为“硬件复杂、效率差”的对照，强调固定大小的 metadata + gather 会让硬件更简单、更快。沿着你的模型翻译一下，这套 primitive 很像：**先在 K16/K32 这种小窗口里做 local live-mask / local bitmap / local packed-values。** ([谷歌专利][3])

## 3）低到中等开销：只做 partial decompress / scatter，不做 full reconstruct

你最关心的是：既然下游 MMA 还是规则化接口，那怎么把前面这些小窗口结果尽量便宜地拼成固定 packet？这里最值得看的是 NVIDIA 的 **US20200285618A1《Decompression techniques for processing compressed data suitable for artificial neural networks》**。它的核心不是“把压缩数据一次性恢复成完整 dense 块”，而是提供 **SCATTER** 指令，让一个线程、线程对（PAIR）、或者四线程组（QUAD）只对**目标大向量中的某个连续子段**做解压；专利里还给了从更密集压缩格式到较稀疏压缩格式的 **partial decompression**，比如 2:16 到 2:4、2:32 到 2:4。它甚至直接说了这样做的动机：传统 sparse decompression 常常需要复杂的 arithmetic circuitry，而它要尽量避免这种复杂度。沿着你的 `g×64 packet` 模型，这很像一个 **packet builder 的后半段**：前面的小窗口 metadata 已经告诉你哪些 K 位置活着，SCATTER/partial-decompress 负责只把当前 packet 需要的那一段 materialize 出来，而不是把整块窗口先恢复成 dense 再二次筛。([谷歌专利][4])

## 4）低到中等开销：把 activation 侧的“找 0”绑定到 LSU 载入路径

如果 Rubin 不只吃 fixed-weight zeros，还想顺带利用 activation 侧的 exact zero 或 threshold-zero，那么最像你说的“尽量减少额外运算”的，不是单独发一轮 compare kernel，而是 NVIDIA 的 **US11977888B2《Inline data inspection for workload simplification》**。这件专利把 inspection circuit 直接挂在 **LSU（Load/Store Unit，加载/存储单元）** 和 data storage 之间：load 进来时就判断是不是 0、或者是不是低于 threshold，直接给 LSU 返回 predicate；数据可以存寄存器，也可以直接丢弃；更重要的是，它明确说**不需要额外插 explicit zero-detection / comparison instructions**。这正是最省额外运算的一类设计：把“判断它是不是该删”这件事，埋进本来就要发生的 load 流水里。([谷歌专利][5])

## 5）辅助但很重要：转置友好的压缩格式，避免为了 `W^T` 再做一遍重排

你前面多次提到内部很可能会把 `A@W` 改写成 `W^T A^T` 去适配 sparse-A-like 路径。NVIDIA 的 **US11127167B2《Efficient matrix format suitable for neural networks》** 正好补了这一块：它提出一种 diagonal storage format，用 non-zero values + bitmap/mask 存稀疏矩阵，并且强调可以从同一压缩表示里**容易地生成 original、transposed、compacted original、compacted transposed** 这些形式；甚至可以在 load 指令里直接把压缩存储解到 dense 或 dense-transposed fragment 和 metadata in registers，而**不必先重构完整矩阵**。这对你的权重张量尤其有用：如果内部 kernel 真想按 `W^T` 来组织 sparse operand，这类格式能显著减少“单纯为了转置而做的前端额外工作”。而且这件专利明确说 diagonal-wise storage **不限于 4:8**，可以用于“any type of sparsity”。([谷歌专利][6])

## 把上面五件东西拼起来，最像你那条“低额外运算”路径

如果我把这些公开件拼成一条**最像 Rubin、同时额外运算最少**的路径，顺序会是这样：

先在模型加载或首次 plan 时，把 `W` 按小窗口预打包。对你这个 `[4096,512]` 的权重，更像是按 `K16/K32` 窗口生成 `{packed values, bitmap/indices, scale}` 的 record；如果内部想走 `W^T A^T`，就直接存成 transposed-friendly 的压缩格式。这样一来，**weights 侧的找零/找索引/转置**都从 runtime 移走了。这个阶段最像 `matrix value indication` + `efficient matrix format` + `GENMETADATA/GATHER`。([谷歌专利][2])

运行时，`A` 还是按常规 tile 被载入；如果还想吃 activation 侧的 exact-zero 或 threshold-zero，就用 LSU 路径上的 inline inspection 顺手给每个小窗口生成 predicate / mask。然后，前端读取若干连续的小窗口 record，把活着的 K 位置往一个 packet buffer 里装，直到凑满一个固定的 `g×64` packet。这里**公开文献没有把 “packet builder” 四个字明写出来**，但前面那几件专利已经把它需要的 primitive 基本给齐了：metadata generation、gather、partial scatter/decompress、transposed load。([谷歌专利][5])

之后再把这个固定 packet 交给规则化 MMA。Blackwell 公开架构已经说明 Tensor Cores 是做 single-instruction MMA 的小矩阵引擎，而且和 **TMEM（Tensor Memory，张量内存）** 紧耦合；Rubin 又把 adaptive compression 放在 Transformer Engine 下面，并强调 preserves existing programming model、improved front-end/decode efficiency。把这些拼起来，我更倾向于：**真正的“跨 K 重打包”发生在 MMA 前端，而不是在 accumulator 阵列里做任意 scatter-add。** ([NVIDIA Developer][7])

## 更重、但也更“完整”的动态路径：Stitch-X、SNAP、SCNN

如果不满足于 one-sided sparse，而是想让 **W 和 A 两边都以 fully unstructured 方式动态参与**，NVIDIA 相关研究里也有更激进的路线，但额外开销明显更高。

**Stitch-X** 的思路最像“runtime packet discovery”：它用一个 **PDU（Parallelism Discovery Unit，并行度发现单元）** 在运行时把 sparse weights 和 sparse activations stitch 成可并行执行的 pair；PDU 里有 `C×C` comparator matrix、priority encoding 和 address lookup，论文报告这部分 decode/stitching 面积开销不到 8%，整体对 dense baseline 有 3.8× speedup，平均计算单元利用率约 74%。这条路的味道很像你说的“跨更长 K 范围去装箱”，但代价是前端真的要做 runtime search。([CloudFront][8])

**SNAP** 更进一步。它用 **AIM（Associative Index Matching，关联式索引匹配）** 单元和 sequence decoder，在压缩后的 W/IA 数组里做并行 index matching，去找有效 pair；同时用 two-level psum reduction 把输出端的 contention 和 writeback traffic 压下去。论文报告平均 compute utilization 约 75%，psum writeback traffic 相比之前的 sparse accelerator 降了 22×，但前端和后端都引入了明显的额外逻辑。对“weights-only 固定零值”的场景，这套通常太重了。([EECS Berkeley][9])

再往上，就是 **SCNN** 这种 fully irregular sparse-dataflow：non-zero values 从一开始就带着 position/coordinate tag 走，乘出来的 product 也带地址，经过 crossbar 路由到 addressable accumulator array 做 scatter-add。这个方向最灵活，但额外的坐标计算、crossbar 和 accumulator logic 也最重；它更像一台专用 sparse machine，而不像今天公开的 Tensor Core/PTX contract。([谷歌专利][10])

## 我对“尽可能减少额外运算”的判断

沿着你的 `g×64 packet` 模型，我会把路径优先级排成这样：

**最省额外运算的组合**：
固定权重离线预打包 → 小窗口 metadata/bitmap → 载入时 activation inspect → 近计算端 partial scatter/decompress → 规则化 MMA。
这条路把额外工作压成：metadata 读写、少量 bit 操作、少量 mux/scatter、以及 packet buffer 的填充逻辑。它尽量避免了 CSR/COO 的 pointer chasing，也避免了 full dense reconstruct。([谷歌专利][3])

**中等开销的组合**：
在上面基础上，再加 Stitch-X/SNAP 那种 prescan / AIM / decoder。
只有当你真的想同时吃 **两边** 的动态非结构稀疏，而且一侧的固定 metadata 已经不够时，这类逻辑才值得。([CloudFront][8])

**最重的组合**：
SCNN 式坐标携带 + scatter-add accumulator。
这条路理论上最通用，但和 Rubin 官方“保持现有编程模型”、以及公开 PTX 仍只暴露 structured sparse MMA 的事实，不是一个味道。([谷歌专利][10])

补一个很贴近你问题的**非 NVIDIA 旁证**：ISCA 2021 的 **Dual-side Sparse Tensor Core** 明确主张“为了让 Tensor Core 友好，最好走 bitmap + outer-product 风格，而不是 inner-join + prefix-sum/barrier 的前端”，并把后者归因于复杂 prefix-sum hardware 和 barrier 开销。虽然它不是 NVIDIA 的公开设计，但它和上面那条“低额外运算 stack”的方向非常一致。([Homepage][11])

一句话收束：

**我没有找到一篇公开文献直接写出 Rubin 的“跨 K 重打包器”，但我找到了几乎完整的一套 NVIDIA 零件库：offline value indication / prepack、LSU 侧 inline inspection、GENMETADATA/GATHER、partial SCATTER/decompress、以及 transposed-friendly sparse format。把这些拼在一起，正好就是你要的那条“先小窗口识别/编码，再跨更长 K 范围重组成更少 `g×64` packet、再喂给规则化 MMA”的最省额外运算路径。**

下一条我可以把这套专利拼成一张专门针对你 `W[4096,512]` 的“**K16 records → packet builder → g×64 packet → tcgen05-like MMA**”时序图。

[1]: https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/ "https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/"
[2]: https://patents.google.com/patent/US20220366007A1/en "https://patents.google.com/patent/US20220366007A1/en"
[3]: https://patents.google.com/patent/US11489541B2/en "https://patents.google.com/patent/US11489541B2/en"
[4]: https://patents.google.com/patent/US20200285618A1/en "https://patents.google.com/patent/US20200285618A1/en"
[5]: https://patents.google.com/patent/US11977888B2/en "https://patents.google.com/patent/US11977888B2/en"
[6]: https://patents.google.com/patent/US11127167B2/en "https://patents.google.com/patent/US11127167B2/en"
[7]: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ "https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/"
[8]: https://d1qx31qr3h6wln.cloudfront.net/publications/SysML_StitchX_2018.pdf "https://d1qx31qr3h6wln.cloudfront.net/publications/SysML_StitchX_2018.pdf"
[9]: https://people.eecs.berkeley.edu/~ysshao/assets/papers/snap2021-jssc.pdf "https://people.eecs.berkeley.edu/~ysshao/assets/papers/snap2021-jssc.pdf"
[10]: https://patents.google.com/patent/US20180046900A1/en "https://patents.google.com/patent/US20180046900A1/en"
[11]: https://chenzhangsjtu.github.io/files/2021-ISCA.pdf "https://chenzhangsjtu.github.io/files/2021-ISCA.pdf"
