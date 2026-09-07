你的理解**可以成立，但需要加一个非常关键的限定**：

> **可以始终保持 `[B,Sq,Sc]` 的逻辑 shape，用一个 0/1 Top-K mask 统一 dense / sparse 两阶段；但是这个 mask 必须在 Softmax/KL 的归一化域确定之前生效。**
>
> 如果是“先按全部 `Sc` 位置计算 dense KL 和 \(dI=Q-P\)，最后再把非 Top-K 的梯度乘 0”，那就**不等价**于 DeepSeek sparse-stage 的 KL。

而且现在我们已经有相当强的公开证据支持这个结论。

---

# 1. 先给结论：两种 sparse 实现都可以

设 V4 CSA：

$$
I\in\mathbb R^{B\times S_q\times S_c},
\qquad
S_c=S_q/m
$$

Top-K mask：

$$
M\in\{0,1\}^{B\times S_q\times S_c}
$$

其中每一行恰好 K 个位置：

$$
M[b,q,j]=1
$$

## 实现 A：full-shape + mask

保持：

$$
I:[B,S_q,S_c]
$$

但把非 Top-K：

$$
I'_{bqj}
=
\begin{cases}
I_{bqj},&M_{bqj}=1\\
-\infty,&M_{bqj}=0
\end{cases}
$$

然后再：

$$
Q=Softmax(I')
$$

因此：

$$
Q:[B,S_q,S_c]
$$

但非 Top-K：

$$
Q_{bqj}=0
$$

然后 teacher 也只在 Top-K 域内归一化。

最终：

$$
dI:[B,S_q,S_c]
$$

非 Top-K 自动为 0。

---

## 实现 B：直接 gather 成 compact Top-K

直接：

$$
I_{sel}=Gather(I,TopK)
$$

得到：

$$
[B,S_q,K]
$$

然后：

$$
Q_{sel}=Softmax(I_{sel})
$$

teacher：

$$
P_{sel}:[B,S_q,K]
$$

计算：

$$
KL(P_{sel}\|Q_{sel})
$$

得到：

$$
dI_{sel}:[B,S_q,K]
$$

最后 scatter：

$$
[B,S_q,K]
\rightarrow
[B,S_q,S_c]
$$

非选中位置补 0。

---

**A 和 B 数学上可以完全等价。**

而且非常有意思的是：

* NVIDIA **Megatron-LM reference/naive 实现采用 A 的思路**；
* NVIDIA **cuDNN 高性能 sparse kernel 采用 B 的思路**。

这正好回答了你的疑问。([GitHub][1])

---

# 2. 但你说的“先算完整 KL，再 mask 梯度”要小心

关键区别就在这里。

假设：

$$
I=[2,1,0,-1]
$$

即：

$$
S_c=4
$$

Teacher：

$$
P=[0.1,0.2,0.6,0.1]
$$

假设 Top-2 选：

$$
\mathcal S=\{0,1\}
$$

即：

$$
M=[1,1,0,0]
$$

---

# 3. 错误做法：先 full Softmax，再 mask 梯度

全部四个位置做：

$$
Q=Softmax([2,1,0,-1])
$$

得到：

$$
Q
\approx
[0.644,0.237,0.087,0.032]
$$

full KL 的梯度：

$$
dI=Q-P
$$

所以：

$$
dI
=
[0.544,0.037,-0.513,-0.068]
$$

然后你再乘：

$$
M=[1,1,0,0]
$$

得到：

$$
\boxed{
dI_{postmask}
=
[0.544,\ 0.037,\ 0,\ 0]
}
$$

这**不是** sparse KL 的梯度。

为什么？

因为前两个概率：

$$
0.644,\quad0.237
$$

在计算时仍然受后面：

$$
e^0,\quad e^{-1}
$$

影响。

Softmax denominator 是：

$$
e^2+e^1+e^0+e^{-1}
$$

也就是说：

> 你虽然最后不让 block 2、3 回传梯度，但它们已经参与了 block 0、1 的概率归一化。

所以太晚了。

---

# 4. 正确 sparse KL：先限定 Softmax domain

Top-2 选：

$$
[2,1]
$$

那么：

$$
Q_{sel}
=
Softmax([2,1])
$$

得到：

$$
Q_{sel}
=
[0.731,0.269]
$$

Teacher 也要在 selected set 内构成概率分布。

原：

$$
P=[0.1,0.2,0.6,0.1]
$$

选：

$$
[0.1,0.2]
$$

然后重新归一化：

$$
P_{sel}
=
\left[
\frac{0.1}{0.3},
\frac{0.2}{0.3}
\right]
$$

即：

$$
P_{sel}
=
[0.333,0.667]
$$

于是：

$$
dI_{sel}
=
Q_{sel}-P_{sel}
$$

得到：

$$
[0.398,-0.398]
$$

scatter 回 full shape：

$$
\boxed{
dI
=
[0.398,-0.398,0,0]
}
$$

和刚才：

$$
[0.544,0.037,0,0]
$$

明显不同。

---

# 5. 但 full-shape 实现完全可以得到正确结果

这正是你的想法经过修正之后很漂亮的地方。

仍然保留：

$$
I:[B,S_q,S_c]
$$

但先应用 mask：

$$
I'
=
[2,1,-\infty,-\infty]
$$

然后：

$$
Softmax(I')
$$

自然得到：

$$
Q'
=
[0.731,0.269,0,0]
$$

Teacher 也写成：

$$
P'
=
[0.333,0.667,0,0]
$$

那么：

$$
dI=Q'-P'
$$

直接：

$$
\boxed{
[0.398,-0.398,0,0]
}
$$

与 compact `[K]` 版本完全相同。

所以你真正可以统一成：

$$
\boxed{
dI=M\odot(Q_{masked}-P_{masked})
}
$$

这里关键是：

$$
\boxed{
Q_{masked}
\neq
M\odot Q_{dense}
}
$$

因为 Softmax normalization domain 已经不同。

---

# 6. 用数学严格证明 full-mask 与 `[K]` 等价

固定一个 query。

Top-K 集合：

$$
\mathcal S
$$

定义 full-shape masked logits：

$$
\tilde I_j
=
\begin{cases}
I_j,&j\in\mathcal S\\
-\infty,&j\notin\mathcal S
\end{cases}
$$

那么：

$$
\tilde Q_j
=
Softmax(\tilde I)_j
$$

对于：

$$
j\in\mathcal S
$$

有：

$$
\tilde Q_j
=
\frac{e^{I_j}}
{\sum_{r\in\mathcal S}e^{I_r}}
$$

对于：

$$
j\notin\mathcal S
$$

有：

$$
\tilde Q_j=0
$$

而如果直接 gather：

$$
I_{\mathcal S}
\in\mathbb R^K
$$

做：

$$
Q_{\mathcal S}
=
Softmax(I_{\mathcal S})
$$

显然：

$$
Q_{\mathcal S,j}
=
\frac{e^{I_j}}
{\sum_{r\in\mathcal S}e^{I_r}}
$$

所以：

$$
\boxed{
\tilde Q_{\mathcal S}=Q_{\mathcal S}
}
$$

完全一样。

---

# 7. KL 也完全一样

如果 teacher 在 selected set 上归一化为：

$$
\tilde P_j
=
\begin{cases}
P^{sel}_j,&j\in\mathcal S\\
0,&j\notin\mathcal S
\end{cases}
$$

full-shape KL：

$$
D_{KL}(\tilde P\|\tilde Q)
=
\sum_{j=1}^{S_c}
\tilde P_j
\log
\frac{\tilde P_j}{\tilde Q_j}
$$

因为：

$$
j\notin\mathcal S
$$

时：

$$
\tilde P_j=0
$$

所以实际上：

$$
=
\sum_{j\in\mathcal S}
P^{sel}_j
\log
\frac{P^{sel}_j}{Q^{sel}_j}
$$

正好：

$$
\boxed{
D_{KL}
(
P_{\mathcal S}
\|
Softmax(I_{\mathcal S})
)
}
$$

所以：

$$
\boxed{
full\ masked\ KL
\equiv
compact\ TopK\ KL
}
$$

---

# 8. DeepSeek 论文明确写的是 selected-domain KL

这个首先不是我们的猜测。

DeepSeek-V3.2 官方 sparse training 的 Eq. 4 是：

$$
\boxed{
\mathcal L^I
=
\sum_t
D_{KL}
\left(
p_{t,\mathcal S_t}
\|
Softmax(I_{t,\mathcal S_t})
\right)
}
$$

其中：

$$
\mathcal S_t
=
\{
s\mid I_{t,s}\in TopK(I_{t,:})
\}
$$

正文还明确说：

> sparse stage 中继续对齐 Indexer 与 main attention，但只考虑 selected token set。

所以从**数学定义**来看，DeepSeek 明确不是：

$$
D_{KL}(p_{t,:}\|Softmax(I_{t,:}))
$$

算完以后再随便把一部分 gradient 砍掉。

而是 KL 本身的定义域已经从：

$$
:
$$

缩成了：

$$
\mathcal S_t
$$

。([arXiv][2])

---

# 9. 更强证据一：Megatron-LM 的 DeepSeek DSA 实现恰好就是你说的 full-shape mask

这可能是现在最值得你看的代码。

NVIDIA Megatron-LM 已经有：

`experimental_attention_variant/dsa.py`

并且函数注释明确说：

> `sparse_loss=True`: only the top-k indices will be used to compute the loss. ([GitHub][1])

它首先仍然构造：

$$
index\_scores:
[B,S_q,S_k]
$$

以及：

$$
attention\_scores:
[B,H,S_q,S_k]
$$

然后构造一个：

$$
\boxed{
index\_mask:[B,S_q,S_k]
}
$$

初始化：

$$
-\infty
$$

再把 Top-K 位置 scatter 成有效值。([GitHub][1])

---

# 10. 最关键的是执行顺序

当：

```python
sparse_loss == True
```

代码先做：

```python
attention_scores += index_mask.unsqueeze(1)
index_scores += index_mask
```

也就是：

$$
[B,H,S_q,S_k]
$$

和：

$$
[B,S_q,S_k]
$$

在 **Softmax 之前** 把非 Top-K 设置成无效位置。([GitHub][1])

然后才：

```python
index_log_scores = masked_log_softmax(...)
```

teacher 也使用相同 selected valid mask 计算。

也就是说 Megatron reference 实现从 shape 看确实是：

$$
\boxed{
[B,S_q,S_k]
}
$$

没有先压成：

$$
[B,S_q,K]
$$

。

这和你说的“dense/sparse 统一 full shape”几乎完全一致。

---

# 11. 然后 backward 也确实在 full shape 上

NVIDIA 的 naive backward 明确保持：

$$
grad\_index\_scores:
[B,S_q,S_k]
$$

代码还明确写了 KL 对 logits 的精确梯度：

$$
\boxed{
predict\times target.sum(-1)-target
}
$$

如果 target 已归一化：

$$
target.sum(-1)=1
$$

就变成：

$$
\boxed{
predict-target
}
$$

然后代码又明确：

```python
# Zero out gradients for masked positions.
if sparse_loss:
    ...
grad_index_scores_logits *= valid_mask
```

即在最后保证非 Top-K：

$$
dI=0
$$

。([GitHub][1])

所以从**reference implementation**的视角，你完全可以画：

```text
Dense
M = causal_valid_mask

Sparse
M = causal_valid_mask & topk_mask

             I [B,Sq,Sc]
                   │
                 mask
                   │
                   ▼
           I_masked [B,Sq,Sc]
                   │
           masked softmax
                   ▼
             Q [B,Sq,Sc]
                   │
                  KL
                   │
                   ▼
            dI [B,Sq,Sc]
                   │
                 mask
                   ▼
            dI [B,Sq,Sc]
```

这个抽象我认为**非常合理**。

---

# 12. 甚至可以把 dense / sparse 写成同一个公式

定义：

$$
V_{\text{causal}}
\in\{0,1\}^{B\times S_q\times S_c}
$$

Dense stage：

$$
\boxed{
M_{\text{dense}}
=
V_{\text{causal}}
}
$$

Sparse stage：

$$
\boxed{
M_{\text{sparse}}
=
V_{\text{causal}}\odot M_{\text{topk}}
}
$$

统一：

$$
I^{mask}_{j}
=
\begin{cases}
I_j&M_j=1\\
-\infty&M_j=0
\end{cases}
$$

然后：

$$
Q=MaskedSoftmax(I,M)
$$

Teacher：

$$
P=MaskedNormalize(A,M)
$$

loss：

$$
L
=
D_{KL}(P\|Q)
$$

gradient：

$$
\boxed{
dI
=
M\odot(Q-P)
}
$$

忽略 loss reduction coefficient。

这就是一个非常干净的统一数学描述。

---

# 13. 但是高性能 kernel 为什么还是 `[B,Sq,K]`？

因为 full shape 在真实 V4 上太贵。

比如：

$$
S_q=64K
$$

$$
m=4
$$

所以：

$$
S_c=16K
$$

如果：

$$
B=1
$$

full score：

$$
[B,S_q,S_c]
=
[1,65536,16384]
$$

元素数量：

$$
65536\times16384
\approx1.07\times10^9
$$

光一个 FP32 tensor：

$$
\approx4.0GB
$$

你如果为了 sparse loss 还 materialize：

* index score；
* teacher score；
* probabilities；
* gradients；

显然非常浪费。

而 V4-Flash Top-K 只有：

$$
K=512
$$

那么：

$$
[B,S_q,K]
=
[1,65536,512]
$$

只有：

$$
33.6M
$$

个元素。

FP32：

$$
\approx128MB
$$

所以两者相差：

$$
\frac{16384}{512}
=
\boxed{32\times}
$$

---

# 14. 更强证据二：NVIDIA cuDNN 的生产级 DSA kernel 就直接使用 `[B,Sq,K]`

NVIDIA 最新 cuDNN Frontend 的 DSA 文档非常明确。

它专门区分：

### Sparse Indexer Score Recompute

输入 Top-K indices，输出：

$$
\boxed{
predict:[B,S_q,topk]
}
$$

定义就是：

$$
predict[b,q,i]
=
Softmax_i(
IndexerScore_{topk[i]}
)
$$

### Sparse Attention Score Recompute

输出 teacher：

$$
\boxed{
target:[B,S_q,topk]
}
$$

并且是 selected Top-K 上重新做归一化的 target。([NVIDIA Docs][3])

然后：

### Sparse Indexer Backward

直接消费：

$$
attn\_score:
[B,S_q,topk]
$$

$$
index\_score:
[B,S_q,topk]
$$

以及：

$$
topk\_indices:
[B,S_q,topk]
$$

再计算：

$$
dQ,\quad dW,\quad dK
$$

。([NVIDIA Docs][3])

---

# 15. 同时它还有单独的 DenseIndexBackward

文档另外提供：

### Dense Indexer Score Recompute

返回 full：

$$
[B,S_q,S_k]
$$

### Dense Indexer Backward

输入：

$$
attn\_score,index\_score:
[B,S_q,S_k]
$$

。([NVIDIA Docs][3])

这实际上非常漂亮地表明：

```text
数学逻辑：

Dense  = masked domain = 全部 causal positions
Sparse = masked domain = causal ∩ TopK


Reference实现：

都可以存成 [B,Sq,Sc]


生产kernel：

Dense  -> [B,Sq,Sc]
Sparse -> [B,Sq,K]
```

不是数学逻辑不同，而是**storage / compute representation 不同**。

---

# 16. 这份 NVIDIA 实现和 V4 有多大参考价值？

这是一个需要严谨区分的地方。

DeepSeek 自己确实没有公开其完整 pretraining / sparse training implementation；公开的 V3.2 repository 主要是 inference 示例代码，我也没有找到 DeepSeek 官方训练 loss/backward 的完整实现。([GitHub][4])

但 NVIDIA cuDNN Frontend 2026 年已经明确把这一套标为：

> DSA/CSA kernels for DSv4 and DSv3.2

而且包含：

* Indexer Forward；
* Top-K；
* sparse score recompute；
* dense score recompute；
* sparse Indexer backward；
* dense Indexer backward。

([GitHub][5])

所以它不是 DeepSeek 内部训练代码的“一比一源码证明”，但已经是目前非常强的**公开实现证据**。

---

# 17. 对 V4 来说，只需要把 \(S_k\) 换成 compressed \(S_c\)

V3.2 DSA：

$$
I:
[B,S_q,S_k]
$$

V4 CSA：

$$
\boxed{
I:
[B,S_q,S_c]
}
$$

其中：

$$
S_c\approx S_q/m
$$

V4：

$$
m=4
$$

所以统一表达就是：

### Dense Indexer warm-up

$$
M
=
M_{\text{causal}}
$$

$$
Q
=
MaskedSoftmax(I,M)
$$

shape：

$$
[B,S_q,S_c]
$$

---

### Sparse stage

$$
M
=
M_{\text{causal}}
\land
M_{\text{TopK}}
$$

数学上仍可以：

$$
Q
=
MaskedSoftmax(I,M)
$$

shape 保持：

$$
[B,S_q,S_c]
$$

然后：

$$
\boxed{
dI=M\odot(Q-P)
}
$$

但优化实现只存有效位置：

$$
Q_{compact}:
[B,S_q,K]
$$

于是后续 backward 只处理 K 个位置。

---

# 18. 还有一个容易忽略的地方：teacher 也必须使用相同 mask

不能只对 Indexer：

$$
I
$$

做 Top-K mask。

Teacher distribution：

$$
P
$$

也必须限制到同一个：

$$
\mathcal S_q
$$

否则两边不是定义在同一个 probability space 上。

NVIDIA Megatron 实现正是：

```text
attention_scores ── TopK mask ── softmax ── head sum ── L1 normalize
index_scores     ── TopK mask ── softmax
```

所以最终：

$$
P,Q
$$

共享：

$$
M_{\text{topk}}
$$

。([GitHub][1])

cuDNN sparse recompute 也是一样：

$$
target:[B,S_q,K]
$$

和：

$$
predict:[B,S_q,K]
$$

使用同一组：

$$
topk\_indices
$$

。([NVIDIA Docs][3])

---

# 19. 所以我会把你原来的说法改成这一句

你的原话大致是：

> sparse 不需要算 `[B,Sq,K]` 的 KL，可以算 `[B,Sq,Sc]` 全维度 KL gradient，最后 mask 决定回传位置。

更准确应该改成：

> **Sparse stage 可以在逻辑上继续使用 `[B,Sq,Sc]` 的 full tensor 表达，但 Top-K mask 必须参与 KL 两边的归一化，也就是先 masked-softmax / masked-normalize，再计算 KL。这样得到的 full-shape gradient 在非 Top-K 位置为 0，与 `[B,Sq,K]` compact KL 完全等价。**

也就是：

$$
\boxed{
\text{不是 }
M\odot
\left[
Softmax(I_{\rm full})-P_{\rm full}
\right]
}
$$

而是：

$$
\boxed{
M\odot
\left[
MaskedSoftmax(I,M)
-
MaskedNormalize(P,M)
\right]
}
$$

这个区别非常重要。

---

# 20. 我现在更推荐你这样理解整个训练

把 dense/sparse 的区别只归结为一个：

$$
\boxed{\text{valid-set}}
$$

Dense：

$$
\mathcal V_q
=
\{\text{所有 causal compressed blocks}\}
$$

Sparse：

$$
\mathcal V_q
=
\{\text{Top-K selected causal compressed blocks}\}
$$

然后统一：

$$
Q_j
=
\frac{
M_j e^{I_j}
}{
\sum_rM_r e^{I_r}
}
$$

Teacher：

$$
P_j
=
\frac{
M_j A_j
}{
\sum_rM_r A_r
}
$$

其中 \(A_j\) 是聚合后的 main-attention importance。

KL：

$$
L_q
=
\sum_j
P_j\log\frac{P_j}{Q_j}
$$

gradient：

$$
\boxed{
\frac{\partial L_q}{\partial I_j}
=
M_j(Q_j-P_j)
}
$$

这样：

* Dense：\(M=M_{\rm causal}\)
* Sparse：\(M=M_{\rm causal}\land M_{\rm TopK}\)

**整个数学和 backward 都完全统一。**

而在 kernel 层再决定：

* dense 用 `[B,Sq,Sc]`；
* sparse 把有效位置 compact 成 `[B,Sq,K]`。

我认为这是目前最准确、也最适合你后面看 CUDA/CuTe backward kernel 的理解方式。([模型范围][6])

[1]: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/experimental_attention_variant/dsa.py "Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py at main · NVIDIA/Megatron-LM · GitHub"
[2]: https://arxiv.org/abs/2512.02556?utm_source=chatgpt.com "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
[3]: https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html?utm_source=chatgpt.com "DeepSeek Sparse Attention (DSA) — NVIDIA cuDNN"
[4]: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py?utm_source=chatgpt.com "DeepSeek-V3.2-Exp/inference/model.py at main · deepseek-ai/DeepSeek-V3.2-Exp · GitHub"
[5]: https://github.com/NVIDIA/cudnn-frontend/releases?utm_source=chatgpt.com "Releases · NVIDIA/cudnn-frontend · GitHub"
[6]: https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.2/resolve/master/assets/paper.pdf?utm_source=chatgpt.com "<visual_element id=\"e1\">"
