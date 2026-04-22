我看完了。结论先说：

从 **v2.1.0 到现在**，`transform_sf_into_required_layout` 的**接口形态一共经历了 2 次演进**，所以可以分成你说的 **3 个版本阶段**：

1. **早期版本**：只有一个 `tuple[int,int,int] recipe`
2. **中间过渡版本**：同时支持 `recipe` 和 `recipe_ab`
3. **当前版本**：统一成一个 `variant recipe`，即单个参数里区分 `tuple[int,int,int]` 和 `tuple[int,int]`

另外，当前 Python 层 `deep_gemm/utils/layout.py` 已经基本不再实现逻辑，只是从扩展模块 `.._C` 里 re-export；真正实现现在在 `csrc/apis/layout.hpp`。 

---

## 0. 时间线和对应代码阶段

`v2.1.0` 和 `v2.1.1` 都存在于仓库 tag 历史里；而把 `recipe_ab` 改成当前 `variant recipe` 的，是 PR **#304**，这个 PR 在 **2026-04-17** 合并进主分支。([GitHub][1])

所以可以按下面理解：

* **阶段 A：v2.1.0 / v2.1.1**
* **阶段 B：v2.1.1 之后、PR #304 合并之前的主线版本**
* **阶段 C：PR #304 合并后的当前版本**

其中 `v2.1.0` 和 `v2.1.1` 这两个 tag 下，这个函数的签名和内部实现是一样的。 

---

## 1. 版本一：只有 `recipe: tuple[int,int,int]`

### 接口

在 `v2.1.0` / `v2.1.1`，函数签名是：

```cpp
transform_sf_into_required_layout(
    sf,
    mn,
    k,
    recipe: std::tuple<int, int, int>,
    num_groups: optional<int>,
    is_sfa: bool,
    disable_ue8m0_cast: bool
)
```

也就是 Python 侧等价于：

```python
transform_sf_into_required_layout(
    sf, mn, k, recipe,
    num_groups=None,
    is_sfa=False,
    disable_ue8m0_cast=False
)
```

这里 `recipe` 是一个 **3-tuple**，函数内部会按 `is_sfa` 去解释这个 tuple：

* 如果是 `SFA`，取 `recipe[0]`
* 如果是 `SFB`，取 `recipe[1]`
* `recipe[2]` 始终是 `gran_k` 

对应代码核心就是：

```cpp
const auto& gran_mn = is_sfa ? std::get<0>(recipe) : std::get<1>(recipe);
const auto& gran_k = std::get<2>(recipe);
```



### 这一版的特点

这一版的思路很直接：

* **一个统一 recipe** 同时描述 A/B 两边的缩放布局
* 当前处理的是哪一边，通过 `is_sfa` 告诉函数
* 所以它本质上假设：**A 和 B 共用一个“三元 recipe 规范”**

### 内部实现特征

这一版内部逻辑比较“写死”：

#### 1）只支持 `gran_k = 128`

SM100 分支里只处理 `128`，没有 `32`。

#### 2）SM100 对 FP32 的处理分两条硬编码路径

* `(FP32, 1, 128)`：直接 pack 成 UE8M0 + TMA aligned + MN-major
* `(FP32, 128, 128)`：先把 128 粒度 broadcast 到逐行，再 pack

代码里是分开的两段：

```cpp
if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and arch_major == 10)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(sf);

if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and arch_major == 10) {
    const auto& broadcasted = sf.index_select(... floor_divide_(128));
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
}
```



#### 3）INT 分支也只接受 `(1,128)`

```cpp
if (sf.scalar_type() == torch::kInt and gran_mn == 1 and gran_k == 128 and arch_major == 10)
    return check_sf_layout(... torch::kInt);
```



#### 4）`k_grouped` 版本也被固定死为 `(1,1,128)`

```cpp
DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
```

并且 SM100 下调用的 pack helper 也没有 `gran_k` 参数。

---

## 2. 版本二：加入 `recipe_ab`

这个阶段不是 `v2.1.0` tag，而是后续主线里的过渡状态。你说的现象是对的：这一版 `transform_sf_into_required_layout` 开始同时接受两种 recipe 形式。对应代码我用 PR #304 合并前的 base commit 抓到了。

### 接口

这一版函数签名变成：

```cpp
transform_sf_into_required_layout(
    sf,
    mn,
    k,
    recipe: optional<std::tuple<int, int, int>>,
    recipe_ab: optional<std::tuple<int, int>>,
    num_groups: optional<int>,
    is_sfa: bool,
    disable_ue8m0_cast: bool
)
```

也就是：

* 要么传老的 `recipe` 三元组
* 要么传新的 `recipe_ab` 二元组

不能两个都用。函数内部显式检查了这一点：

* 有 `recipe` 时，要求没有 `recipe_ab`
* 没有 `recipe` 时，要求 `recipe_ab` 必须有值 

核心逻辑：

```cpp
if (recipe.has_value()) {
    DG_HOST_ASSERT(not recipe_ab.has_value());
    gran_mn = is_sfa ? std::get<0>(recipe.value()) : std::get<1>(recipe.value());
    gran_k = std::get<2>(recipe.value());
} else {
    DG_HOST_ASSERT(recipe_ab.has_value());
    std::tie(gran_mn, gran_k) = recipe_ab.value();
}
```



### 这版和第一版的本质区别

### 区别 1：开始支持“单边 recipe”

`recipe_ab` 是个 **2-tuple**，本质就是：

```python
recipe_ab = (gran_mn, gran_k)
```

这意味着：

* 调这个函数时，传进来的这个 `sf` 已经是单边的了
* 不再需要依赖“三元 recipe + is_sfa 去区分 A/B”
* 也就是从“**一个 recipe 管 A/B 两边**”，演进到了“**对当前这一边直接给出自己的 recipe**”

这是接口语义上最重要的变化。

### 区别 2：为 `SFA/SFB` 成对处理做准备

同一版里新增了一个 helper：

```cpp
transform_sf_pair_into_required_layout(
    sfa, sfb, m, n, k,
    optional<tuple<int,int,int>>& recipe,
    optional<tuple<int,int>>& recipe_a,
    optional<tuple<int,int>>& recipe_b,
    ...
)
```

这个 helper 的意思非常明确：

* 仍然支持老式的统一 `recipe`
* 也支持 **`recipe_a` + `recipe_b`** 这种 A/B 分别指定的形式

而它内部在调用 `transform_sf_into_required_layout` 时：

* 如果走统一 `recipe`，就传 `recipe` + `is_sfa=true/false`
* 如果走拆开的 recipe，就把 `recipe_a` / `recipe_b` 分别作为 `recipe_ab` 传下去 

也就是说：

* **公开的单边函数** 用的是 `recipe_ab`
* **内部的双边 helper** 用的是 `recipe_a` / `recipe_b`

这两个命名你要区分开。

### 区别 3：内部逻辑从“只支持 128”扩展到“支持 32 或 128”

这是这版内部实现里最关键的功能变化之一。

在这一版里，SM100 路径从：

```cpp
gran_k == 128
```

扩成了：

```cpp
(gran_k == 32 or gran_k == 128)
```

无论 FP32 还是 INT，都是这样。

### 区别 4：SM100 的 FP32 分支被泛化了

以前版本对 `(gran_mn=1, gran_k=128)` 和 `(gran_mn=128, gran_k=128)` 写了两段分支。
这版把它统一成了“先看 `gran_mn` 是否为 1，不是就 broadcast，再 pack”。

核心代码：

```cpp
if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
    const auto& broadcasted = gran_mn == 1 ? sf :
                              sf.index_select(-2, torch::arange(mn, ...).floor_divide_(gran_mn));
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
}
```

相比第一版，这里有两个明显变化：

* `gran_k` 从固定 128 扩成了 32/128
* `gran_mn` 不再只写死支持 1 和 128 两种硬编码分支，而是统一成 `gran_mn==1 ? sf : broadcast(...)`



### 区别 5：但 `k_grouped` 这一段此时还没完全跟上

虽然单边 `transform_sf_into_required_layout` 已经支持 `gran_k=32`，但当时 `transform_k_grouped_sf_into_required_layout` 还是：

```cpp
DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
```

并没有放开到 32。

---

## 3. 版本三：统一成 `variant recipe`（当前版本）

这就是当前主分支的形态。PR #304 的 patch 非常清楚地显示了这次改动：把中间版的两个参数 `recipe` / `recipe_ab`，合并成了一个 `variant` 类型的 `recipe`。

### 接口

当前签名：

```cpp
transform_sf_into_required_layout(
    sf,
    mn,
    k,
    recipe: std::variant<std::tuple<int,int,int>, std::tuple<int,int>>,
    num_groups: std::optional<int>,
    is_sfa: std::optional<bool>,
    disable_ue8m0_cast: bool
)
```

对应 pybind 注册也改成只暴露一个 `recipe` 参数：

```cpp
m.def("transform_sf_into_required_layout", ...,
    py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
    py::arg("num_groups") = std::nullopt,
    py::arg("is_sfa") = std::nullopt,
    py::arg("disable_ue8m0_cast") = false);
```



### 这一版和中间版的本质区别

### 区别 1：把“两个互斥参数”收敛成“一个判别联合参数”

中间版是：

* `recipe: Optional[tuple3]`
* `recipe_ab: Optional[tuple2]`

当前版变成：

* `recipe: tuple3 | tuple2`

这相当于把“API 层的互斥关系”从**运行时两个参数互斥**，变成了**类型层面的二选一**。
接口更干净，也更不容易误传。

### 区别 2：`is_sfa` 也同步变成了 `optional<bool>`

因为当前版里 `recipe` 可能是两种形态：

* 如果 `recipe` 是 `tuple3`，就必须知道当前是 A 还是 B，所以 `is_sfa` 必须有值
* 如果 `recipe` 是 `tuple2`，它已经是单边 recipe 了，就**不该再传 `is_sfa`**

当前实现明确这样写了：

```cpp
if (auto p = std::get_if<std::tuple<int, int, int>>(&recipe)) {
    DG_HOST_ASSERT(is_sfa.has_value());
    gran_mn = is_sfa.value() ? std::get<0>(*p) : std::get<1>(*p);
    gran_k = std::get<2>(*p);
} else if (auto p = std::get_if<std::tuple<int, int>>(&recipe)) {
    DG_HOST_ASSERT(not is_sfa.has_value());
    std::tie(gran_mn, gran_k) = *p;
}
```



这比中间版更严格、更清晰。

### 区别 3：`transform_sf_pair_into_required_layout` 也一起收敛了

当前版的 pair helper 仍然保留：

* 统一 `recipe`（3-tuple）
* 或 `recipe_a` + `recipe_b`（两个 2-tuple）

但它内部不再把“二元 recipe”塞给 `recipe_ab` 参数，而是直接把 `tuple2` 作为 `variant recipe` 传下去。

而且当前版多加了更清楚的约束：

```cpp
DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
DG_HOST_ASSERT(recipe_a.has_value() != recipe.has_value());
```

意思是：

* 要么用统一 `recipe`
* 要么用 `recipe_a + recipe_b`
* 不能混用，也不能缺一边



### 区别 4：`k_grouped` 这次也补齐了 32 支持

当前版里：

```cpp
DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);
const int gran_k = std::get<2>(recipe);
DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
```

然后在 SM100 下调用：

```cpp
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, gran_k);
```

也就是：

* 不再固定只能 `(1,1,128)`
* 现在 `(1,1,32)` 也合法
* downstream helper 也显式接收 `gran_k`

这是相对中间版又前进了一步。

---

## 4. 三个版本的输入区别，直接对照

### 版本一：统一三元组

```python
transform_sf_into_required_layout(
    sf, mn, k,
    recipe=(gran_mn_a, gran_mn_b, gran_k),
    num_groups=None,
    is_sfa=True/False,
    disable_ue8m0_cast=False
)
```

含义：

* `recipe[0]` 给 A
* `recipe[1]` 给 B
* `recipe[2]` 给 K 方向
* 当前处理 A 还是 B，靠 `is_sfa` 决定



---

### 版本二：两种写法并存

```python
# 老写法
transform_sf_into_required_layout(
    sf, mn, k,
    recipe=(gran_mn_a, gran_mn_b, gran_k),
    recipe_ab=None,
    num_groups=None,
    is_sfa=True/False,
    disable_ue8m0_cast=False
)

# 新写法
transform_sf_into_required_layout(
    sf, mn, k,
    recipe=None,
    recipe_ab=(gran_mn, gran_k),
    num_groups=None,
    is_sfa=True/False,   # 形式上还在，但 recipe_ab 分支本质不需要它
    disable_ue8m0_cast=False
)
```

含义：

* 兼容老接口
* 新增单边 recipe 表达法



---

### 版本三：统一成一个 `variant`

```python
# 三元组形式
transform_sf_into_required_layout(
    sf, mn, k,
    recipe=(gran_mn_a, gran_mn_b, gran_k),
    num_groups=None,
    is_sfa=True/False,
    disable_ue8m0_cast=False
)

# 二元组形式
transform_sf_into_required_layout(
    sf, mn, k,
    recipe=(gran_mn, gran_k),
    num_groups=None,
    is_sfa=None,
    disable_ue8m0_cast=False
)
```

含义：

* API 只有一个 `recipe`
* 传三元组时表示“统一 recipe”
* 传二元组时表示“单边 recipe”
* `is_sfa` 只有在三元组模式下才合法



---

## 5. 函数内部实现变化，按“核心逻辑”总结

### 第一阶段 → 第二阶段

主要变化有 3 个：

#### A. 参数解释逻辑从“只认 triple”变成“triple / pair 双通道”

这是接口层最大的变化。 

#### B. SM100 支持从 `gran_k=128` 扩展到 `gran_k in {32, 128}`

这是功能层最大的变化。 

#### C. FP32 的 SM100 路径从“两个硬编码分支”变成“一个泛化 broadcast + pack 路径”

代码更通用，也更适配 FP4/新布局需求。 

---

### 第二阶段 → 第三阶段

主要变化有 4 个：

#### A. `recipe + recipe_ab` 两个互斥参数，合并成一个 `variant recipe`

API 更整洁。

#### B. `is_sfa` 从 `bool` 变成 `optional<bool>`

使它和 `variant recipe` 的两种模式严格对应。 

#### C. `transform_sf_pair_into_required_layout` 的分派逻辑更严格

明确“要么统一 recipe，要么 recipe_a+recipe_b”。

#### D. `k_grouped` 路径终于也支持 `gran_k=32`

并把 `gran_k` 透传到下层 pack helper。 

---

## 6. 我给你的最终归纳

你可以把这三个版本理解成下面这条演化线：

### 第一版：**统一 recipe 驱动**

* 只支持 `(a_mn, b_mn, k)`
* 当前是 A 还是 B，靠 `is_sfa`
* SM100 只处理 `k=128`

### 第二版：**开始支持单边 recipe**

* 在保留老 triple recipe 的同时，加了 `recipe_ab=(mn, k)`
* 内部开始适配 FP4 / `gran_k=32`
* 但接口上有两个互斥参数，比较别扭

### 第三版：**类型统一**

* 把 `recipe` / `recipe_ab` 合并成一个 `variant recipe`
* 三元组表示“统一 recipe”
* 二元组表示“单边 recipe”
* `is_sfa` 也随之变成可选，只在三元组模式下使用
* `k_grouped` 路径也补齐到 32/128

  

---

## 7. 一句话回答你最关心的点

**从 v2.1.0 到现在，这个函数接口本质上改了 2 次：**

* **第一次**：从“只接受统一 triple recipe”变成“同时接受 triple recipe 和单边 `recipe_ab`”
* **第二次**：把前面的“双参数互斥接口”收敛成“单个 `variant recipe` 接口”

同时，内部实现也从**只支持 SM100 下 `gran_k=128` 的硬编码分支**，演进成了**支持 `gran_k=32/128` 的通用化路径**，并把这套能力扩展到了 pair helper 和 k-grouped helper。  

如果你愿意，我下一条可以继续给你整理成一张“**三版本签名 + 输入语义 + 内部分支差异**”的对照表。

[1]: https://github.com/deepseek-ai/DeepGEMM/tags "Tags · deepseek-ai/DeepGEMM · GitHub"
