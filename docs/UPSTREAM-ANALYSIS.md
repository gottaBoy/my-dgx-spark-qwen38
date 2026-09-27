# 三个已落地项目的分析,以及我从这里抄了什么

本文只记录**读代码和跑机器得到的事实**,不记录计划。每条都标了来源:仓库文件、或本
机实测。凡是我先推测、后被实测否证的,都写在了"我推断错的地方"一节里 —— 那部分是
这份分析里最有价值的东西。

三个 fork 都在本地目录:
[Qwen3.8-27B-SGLang-DGX-Spark](/home/my/workspace/llm/Qwen3.8-27B-SGLang-DGX-Spark)、
[dgx-spark-qwen38](/home/my/workspace/llm/dgx-spark-qwen38)、
[qwen38-27b-nvfp4-sm121-sglang](/home/my/workspace/llm/qwen38-27b-nvfp4-sm121-sglang)。

## 三个项目各自的实质

| | MiaAI-Lab | hasso5703 | r0b0tlab |
|---|---|---|---|
| 规模 | 14 个受跟踪文件 | 264 文件 / 4.4 万行 Python | 87 文件 |
| 真正强的地方 | 实测数据与测量方法学 | 工程化与运维面 | 评测协议与证据可复核性 |
| 主要交付 | 三个可换引擎的 Docker 启动脚本 | 收敛式安装器 + systemd + 代理 + cockpit | digest pin + click-run + 语义门禁 |
| 对机器的假设 | 独占 | 独占 | 独占(单机一次一模式) |

一句话:三家都把这台的机器当成自己的。这台不是 —— 它有 25 个常驻容器、没有任何
OOM 守护(`systemd-oomd` 与 `earlyoom` 都是 inactive),而 `nvidia-smi` 在 GB10 上
取不到显存。所以本仓库的全部差异都围绕这一件事。

## 从 MiaAI 抄的(它的实测最扎实)

| 抄来的东西 | 出处 | 备注 |
|---|---|---|
| 两调用净解码差分测吞吐 | `bench/ndec.py` | 同 prompt 跑 60 与 600 token 再相减,消掉 prefill 与模板固定开销 |
| 丢弃引导后第一次测量 | `bench/ab-image.sh` | 它观测到 89 / 183 / **−371** tok/s 这类冷启动伪值 |
| 交错实验顺序 `A B B A` | `ab-image.sh` | 同镜像两次引导差 6.5%,不交错就是在测机器的漂移 |
| 代码差分 <15% 记为噪声 | README「Measured on this box」 | 我把它写进了 `bench` 的脚注 |
| SSE 事件数 ≠ token 数 | README 计数勘误节 | DFlash2 每事件约 3.75 token;必须读 `completion_tokens` |
| 单表原则:性能数字只有一个家 | README | 我的 `runs`/`compare` 同理,数字必须连着配置一起存在 |
| `--mamba-full-memory-ratio` 要显式设 | `start.sh` | **但我抄错了它的作用,见下文** |
| 完整 40 位 sha 作 draft pin | `start-dflash.sh` | 我原先 pin 的是短 sha |
| 镜像按 index digest pin、并给「如何换 pin」两行 | `start-dflash.sh` + plan 文档 | 我的 `pull` 就是这个检查 |
| YaRN 走 `--json-model-override-args` + 一个容器环境变量 | `start.sh` | 只给 override 不给环境变量 = 静默停在 262K |

MiaAI 最有价值的一点是它诚实:README 里明写「MTP 那一列是过期的,8 月的数字属于
FP4-head 时代,别和上面的表混用」。这种自我标注在我的实现里对应 `compare` 先列配置
差异、再列数字差异。

## 从 hasso 抄的(它的运维面最完整)

| 抄来的东西 | 出处 | 我落在哪 |
|---|---|---|
| flashinfer autotuner 每boot重选内核 →「boot lottery」 | BENCHMARKS.md「The boot lottery」 | `--disable-flashinfer-autotune` 进 CORE_FLAGS |
| `--cuda-graph-max-bs` 与 `--max-running-requests` 必须配套 | 同上,+6.5% c8 | `dflash2` profile |
| 编译图上限按实际 batch 形状封顶 | `--torch-compile-max-bs 4` | 同上 |
| 不要原生跑 SGLang,必须容器 | docs/gb10-memory.md | 全部走容器 |
| `Restart=always` 而非 `on-failure`:崩溃可能以 `SystemExit: 0` 结束 | 同上 | 见「我的取舍」 |
| 显式写死"这台机器哪些做法会杀死别的容器" | README 的 GB10 trap 节 | `doctor` + `guard` + 实算 fraction |
| 单位/常量单一事实源,CI 守住多处拼写一致 | ARCHITECTURE.md 的 12 条不变量 | 我用一条测试实现同类保护(见下) |
| 权重必须宿主机预取,容器内 `HF_HUB_OFFLINE=1` | `qwen38-sglang.service.template` | `prefetch` 子命令 + `Q38_OFFLINE` |

## 从 r0b0tlab 抄的(它的评测最严格)

| 抄来的东西 | 出处 |
|---|---|
| 语义门禁:短、确定、能暴露投机缺陷的探针(`19×23→437`,`417` 是 FP8-KV 缺陷) | `scripts/canary_437.py` |
| 每个 profile 的完整 argv 落成一个函数,不为跑而手拼 | `scripts/serve.sh` |
| 结果以 JSON 提交进仓库,附 image digest 与 checkpoint 校验和 | `perf-summary-*.json`、`official-nvfp4-shards.sha256` |
| 明确禁止某种打包:`unsloth/…`(sglang#34895)、vLLM 的 `Qwen3DSparkModel` | README 开头 |
| 同一 checkpoint 上 K=8 最优、K=9 崩 | `r0b0bench-core-subset-*.json` |

## 本机的三个实测结论

1. **并发被静默钳制,原因是 GDN state 槽位不足。** 引擎的算术写在 pin 住的那个镜像
   里,`kv_cache_configurator._calculate_mamba_ratio`:

   ```
   ratio = 3 (+1 若 extra_buffer 且 overlap 开;+2 若非 lazy)
   max_running_requests = min(请求值, max_mamba_cache_size // ratio)
   ```

   即 `extra_buffer` → 5 槽/请求,`extra_buffer_lazy` → 4。用三个引导校验:

   | 引导 | 策略 | 槽位 | 请求并发 | 授予 | 公式 |
   |---|---|---|---|---|---|
   | dflash2 | `extra_buffer` | 32 | 8 | **6** | `32//5=6` |
   | dflash2 | `extra_buffer` | 96 | 8 | 8 | `96//5=19≥8` |
   | ar | `extra_buffer_lazy` | 32 | 8 | 8 | `32//4=8` |

   所以 `mamba_slots` 在本仓库是**从策略推导的属性**,不是常量。

2. **AR(无投机)基线:code 8.06 / essay 9.97 tok/s。** 对照 DFlash2 的 47.05 / 23.24,
   即 code **5.8×**、essay **2.3×**。这个比值和带宽物理一致:22.14 GiB 权重、273 GB/s
   标称带宽 → 约 10 tok/s 的裸解码上限。三家都没发过纯 AR-only 数字,这是本仓库第一个
   自有测量。

3. **`--enable-torch-compile` 的代价在本机是 449 秒就绪 vs 117 秒。** 这正是 hasso
   记录过的启动/吞吐交换,我把它抄进 profile 时同时把代价写进了注释。

## 我推断错的地方(全部由实测或源码纠正)

这部分比前面几条更有价值,因为它们都是"看起来合理的推理"造成的。

1. **我把并发钳制归因给了 `--mamba-full-memory-ratio`。** 依据是 MiaAI `start.sh` 的
   注释「default 0.9 over-provisions KV and silently clamps concurrency」,于是我照抄
   了 `4.2` 并把这句因果写进了自己的诊断文案。**实测否证**:传 `4.2` + 32 槽,授予仍是
   6;换成 96 槽才变 8。真因是槽位数量,而 hasso 根本**不设这个 flag**。已改。

2. **`mamba_slots` 我先设 4(理由是"过量供给更安全"),后拍 8(理由是"留余量")。**
   两个都是猜。正确做法是把引擎自己的常量读出来:`extra_buffer` 要 5、lazy 要 4,
   余量另算。已改为推导属性。

3. **`draft revision` 我曾拼成 `repo@50307d4` 塞进 `--speculative-draft-model-path`。**
   HuggingFace 会去找一个真名叫 `repo@50307d4` 的仓库,几分钟加载后才失败。两家上游
   都用专门的 `--speculative-draft-model-revision` 且是完整 40 位 sha。

4. **`docker images -q <digest>` 判"镜像在不在"是错的。** 本机实测:`-q` 返回空,而
   `docker run` 同一引用正常。改 `docker image inspect` + 退出码,否则 `start` 会拒绝
   一个本可成功的启动。

5. **本地 HTTP 请求会走代理。** 这台机器设了 `HTTP_PROXY`,而 `no_proxy` 里的 `127.*`
   **不被 Python 当通配符**(它按后缀匹配),所以对健康引擎的请求从代理返回 502,
   看起来和引擎崩了完全一样。同时 Hub API 又**只能**经代理到达(实测:直连超时,经代理
   4.6 秒)。规则按目标区分,收在 `net.py` 一处。

6. **`/proc/meminfo` 单位是 kB。** 我第一次按字节除,健康的 119 GB 机器读成
   `MemAvailable 0.09 GiB`,于是每次启动都被拒。

7. **`"active" in "inactive"`。** `doctor` 的 OOM 守护检查用子串判断,把"没有任何保护"
   的机器报成"有保护"。这条机器恰好就是那种机器。

## 我刻意不抄的

| 不抄 | 为什么 |
|---|---|
| hasso 的 keepalive 代理 + cockpit + 七个 target | 与"在这台共享机器上安全启动"无关;我一次只需要一个车道 |
| hasso 的镜像内 patch 层(`flash-sglang/`) | 已被上游吸收,自己维护 patch 是长期负债 |
| `--privileged`(MiaAI) | 需要的是设备与 IPC,不是主机控制权 |
| 绑 `0.0.0.0` 且无鉴权(MiaAI) | 这台机器上还有别人的数据库和网关 |
| `Restart=always`(hasso) | 见下 |

**`Restart=` 这条要说清取舍。** hasso 需要 `always`,因为它的 `ExecStart` 直接就是
`docker run`,引擎以 0 退出(Triton 编译崩溃实测如此)对 systemd 就是"成功",
`on-failure` 不会重拉。我这里 `ExecStart` 是一个跟随容器日志的 supervisor,它能问一
个 unit 问不出的问题:这次消失是不是有人要求的。所以我保留 `on-failure` + 显式
`SuccessExitStatus=137 143`,把"区分主动停止与崩溃"的责任放在 wrapper 里。代价是:
如果 supervisor 自己以 0 退出,unit 会显示 inactive 而容器已死 —— 这条必须在代码里
堵掉,不是靠 unit 的 `always` 兜住。

## 一处仍未解决、且我不打算自己造的问题

**僵尸请求。** 客户端中途断连后,引擎会继续解码到 `max_tokens`,占住一个并发槽。
两家都独立撞上:MiaAI 观测到一次断连解码 16.75 分钟、6,466 行刷屏;hasso 一天 6,582
行、一个请求为不存在的人解了 6 分钟。上游修复是 sglang#35255(2026-09-04 合入)。

- MiaAI 的解法:换镜像(`dev-qwen38-27b-dflash2` 早于该修复,它已 bump 到
  `nightly-cu134-20260909`);
- hasso 的解法:在前面加代理,自己给每个请求命名(`x-override-rid`)并在关连接**之前**
  发 abort。

**我 pin 的镜像不含这个修复。** 我没有代理层,所以只有换镜像这一条路,而那意味着重拉
13.4 GiB,并且 MiaAI 新的 pin 是**单架构 manifest**(本机核实:`00205b89` 只有 arm64,
而我 pin 的 `616a3e97` 是 amd64+arm64 的 index)。这个决定该由使用这台机器的人做,
所以我把它写在这里并把 `doctor` 的检查留着,而不是替你换。

## 我把它们的"CI 不变量"简化成了一条测试

hasso 用 12 条 CI 不变量守"同一份清单被写四遍"(selector 选项、action 枚举、
install.sh 的 case、cockpit 的 BUILTIN_IDS)。我只有一个 CLI,所以不需要那些;但同类
问题在别处咬到我:注释和报错里印出的命令名会失效。

`tests/test_cli.py` 因此做两件事:命令清单**从 `--help` 解析**(不维护第二份),以及
扫遍源码/文档里所有 `qwen38 <cmd>`,断言它真实存在。这条测试上线当天就抓到
`qwen38 preflight`(一条我写了、但从未实现的建议)。 <!-- name-check:ignore -->

另一个同类:`tests/test_no_dead_blocks.py` 用 AST 找「`return` 之后还有语句」和被遮蔽的
同名 `def`。这类缺陷能编译、能通过任何行为测试,我确实造出过一次。
