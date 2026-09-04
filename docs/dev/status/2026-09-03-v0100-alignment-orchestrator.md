# InferRef 0.10.0 Alignment Orchestrator 实现完成状态报告

- **日期**: 2026-09-03
- **版本目标**: InferRef 0.10.0
- **环境**: Windows 11 25H2, PowerShell 7, Python 3.13.11
- **规范基准**: `docs/spec/InferRef_0.10_Alignment_Orchestrator.md`
- **前序移交**: `docs/dev/status/2026-09-03-v0100-wave1-handoff.md`

---

## 1. 任务完成概览 (Layers 1-9)

| 层级 | 模块/功能 | 状态 | 核心交付物 |
|---|---|---|---|
| **L1** | 状态快照、事件与存储 | `COMPLETED` | `inferref/alignment/schema.py`, `inferref/alignment/store.py` (修复 Win32 PID 无损探测) |
| **L2** | 边界发现、图谱与规划 | `COMPLETED` | `inferref/alignment/boundaries.py`, `inferref/alignment/planner.py` (修复空区域与无损序列化) |
| **L3** | 证据规约与治具矩阵 | `COMPLETED` | `inferref/alignment/evidence.py`, `inferref/alignment/matrix.py` |
| **L4** | 执行编排与用例缓存 | `COMPLETED` | `inferref/alignment/runner.py`, `inferref/alignment/testcase_cache.py` |
| **L5** | 因果收敛与归因分类 | `COMPLETED` | `inferref/alignment/narrowing.py`, `inferref/alignment/classification.py` |
| **L6** | CLI 命令组 (`inferref align`) | `COMPLETED` | `inferref/cli/align.py`, `inferref/cli/main.py` |
| **L7** | Agent 门面与 MCP 工具 | `COMPLETED` | `inferref/agent/service.py`, `inferref/agent/mcp_server.py` |
| **L8** | Skill 参考与工作流检查器 | `COMPLETED` | `skills/inferref/references/cli.md`, `scripts/check_agent_workflow_skill.py` |
| **L9** | 确定性 E2E 场景与回归门禁 | `COMPLETED` | `tests/core/test_alignment_e2e.py`, 全量验收测试集通过 |

---

## 2. 关键缺陷修复与规范对齐细节

1. **P0 Windows PID 无损探测修复 (`store.py`)**:
   - 原代码中 `_lock_is_stale()` 在 Windows 环境调用 `os.kill(pid, 0)`，会向目标进程请求 `PROCESS_TERMINATE` 权限，在 Windows 上导致误杀 pytest 进程。
   - 采用 Win32 `ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)` 结合 `GetExitCodeProcess` 探测，并确保句柄及时 `CloseHandle`，实现非破坏性存活探测。

2. **边界空区域排除与无损序列化 (`boundaries.py`)**:
   - 在 `_raw_boundaries()` 中增加 `if not nodes: continue`，排除包含 0 个算子的空 Region。
   - 在 `BoundaryInventory.to_dict()` 与 `from_dict()` 中增加对 `raw_boundaries` 和 `aliases` 的双向序列化与重建，保留原始边界记录与图遍历索引一致性。
   - 为 `BoundaryInventory` 增加 `__contains__` 操作符支持。

3. **证据模式与观察字段规范 (`evidence.py`)**:
   - 严格约束 `evidence.observations` 白名单键集合 (`provided_inputs_valid`, `engine_inputs_observed`, `engine_input_observation`, `outputs_match`, `layout_match`, `first_failure`, `output_roles`, `comparison_status`, `metrics`, `diagnostics`)。
   - 严禁出现 `inputs_match`；不支持信息作为结构化列表放入 `diagnostics` 中。

4. **治具矩阵事件顺序与崩溃恢复 (`matrix.py`)**:
   - 以事件序号而非秒级 `created_at` 或文件名确定当前证据，保证重载前后结果一致。
   - 未被事件引用的崩溃孤立证据不参与投影；提交前结果必须通过显式 `prospective_evidence` 参数加入。

5. **适配器哈希一致性 (`hashing.py`)**:
   - 统一 `_adapter_public_record()` 中对文件路径与对象实例的加载逻辑，确保 `adapter_normalized_sha256(adapter_path)`、`normalized_adapter_sha256(adapter)` 与 `normalized_adapter_sha256(adapter.to_dict())` 产生 100% 相同的结果，杜绝重新加载时误判 stale。

6. **POSIX 虚拟环境引擎身份 (`runner.py`)**:
   - 受信任适配器使用 `{python}` 时，先解析 POSIX venv 的解释器符号链接，再对实际可执行文件和内联脚本计算稳定身份；通用 artifact 哈希仍保持拒绝符号链接的安全策略。

---

## 3. 修改与新建文件清单

### 修改文件 (Existing Files Modified)
- `inferref/agent/mcp_server.py`: 注册 8 个 `inferref_align_*` 工具，绑定路径安全策略。
- `inferref/agent/service.py`: 暴露 8 个 `align_*` 核心门面函数，并在 `capabilities()` 中注册对应操作与格式。
- `inferref/cli/main.py`: 注册 `align` 命令组与帮助文档。
- `scripts/check_agent_workflow_skill.py`: 在 `COMMAND_GROUPS` 中增加 `align`。
- `skills/inferref/references/cli.md`: 补充 `inferref align` CLI 命令与 MCP 工具说明。

### 新增模块与测试文件 (New Files)
- `inferref/alignment/__init__.py`: 导出全部公共 API（共 62 个符号）。
- `inferref/alignment/schema.py`: 严格模式与快照/事件/证据定义。
- `inferref/alignment/boundaries.py`: 边界发现、图解析与因果传递。
- `inferref/alignment/planner.py`: 启发式规划器与运行资格评估。
- `inferref/alignment/store.py`: 事务性对齐会话存储与事件重放。
- `inferref/alignment/evidence.py`: 证据规范化与记录构建。
- `inferref/alignment/matrix.py`: 治具矩阵聚合与度量指标计算。
- `inferref/alignment/testcase_cache.py`: 基于内容寻址的测试用例提取与缓存。
- `inferref/alignment/runner.py`: 对齐执行编排与引擎标识解析。
- `inferref/alignment/narrowing.py`: 因果收敛与最小确证失败边界计算。
- `inferref/alignment/classification.py`: 确定性失效域分类规则与决策层。
- `inferref/alignment/report.py`: Markdown 报告与 JSON sidecar 生成器。
- `inferref/cli/align.py`: `inferref align` 8 个子命令的具体实现。
- `tests/core/test_alignment_store.py`: 存储、锁与事件重放测试。
- `tests/core/test_alignment_planner.py`: 规划与优先级测试。
- `tests/core/test_alignment_evidence.py`: 证据规范化与事件排序测试。
- `tests/core/test_alignment_runner.py`: 运行器、引擎修订与测试用例复用测试。
- `tests/core/test_alignment_diagnosis.py`: 因果收敛与分类规则测试。
- `tests/core/test_alignment_cli.py`: CLI 工作流与跨入口一致性测试。
- `tests/core/test_alignment_agent.py`: Agent 门面服务测试。
- `tests/agent/test_alignment_mcp.py`: MCP 协议与路径策略测试。
- `tests/core/test_alignment_e2e.py`: pipeline 故障隔离与修复测试。

---

## 4. 测试与验证结果

独立审查七轮完成后执行最终发布前门禁：

```powershell
python -m pytest -q <alignment test files>                # 62 passed
python -m pytest -q                                       # 814 passed, 9 skipped, 3 deselected
ruff check inferref tests scripts                         # All checks passed
python -m compileall -q inferref tests                    # passed
python scripts/check_agent_workflow_skill.py --repo .     # passed; 14 MCP tools
cmake --build cpp/build --config Release                  # passed
ctest --test-dir cpp/build -C Release --output-on-failure # 3/3 passed
git diff --check                                          # no errors
```

**总计**: 对齐编排器专项 62/62、全工程 814 项测试全部通过；9 项按环境预期跳过，3 项 slow 测试由默认门禁取消选择。

---

## 5. 遗留风险与注意事项 (Remaining Risks)

1. **外部引擎长任务与 TDR 隔离**:
   - 在 Windows 环境下运行某些第三方原生执行器时，若引擎驱动崩溃可能触发 TDR 重置；`AlignmentRunner` 当前通过执行器超时与进程树隔离守护，但无法恢复操作系统显卡驱动级别的 TDR 事件。
2. **多进程并发写锁**:
   - `store.lock()` 已支持跨进程基于锁文件的原子自旋锁与 stale 锁探测。但在极高频分布式写入场景，建议配置合适的自旋重试等待间隔。
3. **发布资格**:
   - 七轮独立验收已完成，版本元数据随发布提交统一更新为 `0.10.0`；远端分支与标签 CI 仍是创建 GitHub Release 前的最终门禁。
