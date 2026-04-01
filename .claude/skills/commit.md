---
name: commit
description: Use this skill when the user asks to commit, create a git commit, or push code changes. Enforces project commit conventions.
---

# Git Commit Conventions

When creating git commits for this project, follow these rules:

## Commit Message Format

1. **Language**: Commit messages MUST be written in **English only**
2. **Style**: Use 如下 format:
   ```
   [<scope>] <描述>
   ```

   | scope | 适用场景 |
   |-------|---------|
   | `triton` | Triton Ascend 侧改动 |
   | `ascendc` | AscendC 侧改动 |
   | `benchmark` | Benchmark case / 评测逻辑 |
   | `router` | op-router / 路由逻辑 |
   | `infra` | CI、脚本、构建 |
   | `docs` | 文档 |

   示例：
   - `[triton] 新增 layernorm 算子生成支持`
   - `[ascendc] dsl-lowering tiling pass 优化`
   - `[benchmark] NPUKernelBench level2 新增 10 case`
   - `[router] op-router 增加 CUDA→Ascend 路由分支`


## Authorship

- **Do NOT** include `Co-Authored-By` trailers referencing Claude or any AI assistant
- The commit author should be the user's own git config identity only

## Example

```
`[triton] 新增 layernorm 算子生成支持`
```

## Workflow

1. Run `git status` and `git diff` to review changes
2. Stage relevant files by name (avoid `git add -A`)
3. Compose commit message following the rules above
4. Use HEREDOC format for multi-line messages
5. Push only when the user explicitly requests it

## 注意点
1. 在提交信息中**绝对不要**以任何形式体现 Claude