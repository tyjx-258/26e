# AGENTS.md

## Git 工作流

- 改动文件后不要自动提交；等用户确认某个问题**真正解决**后，再提交并推送到远端 `origin`。
- 提交前先跑静态自检（`pyflakes` + `py_compile`），确认没有未定义名字、语法等问题。
- 提交信息使用 Conventional Commits：`type(scope): 摘要`，摘要用中文说明改了什么、为什么改。
- 一次提交只包含一个逻辑变更。
