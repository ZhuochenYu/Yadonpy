# YadonPy Maintenance Principles

## Purpose

This file defines the default principles for understanding, modifying, and refactoring YadonPy.
Future maintenance should follow these rules unless the user explicitly asks for a deliberate exception.

## Requirement-First Rules

1. Start from the original user need.
Do not assume the user has already fully clarified the goal, constraints, success criteria, or implementation path.

2. Clarify only when ambiguity is truly critical.
Only stop to ask first when different interpretations would lead to clearly different solutions or a materially higher error cost.
Otherwise, continue with the most reasonable interpretation and explicitly state the assumption.

3. Design only for the stated goal.
When proposing a modification or refactor, default to the goal the user explicitly asked for.
Do not silently expand the business goal, introduce substitute workflows, or redesign adjacent business behavior without need.

4. Prefer the minimum complete correct solution.
Prefer the smallest solution that fully closes the requested problem.
Do not prefer patch-on-patch compatibility hacks over a structurally correct solution.
If the shortest path conflicts with avoiding structural mistakes, choose the minimum correct non-fragile solution.

5. Avoid unrelated fallback branches.
Do not add downgrade paths, extra compatibility branches, or defensive side routes unrelated to the current requirement.
To keep logic closed, it is acceptable to add necessary input constraints, state checks, invariant checks, and boundary protection.

## Mandatory Chain Check Before Output

Before proposing or implementing a change, check the full path of the change in this order:

1. Input
What enters the function, workflow, or API, and what assumptions are being made about it.

2. Processing flow
How the input moves through the actual code path, including transformations, cache hits, restart/resume behavior, and external-tool calls.

3. State changes
What files, metadata, object properties, cached artifacts, and workflow state are created, reused, mutated, or invalidated.

4. Output
What the user or downstream code receives: return values, exported files, generated metadata, analysis results, and side effects.

5. Upstream and downstream impact
What existing callers, cached data, restart logic, tests, examples, and external tools may be affected.

Any part that cannot be verified must be marked explicitly as an assumption or unverified prerequisite.
Never present inference as confirmed fact.

## YadonPy-Specific Architecture Constraints

1. Preserve the script-first workflow style.
YadonPy is designed so users can write direct scientific scripts instead of being forced into a large framework.
Changes should keep the public flow understandable from examples and top-level API calls.

2. Protect restart/resume semantics.
`restart`, `RunOptions`, `WorkDir`, and workflow resume state are part of the product behavior, not incidental utilities.
Do not introduce destructive cleanup, silent re-run behavior, or hidden skip logic by default.

3. Respect the MolDB-first design.
MolDB is for expensive reusable artifacts such as geometry, charge variants, and optional bonded patches.
Do not reintroduce persistent topology-cache thinking as a default storage model.
Topology and GROMACS export artifacts should remain generated on demand.

4. Treat exported files and metadata as part of the contract.
In this project, `.gro`, `.top`, `.itp`, `.ndx`, JSON manifests, workdir metadata, and resume state are all behaviorally important.
When changing logic, verify whether file layout, names, metadata schema, or restart reuse rules are affected.

5. Prefer explicitness over hidden magic.
This repository often chooses explicit staged folders, explicit manifests, explicit route specs, and explicit workflow records.
New changes should fit that style unless there is a strong reason not to.

6. Validate scientific workflow closure, not only local code correctness.
A change is not complete just because a function passes locally.
It should still make sense across molecule construction, charge assignment, force-field assignment, export, GROMACS workflow execution, restart/reuse, and analysis.

## Maintenance Checklist

Before finalizing a change, quickly verify:

- the user goal is still the only target being solved;
- assumptions are written down where verification is missing;
- restart/resume behavior is still coherent;
- MolDB/read-write behavior is still coherent;
- generated artifacts and metadata remain internally consistent;
- tests or examples covering the changed path are checked when possible.

## Update rules

1. 确认当前目录下有 `history_version` 文件夹。每次更新后自动增加一个小版本号，并把旧版本移动到 `history_version` 中。
2. 保证当前目录下只保留当前版本文件夹，以及一个以 `.tar` 结尾的当前版本压缩包。
3. 每次代码更新完成后，把当前版本自动同步上传到云端 GitHub 仓库。默认自动完成提交、推送，并把更新合并到远端主分支。
4. 每次更新前后都检查是否存在 `__pycache__`、`.pytest_cache`、`.yadonpy_cache`、`yadonpy.egg-info` 等生成物；如果存在，不要保留在当前发布目录里，只保留有用的源码、文档和发布产物。
