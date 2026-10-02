# Communication and Style

- Write in a direct, concise, information-dense, and technical manner.
- Keep language simple, clear, minimal, and easy to understand with logical progression and zero redundancy.
- Do not use emojis, em-dashes, fluff, filler, or marketing language.
- Avoid fancy bullet formatting, especially bold prefixes or titles at the beginning of bullet points. Use plain, simple bullets.
- Use inclusive, modern terminology at all times.
- Focus on understanding and applying underlying core concepts rather than surface jargon.
- Aim for the shortest possible output that completely satisfies all requirements.

# Repository Management

- Keep this file lean. Only add new guidelines or project instructions with explicit user permission.
- Keep .gitignore minimal. Only add entries when required and do not bloat it.
- Use conventional commits for all git commit messages.
- Use a project-local temp/ directory as an ephemeral scratchpad and delete it after task completion. Never use system /tmp/.
- Route all downloads, cached weights, tokenizers, and dataset artifacts into a project-local cache/ directory. Never pollute the user's home directory (~/.cache).

# Codebase Architecture

- Maintain a modular structure with clear separation of concerns.
- Write readable, maintainable, and reusable code with clean interfaces.
- Keep code well commented, clearly explaining rationale and non-obvious logic.
- Use Python's standard logging module exclusively for all workflow, diagnostic, and progress reporting. Do not use raw print() statements. Route logs to both the console and a project-local logs/ directory.
- Ensure all pipeline execution stages are strictly idempotent and resumable, skipping previously completed and validated artifacts unless an explicit --overwrite flag is supplied.

# Performance and Portability

- Write platform-agnostic code compatible across macOS, Linux, and Windows.
- Dynamically detect and leverage available hardware accelerators (CUDA, MPS, CPU).
- Maximize performance by parallelizing workloads across available compute resources.
- Standardize numerical precision on torch.float32 across all model representations, covariance computations, and transformations unless double precision is explicitly required.

# Environment and Tooling

- Always use uv for all environment, package, and script execution workflows. Never invoke raw python or python3 directly; always use uv run.
- Always use the latest version of Python and the latest versions of all packages and tools. Downgrade only when strictly required for package compatibility.
- Maintain minimal dependencies.
- Do not install packages or system tools without explicit prior user approval.
- Use ruff for linting and code formatting.
- Use ty for static type checking.
- Provide explicit static type annotations on all function signatures and module interfaces.

# Reproducibility

- Ensure deterministic execution by setting random seeds across torch, numpy, and random when stochastic operations occur.
- Ensure all artifacts, metrics, and results are programmatically reproducible from execution scripts.