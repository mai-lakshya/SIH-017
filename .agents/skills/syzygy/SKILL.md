---
name: syzygy
description: Master multi-agent engineering meta-framework & Spec-Driven Development (SDD) engine. Use for dynamic multi-agent project synthesis, automated 8-file SDD generation (.spec/PRD.md, TechSpec.md, Architecture.md, AppFlow.md, Design.md, Rules.md, Schema.md, Tracker.md), academic research extraction (43M+ papers via arXiv), SIH winning presentation deck generation (.pptx), OpenDesign DESIGN.md contracts, and ASD-STE100 anti-AI-slop auditing.
---

# SYZYGY: Master Multi-Agent Engineering Meta-Framework

SYZYGY is an autonomous multi-agent engineering engine that enforces the **8-File Spec-Driven Development (SDD)** standard, pre-project specifications, zero AI slop, and champion tooling across 22 engineering tasks.

## ⚡ The 8-File SDD Standard

Before writing application code, ensure or generate the 8 foundational specification files in `.spec/`:
1. `PRD.md`: Goals, personas, functional scope, success metrics
2. `TechSpec.md`: Runtime stack, dependencies, APIs, hardware limits
3. `Architecture.md`: Mermaid topology, multi-agent mesh, IPC protocols
4. `AppFlow.md`: User journeys, state machine, handoffs
5. `Design.md`: OpenDesign DESIGN.md contract (Tokens, Typography, Motion)
6. `Rules.md`: ASD-STE100 anti-slop rules and code directives
7. `Schema.md`: Data models, JSON-RPC, OpenViking (`viking://`) schemas
8. `Tracker.md`: Atomic milestones and verification checklist

## CLI Commands

Execute via the Python environment in the workspace:
```bash
# Synthesize new project with all agents per need
python .agents/skills/syzygy/scripts/syzygy.py new "<Prompt>"

# Academic paper extraction (arXiv API)
python .agents/skills/syzygy/scripts/syzygy.py research "<Topic>"

# Generate native vector PowerPoint presentation (.pptx)
python .agents/skills/syzygy/scripts/syzygy.py deck "<Title>" --out presentation.pptx

# Audit text and specs against ASD-STE100 anti-slop standard
python .agents/skills/syzygy/scripts/syzygy.py validate "<Text>"

# Run pre-flight penetration test
python .agents/skills/syzygy/scripts/syzygy.py audit --target http://localhost:3000
```
