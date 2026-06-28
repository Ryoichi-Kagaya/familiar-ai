# dev/ — Developer Skills for familiar-ai

Claude Code skills for contributors and power users of familiar-ai.

## Available Skills

| Skill | Description |
|-------|-------------|
| [`familiar-add-tool`](./familiar-add-tool/SKILL.md) | Scaffold a new sensor/actuator tool and register it in all required places |
| [`familiar-check-env`](./familiar-check-env/SKILL.md) | Validate `.env` configuration and surface common misconfigurations |
| [`familiar-debug-loop`](./familiar-debug-loop/SKILL.md) | Diagnose a stuck or misbehaving ReAct agent loop |

## Installation

The skills work with both Claude Code and Kimi Code CLI.

### Claude Code

Copy or symlink the skills into your Claude Code skills directory:

```bash
# Copy all familiar-ai dev skills
cp -r dev/familiar-* ~/.claude/skills/

# Or symlink (changes to the repo are reflected immediately)
for d in dev/familiar-*/; do
  ln -sf "$(pwd)/$d" ~/.claude/skills/
done
```

Then invoke from Claude Code with e.g. `/familiar-add-tool`.

### Kimi Code CLI

Symlink the same directories into Kimi's user skills directory:

```bash
mkdir -p ~/.kimi-code/skills
for d in dev/familiar-*/; do
  ln -sf "$(pwd)/$d" ~/.kimi-code/skills/
done
```

Then invoke from Kimi with e.g. `/skill:familiar-add-tool`.
