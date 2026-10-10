# Prompts personnalisés Codex

Codex ne lit que **`~/.codex/prompts/`** (dossier personnel), pas le
`.codex/prompts/` du projet. Ces fichiers font référence ; pour les utiliser :

```bash
mkdir -p ~/.codex/prompts
ln -s "$PWD/.codex/prompts/backlot.md" ~/.codex/prompts/backlot.md
ln -s "$PWD/.codex/prompts/coloriage.md" ~/.codex/prompts/coloriage.md
```
