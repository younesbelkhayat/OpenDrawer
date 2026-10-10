# /backlot — ouvrir le tableau

Ouvrir le tableau Backlot du projet demandé :

```bash
python -m backlot open $ARGUMENTS
```

- Sans argument : vue de tous les projets.
- Idempotent : démarre le serveur si besoin puis ouvre le navigateur.
- En cas d'échec, le signaler et continuer : le tableau observe, il ne bloque jamais.
- Le tableau lit `projects/<id>/` sur le disque ; tenir checkpoints et artefacts à jour selon `skills/meta/checkpoint-protocol.md`.
