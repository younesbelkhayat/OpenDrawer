# Directeur — étape `collect` (livre)

## Rôle

Lister les photos du dossier (jpg, jpeg, png, webp, bmp, tif ; ordre
alphabétique, donc préfixez `01_`, `02_`… pour imposer l'ordre) et créer le
`book_manifest` : une entrée par page, chacune liée à un futur projet
`coloring-page` nommé `<livre>-p01`, `<livre>-p02`…

## À faire

- Initialiser le projet livre (`pipeline_type="coloring-book"`) avec les
  réglages communs (difficulté, style, commercial_safe) dans `settings`.
- Signaler les fichiers ignorés et les photos manifestement inutilisables
  (`subject_detect` rapide si besoin).
- Confirmer l'ordre et la difficulté avec l'utilisateur si ce n'est pas clair.

Pas de gate : enchaîner sur `pages`.
