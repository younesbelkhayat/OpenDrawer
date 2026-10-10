# Relecture — meta-skill

## Quand l'utiliser

Après chaque étape, **avant** d'écrire le checkpoint final ou de présenter
une gate. Vous êtes multimodal : **ouvrez les images** produites (outil de
lecture de fichiers) au lieu de vous fier aux scores. Maximum
`orchestration.max_review_rounds` tours (2) par étape, puis présentez l'état
honnêtement.

## Méthode

1. Relire `review_focus` de l'étape dans le manifeste.
2. Regarder les fichiers : pour une petite image, directement ; pour une
   grande, une version réduite ou un recadrage (visage, mains).
3. Lister les constats : `critical` (bloque : membre en trop, zone ouverte,
   visage méconnaissable, fond non blanc), `suggestion`, `nitpick`.
4. Corriger ce qui peut l'être dans le périmètre approuvé (relancer un outil
   avec d'autres paramètres), puis re-relire.
5. Joindre le résultat au checkpoint (`review`) au format
   `schemas/artifacts/review.schema.json`.

## Grille par étape

| Étape | Vérifier | Correctifs typiques |
|---|---|---|
| intake | tous les sujets détectés, aucun faux sujet ; détourage complet (cheveux, oreilles, mains, queue) ; palette réaliste | `bg_remove` avec un autre modèle (`isnet-general-use` ↔ `birefnet-portrait`) ; `subject_detect` seuil plus bas/haut |
| cartoon | ressemblance (coiffure, lunettes, vêtements), style livre jeunesse (contours nets, aplats), aucun défaut anatomique, fond blanc | nouvelle graine, `strength` ± 0,05, `ip_adapter_scale` ↑ pour la ressemblance, `descriptions` plus précises |
| line_art | silhouette, visage, cheveux, vêtements présents ; pas de texture transformée en traits | `neural` on/off, autre candidat cartoon |
| regions | nombre de zones adapté à l'âge ; aucune zone minuscule ; pupilles/narines noircies ; couleurs du modèle fidèles à la photo | `max_regions` / `min_region_frac`, `color_source`, autre difficulté |
| page | `qa_report` sans échec ; traits nets ; vignette lisible et séparée ; SVG valides | relancer `page_layout` (`thumbnail_ratio`, `margin_mm`) |

`coloring_qa` est déterministe et sert de premier filtre : un `fail` est
toujours `critical`. Un `warn` doit être expliqué à l'utilisateur.

## Ce que la relecture n'autorise pas

Changer de backend, de famille de modèle, de mode de ressemblance ou passer
en `--commercial-safe` sans accord : ce sont des décisions utilisateur
(voir le contrat de communication des décisions dans `AGENT_GUIDE.md`).
