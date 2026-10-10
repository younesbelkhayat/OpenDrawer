# Directeur — étape `intake` (photo)

## Rôle

Comprendre la photo : qui est dessus (personnes, animaux, visages), isoler
les sujets sur fond blanc, relever les **vraies couleurs** (peau, cheveux,
vêtements, pelage). Produit `photo_analysis`.

## Outils, dans l'ordre

1. `subject_detect` `{image_path}` → boîtes des sujets et des visages.
2. `bg_remove` `{input_path, mask_path: assets/intake/foreground_mask.png,
   cutout_path: assets/intake/cutout.png, subject_boxes, subject_kinds}`.
   `model: auto` choisit `birefnet-portrait` (que des personnes) ou
   `isnet-general-use` (animaux ou mélange) ; repli GrabCut sans rembg.
3. `subject_detect` à nouveau avec `foreground_mask_path` et
   `mask_dir: assets/intake` → un masque par sujet (utilisé pour cartoonner
   chaque sujet séparément quand il y en a plusieurs).
4. `palette_extract` `{image_path: cutout, mask_path}` → palette.

Assemblez `photo_analysis` (voir `schemas/artifacts/photo_analysis.schema.json`)
avec `warnings` et `tools_used` des outils.

## Votre valeur ajoutée

- Regardez la photo et le détourage. Écrivez pour chaque sujet une
  `description` anglaise précise (« a smiling little girl with curly red
  hair, round glasses and a striped t-shirt ») : elle améliore beaucoup le
  prompt à l'étape cartoon.
- Signalez les photos difficiles : sujet coupé, très flou, contre-jour,
  foule (> 6 sujets), visage de profil. Proposez un recadrage plutôt que de
  continuer à l'aveugle.

## Critères de sortie

Au moins un sujet ; détourage sans morceau de décor ni membre coupé ;
palette cohérente avec la photo. Pas de gate humaine : enchaîner sur
`cartoon` après relecture.
