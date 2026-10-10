# Architecture d'OpenDrawer

## Chaîne de traitement d'une page

```
photo.jpg
  │ intake      subject_detect (OWLv2 + YuNet/Haar) → bg_remove (BiRefNet/IS-Net, repli GrabCut)
  │             → masques par sujet → palette_extract (k-means LAB)        ⇒ photo_analysis
  │ cartoon     cartoonize → diffusers | comfyui | opencv
  │             img2img + ControlNet Canny + IP-Adapter, 1 génération par sujet,
  │             recomposition alignée sur la photo, score objectif         ⇒ cartoon_set   [gate]
  │ line_art    traits d'encre + frontières d'aplats + silhouette (+ lineart neuronal)
  │             à 2400 px sur le grand côté                                ⇒ line_art
  │ regions     partition → union-find (fusions, remplissages noirs, plafond de zones)
  │             → lissage → traits uniformes + détails → couleurs photo    ⇒ region_map
  │ page        vectorize_svg → page_layout (A4 300 dpi) → coloring_qa     ⇒ coloring_page, qa_report [gate]
  ▼
output/page.png · page.svg · coloriage.png/svg · modele.png/svg
```

## Pourquoi un cartoon *aligné* sur la photo

ControlNet + img2img conservent le cadrage : chaque zone du dessin se
superpose à la même partie de la photo. On peut donc reprendre les vraies
couleurs (palette de la photo) et garder la silhouette exacte du détourage.

## Algorithme des zones (region_segment)

1. Fermeture morphologique des traits bruts, puis composantes connexes des
   pixels libres → zones candidates.
2. Chaque pixel de trait est attribué à la zone libre la plus proche
   (transformée de distance) : on obtient une partition sans trou, d'où les
   adjacences et longueurs de frontière.
3. Union-find :
   - zones trop fines pour un crayon (rayon inscrit < épaisseur de trait) ou
     trop petites → fusion avec la voisine de plus longue frontière ;
   - petites zones sombres, compactes (circularité ≥ 0,45) et entourées par
     une seule zone (≥ 90 %) → **remplies de noir** (pupilles, narines) ;
   - au-delà du maximum de la difficulté → fusion des voisines les plus
     proches en couleur, les petites d'abord.
4. Lissage majoritaire de la carte des zones (filtre médian).
5. Traits finaux = frontières de la partition, épaisseur uniforme selon la
   difficulté (silhouette plus épaisse) + traits ouverts dessinés (sourire)
   + petits points d'encre.
6. Couleur de chaque zone = moyenne du cartoon, ramenée à la couleur de la
   palette photo la plus proche (ΔE76 ≤ 22) ; nom de crayon en français.

## Matériel

`lib/hardware.py` détecte CUDA/MPS et la VRAM, choisit une stratégie
(`full_gpu`, `model_offload`, `sequential_offload`) et une échelle de repli
en cas de manque de mémoire. Sur 6 Go : SDXL en déchargement séquentiel,
SD1.5 entièrement sur le GPU.

## Modèles et licences

`models.yaml` liste chaque jeu de poids (dépôt public, motifs de fichiers,
licence, `commercial_safe`). Les outils ne téléchargent jamais rien :
`model_registry.path()` lève `ModelNotDownloaded` avec la commande à lancer,
et `LicenseError` en mode commercial-safe.

## Checkpoints et gates

`lib/checkpoint.py` valide chaque checkpoint (schéma, artefacts du
`produces`, prédécesseurs terminés, gate humaine), archive l'ancien dans
`history/` et fusionne le `decision_log`.

## Backlot

Serveur FastAPI local (`python -m backlot open`) qui observe `projects/`
(watchfiles + SSE) et affiche, sans jamais écrire, les étapes, les images de
chaque étape, le rapport QA, les décisions et l'activité des outils
(`events.jsonl`, écrit par l'instrumentation de `BaseTool`).
