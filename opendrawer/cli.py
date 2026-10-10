"""OpenDrawer command line.

    python -m opendrawer page photo.jpg [--difficulty facile] [--yes]
    python -m opendrawer book dossier_photos/ [--yes]
    python -m opendrawer resume <projet> [--yes]
    python -m opendrawer doctor
    python -m opendrawer demo
    python -m opendrawer models list | download sdxl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from lib.config_model import OpenDrawerConfig
from lib.difficulty import get_preset
from lib.paths import PROJECTS_DIR
from opendrawer.runner import AutoApprover, BookRun, PageRun, RunSettings, StageError


class ConsoleApprover:
    """Asks in the terminal at each human gate."""

    def __init__(self, stream_in=sys.stdin) -> None:
        self.stream_in = stream_in

    def _ask(self, prompt: str) -> str:
        print(prompt, end="", flush=True)
        line = self.stream_in.readline()
        return line.strip()

    def choose_cartoon(self, project_id: str, cartoon_set: dict[str, Any]) -> Optional[str]:
        print(f"\n[{project_id}] Propositions de cartoon ({cartoon_set['backend']}) :")
        for c in cartoon_set["candidates"]:
            mark = "  ← recommandé" if c["id"] == cartoon_set.get("selected_id") else ""
            score = c.get("score", {}).get("total")
            print(f"  {c['id']}: {c['path']}  (score {score}){mark}")
        answer = self._ask("Numéro à garder (Entrée = recommandé, q = arrêter et décider plus tard) : ")
        if answer.lower() in ("q", "quit", "stop"):
            return None
        if not answer:
            return cartoon_set.get("selected_id")
        wanted = answer if answer.startswith("c") else f"c{answer}"
        ids = {c["id"] for c in cartoon_set["candidates"]}
        if wanted not in ids:
            print(f"  Choix inconnu {answer!r} : arrêt, relancez avec `resume`.")
            return None
        return wanted

    def approve(self, project_id: str, stage: str, summary: dict[str, Any]) -> bool:
        if "coloring_page" in summary:
            cp, qa = summary["coloring_page"], summary.get("qa_report", {})
            print(f"\n[{project_id}] Page prête : {cp['page_png']}")
            print(f"  SVG : {cp['page_svg']}")
            print(f"  Contrôle qualité : {qa.get('status')} — {qa.get('summary')}")
        elif "book_manifest" in summary:
            pages = summary["book_manifest"]["pages"]
            done = sum(1 for p in pages if p["status"] == "completed")
            print(f"\n[{project_id}] Étape {stage} : {done}/{len(pages)} pages validées")
            if summary["book_manifest"].get("contact_sheet"):
                print(f"  Sommaire : {summary['book_manifest']['contact_sheet']}")
        answer = self._ask("Valider ? [O/n] : ")
        return answer.lower() in ("", "o", "oui", "y", "yes")


def _settings_from_args(args: argparse.Namespace, cfg: OpenDrawerConfig) -> RunSettings:
    descriptions = {}
    for item in args.describe or []:
        sid, _, text = item.partition("=")
        if text:
            descriptions[sid.strip()] = text.strip()
    return RunSettings(
        difficulty=get_preset(args.difficulty or cfg.coloring.difficulty).name,
        style=args.style or cfg.generation.style,
        backend=args.backend or cfg.generation.backend,
        family=args.family or cfg.generation.family,
        identity=args.identity or cfg.generation.identity,
        candidates=args.candidates or cfg.generation.candidates,
        seed=args.seed if args.seed is not None else cfg.generation.seed,
        steps=args.steps,
        color_source=args.color_source or cfg.coloring.color_source,
        commercial_safe=bool(args.commercial_safe or cfg.licensing.commercial_safe),
        paper=cfg.page.paper,
        dpi=cfg.page.dpi,
        margin_mm=cfg.page.margin_mm,
        thumbnail_ratio=cfg.page.thumbnail_ratio,
        neural_lineart=args.neural_lineart,
        descriptions=descriptions,
    )


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", help="Identifiant du projet (défaut : nom du fichier/dossier)")
    p.add_argument("--difficulty", help="facile | moyen | difficile (ou easy/medium/hard)")
    p.add_argument("--style", help="Style de cartoon (styles/<nom>.yaml)")
    p.add_argument("--backend", choices=["auto", "diffusers", "comfyui", "opencv"])
    p.add_argument("--family", choices=["auto", "sdxl", "sd15"])
    p.add_argument("--identity", choices=["auto", "faceid", "plus_face", "none"])
    p.add_argument("--candidates", type=int, help="Nombre de propositions de cartoon")
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, help="Étapes de diffusion (défaut : celles du style)")
    p.add_argument("--color-source", choices=["photo", "cartoon"])
    p.add_argument("--neural-lineart", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--describe", action="append", metavar="s1=TEXTE",
                   help="Description anglaise d'un sujet pour le prompt (ex. s1='a girl with curly red hair')")
    p.add_argument("--commercial-safe", action="store_true", help="Uniquement des modèles à licence permissive")
    p.add_argument("--yes", "-y", action="store_true", help="Valider automatiquement (candidat recommandé)")


def _approver(args: argparse.Namespace):
    return AutoApprover() if getattr(args, "yes", False) else ConsoleApprover()


def _report(status: str, project_dir: Path) -> int:
    if status == "completed":
        print(f"\nTerminé : {project_dir / 'output'}")
        return 0
    print(f"\nEn attente de validation. Reprendre avec : python -m opendrawer resume {project_dir.name}")
    return 2


def cmd_page(args: argparse.Namespace) -> int:
    cfg = OpenDrawerConfig.load()
    settings = _settings_from_args(args, cfg)
    run = PageRun.create(args.photo, settings, _approver(args), project_id=args.name)
    print(f"Projet : {run.project_dir}")
    return _report(run.run(), run.project_dir)


def cmd_book(args: argparse.Namespace) -> int:
    cfg = OpenDrawerConfig.load()
    settings = _settings_from_args(args, cfg)
    run = BookRun.create(args.folder, settings, _approver(args), book_id=args.name)
    print(f"Livre : {run.project_dir}")
    status = run.run()
    if status == "completed":
        print(f"\nLivre terminé : {run.project_dir / 'output' / 'livre'}")
        return 0
    return _report(status, run.project_dir)


def cmd_resume(args: argparse.Namespace) -> int:
    marker_path = PROJECTS_DIR / args.project / "project.json"
    if not marker_path.exists():
        print(f"Projet introuvable : {args.project}", file=sys.stderr)
        return 1
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    approver = _approver(args)
    if marker.get("pipeline_type") == "coloring-book":
        run = BookRun(args.project, approver)
        status = run.run()
        return 0 if status == "completed" else _report(status, run.project_dir)
    run = PageRun(args.project, approver)
    return _report(run.run(), run.project_dir)


def cmd_doctor(args: argparse.Namespace) -> int:
    from tools.tool_registry import registry

    summary = registry.provider_menu_summary()
    hw = summary["hardware"]
    print("Matériel :", hw.get("device"), hw.get("gpu_name") or "", f"{hw.get('vram_mb', 0)} Mo VRAM" if hw.get("vram_mb") else "")
    for note in hw.get("notes", []):
        print("  -", note)
    print("\nOutils :")
    for cap in summary["capabilities"]:
        print(f"  {cap['capability']}: {cap['configured']}/{cap['total']} disponibles  {cap['available_tools']}")
    if summary["missing_models"]:
        print("\nModèles manquants :", ", ".join(summary["missing_models"]))
        print("  → python -m opendrawer models download sdxl   (ou sd15, analysis, faceid)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from lib.model_registry import LicenseError, model_registry

    if args.commercial_safe:
        os.environ["OPENDRAWER_COMMERCIAL_SAFE"] = "1"
    if args.action == "list":
        print("Presets :", ", ".join(f"{k} ({len(v)})" for k, v in model_registry.presets().items()))
        for row in model_registry.status():
            flag = "✔" if row["downloaded"] else "·"
            safe = "commercial OK" if row["commercial_safe"] else "non commercial / non vérifié"
            print(f" {flag} {row['key']:<28} {row['license']:<40} {safe:<28} ~{row['approx_size_mb']} Mo")
        return 0
    keys = model_registry.expand(args.keys or ["sdxl"])
    status = 0
    for key in keys:
        try:
            model_registry.check_allowed(key)
            if model_registry.is_downloaded(key):
                print(f"✔ {key} déjà présent")
                continue
            print(f"↓ {key} …", flush=True)
            path = model_registry.download(key)
            print(f"✔ {key} → {path}")
        except LicenseError as exc:
            print(f"✗ {key}: {exc}")
            status = 1
        except Exception as exc:
            print(f"✗ {key}: {exc}")
            status = 1
    return status


def cmd_demo(args: argparse.Namespace) -> int:
    from opendrawer.demo import make_demo_photo

    photo = make_demo_photo(PROJECTS_DIR / "_demo" / "demo_photo.jpg")
    settings = RunSettings(difficulty=get_preset(args.difficulty).name, backend="opencv", candidates=2, seed=0)
    run = PageRun.create(photo, settings, AutoApprover(), title="Démo OpenDrawer")
    print(f"Projet de démonstration : {run.project_dir}")
    return _report(run.run(), run.project_dir)


def cmd_selftest(args: argparse.Namespace) -> int:
    from opendrawer.selftest import run_selftest

    print("Autodiagnostic en cours (la génération peut prendre plusieurs minutes)…", flush=True)
    report = run_selftest(family=args.family, generate=not args.no_generate, steps=args.steps)
    for c in report["checks"]:
        mark = "✔" if c["status"] == "ok" else "✗"
        detail = c["detail"].get("error") if isinstance(c["detail"], dict) and "error" in c["detail"] else ""
        print(f" {mark} {c['check']:<45} {c['seconds']:>7.1f}s  {detail}")
    print(f"\nRapport : {report['report_path']}")
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opendrawer", description="Photo → page de coloriage pour enfant (100 % local)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("page", help="Une photo → une page de coloriage")
    p.add_argument("photo")
    _add_run_options(p)
    p.set_defaults(func=cmd_page)

    b = sub.add_parser("book", help="Un dossier de photos → un livre de coloriage")
    b.add_argument("folder")
    _add_run_options(b)
    b.set_defaults(func=cmd_book)

    r = sub.add_parser("resume", help="Reprendre un projet (après une validation en attente)")
    r.add_argument("project")
    r.add_argument("--yes", "-y", action="store_true")
    r.set_defaults(func=cmd_resume)

    d = sub.add_parser("doctor", help="Matériel, outils et modèles disponibles")
    d.set_defaults(func=cmd_doctor)

    demo = sub.add_parser("demo", help="Page de démonstration sans GPU ni modèle (backend OpenCV)")
    demo.add_argument("--difficulty", default="medium")
    demo.set_defaults(func=cmd_demo)

    st = sub.add_parser("selftest", help="Vérifier GPU, modèles et une vraie génération (rapport JSON)")
    st.add_argument("--family", choices=["auto", "sdxl", "sd15"], default="auto")
    st.add_argument("--steps", type=int, default=12)
    st.add_argument("--no-generate", action="store_true", help="Sans génération (vérifications rapides seulement)")
    st.set_defaults(func=cmd_selftest)

    m = sub.add_parser("models", help="Lister / télécharger les modèles locaux")
    m.add_argument("action", choices=["list", "download"])
    m.add_argument("keys", nargs="*", help="Clés de models.yaml ou presets (sdxl, sd15, faceid, analysis)")
    m.add_argument("--commercial-safe", action="store_true")
    m.set_defaults(func=cmd_models)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StageError as exc:
        print(f"\nÉchec : {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
