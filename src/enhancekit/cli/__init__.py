"""Command line interface for enhancekit."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from ..api import describe_models, list_models, load_model
from ..utils import load_image

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance images using pretrained models")
    parser.add_argument("model", nargs="?", help="Model name to use")
    parser.add_argument("inputs", nargs="*", help="Image paths to process")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--output", type=str, help="Optional output folder")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--no-freeze", action="store_true", help="Do not freeze model parameters")
    return parser


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for name, meta in describe_models().items():
            logger.info("%s - %s (%s)", name, meta["architecture"], meta["task"])
        return

    if not args.model:
        parser.error("Model name is required unless --list is provided")

    model = load_model(args.model, device=args.device, freeze=not args.no_freeze)

    if not args.inputs:
        parser.error("At least one input image path is required")

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.inputs:
        image = load_image(img_path)
        enhanced = model.enhance_image(image)
        if output_dir:
            dest = output_dir / Path(img_path).name
            enhanced.save(dest)
            logger.info("Saved enhanced image to %s", dest)


if __name__ == "__main__":
    main()
