"""Identify *which* dog, on top of the detector that already finds dogs.

The Pi's YOLO answers "there is a dog at (x1,y1,x2,y2)". This package answers
"that is Rex". It deliberately does not retrain the detector: see the module
docstrings in `embed.py` and `gallery.py` for why a two-stage design beats
folding per-dog classes into the detection head.

Everything here runs on the *desktop*, never the Pi. The board is already at
its power and inference limits (see CLAUDE.md), and the client receives the
JPEG plus boxes anyway -- so it can crop and identify locally with no protocol
change and no cost to the Pi.
"""
