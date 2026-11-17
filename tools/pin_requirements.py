"""Write requirements-solution.txt with the exact versions installed on this machine.

    python tools/pin_requirements.py
"""
import os
import platform
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = ["# Exact versions the submission was generated with (Python {}).".format(platform.python_version()),
         "# CPU is enough: the networks are tiny and inference runs on CPU."]
for pkg in ("numpy", "pandas", "torch"):
    lines.append("{}=={}".format(pkg, metadata.version(pkg)))
open(os.path.join(ROOT, "requirements-solution.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
