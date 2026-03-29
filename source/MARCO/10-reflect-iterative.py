#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from run_iterative_wrapper import run


if __name__ == "__main__":
    raise SystemExit(run("marco", "gemma-2-2b-reflection-final"))
