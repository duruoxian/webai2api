"""pytest 公共装置。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 测试套件固定为统一版本形态（v0.1.36 起不再区分 pro/lite）。
os.environ["WEBAI2API_EDITION"] = "pro"

FIXTURES = Path(__file__).resolve().parent / "fixtures"
