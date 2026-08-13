"""
pytest 共享配置与夹具
====================
- 将项目根目录加入 sys.path，使 `import pcb_vision_agent` 可用
- 提供测试场景图片夹具（隔离在临时目录）
"""
import sys
from pathlib import Path

# 项目根目录 = d:\vscode python（pcb_vision_agent 的上级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from pcb_vision_agent.core.agent import generate_test_scenarios  # noqa: E402
from pcb_vision_agent.utils.config import (  # noqa: E402
    OUTPUT_DETECTED,
    OUTPUT_LOGS,
    OUTPUT_REPORTS,
)
from pcb_vision_agent.utils.file_utils import ensure_dir  # noqa: E402


@pytest.fixture(scope="session")
def test_scenarios(tmp_path_factory):
    """生成三个测试场景图片，返回 {名称: 路径}（隔离在临时目录）"""
    out = tmp_path_factory.mktemp("scenarios")
    return generate_test_scenarios(str(out))


@pytest.fixture(scope="session", autouse=True)
def ensure_output_dirs():
    """确保项目输出目录存在（批量/报表测试会写入）"""
    for d in (OUTPUT_DETECTED, OUTPUT_LOGS, OUTPUT_REPORTS):
        ensure_dir(d)
    return True
