"""FoleyCrafter V2A 引擎命令行入口。

允许通过 `python -m engines.foleycrafter_a3b2c1` 启动。
（目录名用下划线，engine_id 字符串保留连字符作为标识符）
"""

from .server import main

if __name__ == "__main__":
    main()
