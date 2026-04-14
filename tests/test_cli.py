from pathlib import Path
import py_compile


def test_cli_module_compiles():
    py_compile.compile(
        str(Path("src/cli.py")),
        doraise=True,
    )
