"""Importing token_lengths has no side effects and reuses the shared doc_text."""
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
for sub_dir in (CODE, CODE / "pylib"):
    sys.path.insert(0, str(sub_dir))
import local_dataset_io  # noqa: E402

OUTPUT_FILE = CODE / "output" / "harness" / "token_lengths.json"


def _file_state(path: Path):
    return (path.exists(), path.stat().st_mtime_ns if path.exists() else None)


def test_import_does_not_write_output_file():
    state_before = _file_state(OUTPUT_FILE)
    import token_lengths  # noqa: F401
    assert _file_state(OUTPUT_FILE) == state_before


def test_doc_text_is_shared_function():
    import token_lengths
    assert token_lengths.doc_text is local_dataset_io.doc_text
    assert callable(token_lengths.main)


def test_import_loads_no_transformers_in_fresh_interpreter():
    # Fresh interpreter: other tests may already have imported transformers in this process.
    probe = (
        "import sys; sys.path.insert(0, 'pylib'); import token_lengths; "
        "sys.exit(1 if 'transformers' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=CODE, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
