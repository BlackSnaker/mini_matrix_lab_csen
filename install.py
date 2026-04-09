from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT / "requirements.txt"
OLLAMA_INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"
OLLAMA_WINDOWS_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
USER_BIN_DIR = Path.home() / ".local" / "bin"
LAUNCHER_SOURCE = ROOT / "run_ollama_brain_lab"
TRAIN_AGENT_SOURCE = ROOT / "train_agent_brain"
COMBAT_DOJO_SOURCE = ROOT / "train_combat_dojo"
TRAIN_COMBAT_SOURCE = ROOT / "train_agent_combat"
LAUNCHER_LINKS = (
    (LAUNCHER_SOURCE, ("run_ollama_brain_lab", "csen_brain_lab")),
    (TRAIN_AGENT_SOURCE, ("train_agent_brain", "csen_train_agent_brain")),
    (COMBAT_DOJO_SOURCE, ("train_combat_dojo", "csen_combat_dojo")),
    (TRAIN_COMBAT_SOURCE, ("train_agent_combat", "csen_train_agent_combat")),
)


def _step(message: str) -> None:
    print(f"\n=== {message} ===")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f">>> {printable}")
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def install_python_requirements(*, upgrade_pip: bool = True) -> None:
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"requirements file not found: {REQUIREMENTS_FILE}")
    _step("Installing Python dependencies")
    if upgrade_pip:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def _ollama_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    current = shutil.which("ollama")
    if current:
        candidates.append(Path(current))
    system = platform.system().lower()
    if system == "windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")
    else:
        candidates.extend([
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
            Path("/bin/ollama"),
        ])
    seen: set[str] = set()
    unique: list[Path] = []
    for item in candidates:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def find_ollama_binary() -> Path | None:
    for candidate in _ollama_candidate_paths():
        if candidate.exists():
            return candidate
    return None


def install_ollama(*, version: str | None = None, no_start: bool = False) -> Path:
    system = platform.system().lower()
    if system in {"linux", "darwin"}:
        return _install_ollama_unix(version=version, no_start=no_start)
    if system == "windows":
        return _install_ollama_windows()
    raise RuntimeError(f"Unsupported platform for Ollama auto-install: {platform.system()}")


def _install_ollama_unix(*, version: str | None, no_start: bool) -> Path:
    _step("Installing Ollama")
    with tempfile.TemporaryDirectory(prefix="csen_ollama_") as temp_dir:
        script_path = Path(temp_dir) / "install_ollama.sh"
        _download(OLLAMA_INSTALL_SCRIPT_URL, script_path)
        mode = script_path.stat().st_mode
        script_path.chmod(mode | stat.S_IXUSR)
        env = os.environ.copy()
        if version:
            env["OLLAMA_VERSION"] = str(version)
        if no_start:
            env["OLLAMA_NO_START"] = "1"
        _run(["sh", str(script_path)], env=env)
    ollama_path = find_ollama_binary()
    if ollama_path is None:
        raise RuntimeError("Ollama install completed, but the `ollama` binary was not found in PATH.")
    return ollama_path


def _install_ollama_windows() -> Path:
    _step("Installing Ollama")
    with tempfile.TemporaryDirectory(prefix="csen_ollama_") as temp_dir:
        installer_path = Path(temp_dir) / "OllamaSetup.exe"
        _download(OLLAMA_WINDOWS_INSTALLER_URL, installer_path)
        _run([str(installer_path)])
    ollama_path = find_ollama_binary()
    if ollama_path is None:
        raise RuntimeError(
            "Ollama installer finished, but `ollama.exe` was not found. "
            "If Windows updated PATH during install, open a new terminal and run `ollama -v`."
        )
    return ollama_path


def ensure_ollama_installed(*, version: str | None = None, no_start: bool = False) -> Path:
    existing = find_ollama_binary()
    if existing is not None:
        _step("Ollama already installed")
        print(existing)
        return existing
    return install_ollama(version=version, no_start=no_start)


def install_cli_launchers() -> list[Path]:
    for source, _names in LAUNCHER_LINKS:
        if not source.exists():
            raise FileNotFoundError(f"launcher source not found: {source}")
    _step("Installing CLI launchers")
    USER_BIN_DIR.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for source, names in LAUNCHER_LINKS:
        try:
            mode = source.stat().st_mode
            source.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
        for name in names:
            target = USER_BIN_DIR / name
            if target.exists() or target.is_symlink():
                try:
                    target.unlink()
                except Exception:
                    pass
            os.symlink(source, target)
            installed.append(target)
    for path in installed:
        print(f"Launcher ready: {path}")
    return installed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Mini-Matrix Lab dependencies and auto-install Ollama if it is missing."
    )
    parser.add_argument("--skip-python", action="store_true", help="Skip installing Python dependencies from requirements.txt.")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip the Ollama auto-install step.")
    parser.add_argument("--skip-launchers", action="store_true", help="Skip installing command launchers into ~/.local/bin.")
    parser.add_argument("--ollama-version", default=None, help="Optional Ollama version for the official installer script.")
    parser.add_argument("--no-ollama-start", action="store_true", help="Do not auto-start Ollama during installation on supported platforms.")
    parser.add_argument("--no-upgrade-pip", action="store_true", help="Do not upgrade pip before installing requirements.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))

    if not args.skip_python:
        install_python_requirements(upgrade_pip=not args.no_upgrade_pip)
    else:
        _step("Skipping Python dependencies")

    if not args.skip_ollama:
        ollama_path = ensure_ollama_installed(
            version=args.ollama_version,
            no_start=bool(args.no_ollama_start),
        )
        print(f"Ollama ready: {ollama_path}")
    else:
        _step("Skipping Ollama auto-install")

    if not args.skip_launchers:
        installed = install_cli_launchers()
        print(f"Commands ready: {', '.join(str(path.name) for path in installed)}")
    else:
        _step("Skipping CLI launcher install")

    _step("Installation complete")
    print("Project dependencies are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
