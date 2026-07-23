from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from .sources import BLOCKED_ARTIFACTS, PHASE_0_2_ARTIFACTS, Artifact


CHUNK_SIZE = 1024 * 1024


def download(artifact: Artifact, raw_dir: Path) -> Path:
    archive_dir = raw_dir / "_archives" / artifact.dataset
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / artifact.name
    if target.exists() and target.stat().st_size > 0:
        print(f"skip download: {artifact.name}")
        return target

    partial = target.with_suffix(f"{target.suffix}.part")
    partial.unlink(missing_ok=True)
    print(f"download: {artifact.name}")
    with urlopen(artifact.url) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        current = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            current += len(chunk)
            if total:
                print(f"  {current / total:6.1%}", end="\r")
    partial.replace(target)
    print(f"  saved {target.stat().st_size / 1024 / 1024:.1f} MB")
    return target


def extract(archive: Path, artifact: Artifact, raw_dir: Path) -> None:
    target_dir = raw_dir / artifact.extract_to
    marker = target_dir / f".{artifact.name}.extracted"
    if marker.exists():
        print(f"skip extract: {artifact.name}")
        return

    print(f"extract: {artifact.name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = target_dir / member.filename
            if target_dir.resolve() not in member_path.resolve().parents:
                raise ValueError(f"unsafe path: {member.filename}")
        zip_file.extractall(target_dir)
    marker.touch()


def import_artifacts(raw_dir: Path) -> None:
    for artifact in PHASE_0_2_ARTIFACTS:
        print(f"\n{artifact.dataset}: {artifact.note}")
        archive = download(artifact, raw_dir)
        extract(archive, artifact, raw_dir)
    if BLOCKED_ARTIFACTS:
        print("\ntracked but not downloaded")
        for artifact in BLOCKED_ARTIFACTS:
            print(f"{artifact.dataset}: {artifact.note}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    import_artifacts(args.raw_dir)


if __name__ == "__main__":
    main()
