"""Application entry point."""
from pathlib import Path
from kedro.framework.project import configure_project


def main():
    configure_project(Path(__file__).parent.name)

    from kedro.framework.cli import main as kedro_main
    kedro_main()


if __name__ == "__main__":
    main()
