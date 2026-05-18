"""Application entrypoint."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from point_labeler.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("point_labeler").setLevel(logging.DEBUG)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
