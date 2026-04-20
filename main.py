import sys
from PySide6.QtWidgets import QApplication
from froth_app.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Add src to path if needed (though with poetry it should be handled)
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    main()

# Confidence interval # Done
# PC 1&2 # Done
# PCA model - feed the data of the first few seconds, the project later data on it # Done
# Air Flow configuration, reagent flows 
# Boiling froth - predict in advance 
