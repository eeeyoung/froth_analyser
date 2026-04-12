# froth_analyser
A modular Python-based application for real-time monitoring and metallurgical analysis of flotation froth via live camera feeds or recorded video.


## Project Structure
froth-analysis-app/
├── pyproject.toml          # Project metadata and dependencies
├── poetry.lock             # Pin-point version control (don't touch manually!)
├── .gitignore              # Updated for Poetry
├── README.md
├── main.py                 # Application entry point
├── src/                    
│   └── froth_app/          # Main package
│       ├── __init__.py
│       ├── ui/             
│       ├── engine/         
│       ├── core/           
│       └── database/       
├── tests/                  # For future unit tests
└── assets/