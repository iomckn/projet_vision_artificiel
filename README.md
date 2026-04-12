# Projet

## Installation

1. Creez et activez un environnement virtuel (recommande).
2. Installez les dependances :

```bash
pip install -r requirements.txt
```

## Lancer le projet

```bash
python app.py
```

L'application Flask tourne alors en mode debug sur http://127.0.0.1:5000/.

## Tutoriel macOS

```bash
cd projet_vision_artificiel
python3 -m env env             # creer le env dans le dossier du projet
source env/bin/activate         # activer le env
pip install -r requirements.txt  # installer les deps
python3 app.py                   # lancer le serveur
```

## Tutoriel Windows

Sous PowerShell :

```powershell
cd airec_frontend
python -m env env              # creer le env dans le dossier du projet
.\env\Scripts\Activate.ps1     # activer le env
python -m pip install -r requirements.txt
python app.py
```

Sous Invite de commandes (cmd) :

```cmd
cd airec_frontend
python -m env env
env\Scripts\activate.bat
python -m pip install -r requirements.txt
python app.py
```