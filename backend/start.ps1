& "$PSScriptRoot\venv\Scripts\Activate.ps1"
$env:PYTHONUTF8 = "1"
python -m uvicorn server.server:app --host 0.0.0.0 --port 8000
