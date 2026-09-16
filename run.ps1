<#
    run.ps1 — start the RAG API + test console.

        .\run.ps1                 # http://127.0.0.1:8099
        .\run.ps1 -Port 9000
        .\run.ps1 -Reload         # auto-restart on code changes
        .\run.ps1 -NoBrowser

    Checks MongoDB and the LLM endpoint first so a failure is obvious here
    rather than halfway through a query.
#>
param(
    [int]    $Port = 8099,
    [string] $BindHost = "127.0.0.1",
    [switch] $Reload,
    [switch] $NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = "C:\Python314\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host ""
Write-Host "  SOC-EYE RAG pipeline" -ForegroundColor Cyan
Write-Host "  --------------------" -ForegroundColor Cyan

# --- MongoDB ---------------------------------------------------------------
$svc = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -ne "Running") {
    Write-Host "  MongoDB service is stopped - starting it..." -ForegroundColor Yellow
    Start-Service MongoDB
    Start-Sleep -Seconds 3
}

# --- preflight (Mongo + LLM reachability) ----------------------------------
$env:PYTHONIOENCODING = "utf-8"
& $py -c @"
import os, sys
from dotenv import load_dotenv
load_dotenv()
ok = True
try:
    from pymongo import MongoClient
    uri, db = os.getenv('MONGODB_URI'), os.getenv('DB_NAME')
    c = MongoClient(uri, serverSelectionTimeoutMS=5000); c.admin.command('ping')
    n = len(c[db].list_collection_names()); c.close()
    print(f'  MongoDB          : HEALTHY  {uri} db={db} ({n} collections)')
except Exception as e:
    ok = False; print(f'  MongoDB          : UNHEALTHY ({e})')
try:
    import llm_client
    llm_ok = llm_client.check_health()
    if not llm_ok:
        ok = False
    print(f"  LLM              : {'HEALTHY' if llm_ok else 'UNHEALTHY'}")
    print(f"  LLM model        : {llm_client.LLM_MODEL}")
    print(f"  LLM endpoint     : {llm_client.LLM_BASE_URL}")
except Exception as e:
    ok = False; print(f'  LLM              : UNHEALTHY ({e})')
try:
    from embedder import get_embedder
    import logging; logging.disable(logging.CRITICAL)
    emb = get_embedder().probe()
    print(f"  Embedding        : {'HEALTHY' if emb['healthy'] else 'UNHEALTHY'}")
    print(f"  Embedding model  : {emb['model']}")
    print(f"  Embedding dim    : {emb['dimension']} (corpus expects {emb['expected_dimension']})")
    print(f"  Embedding endpoint: {emb['endpoint']}")
    print(f"  Embedding provider: {emb['provider']}")
    if emb['error']:
        print(f"  Embedding error  : {emb['error']}")
        print("  -> semantic search is OFF; retrieval falls back to recency-ordered Mongo")
except Exception as e:
    ok = False
    print(f'  Embedding        : UNHEALTHY ({e})')
sys.exit(0 if ok else 1)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Preflight failed - fix the above before starting." -ForegroundColor Red
    exit 1
}

$url = "http://${BindHost}:${Port}"
Write-Host ""
Write-Host "  UI      $url/ui/"        -ForegroundColor Green
Write-Host "  Docs    $url/docs"       -ForegroundColor Green
Write-Host "  Health  $url/api/rag/health"
Write-Host "  Ctrl+C to stop."
Write-Host ""

if (-not $NoBrowser) {
    Start-Job -ScriptBlock { Start-Sleep -Seconds 3; Start-Process "$using:url/ui/" } | Out-Null
}

$uviArgs = @("-m","uvicorn","api_server:app","--host",$BindHost,"--port",$Port)
if ($Reload) { $uviArgs += "--reload" }
& $py @uviArgs
