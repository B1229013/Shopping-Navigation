# No-Python gateway smoke test (raw HTTPS, same as verify_gateway.py proves via the SDK).
# Reads CGU_API_KEY from ../.env and calls one OpenAI chat model + one local model.
#   powershell -ExecutionPolicy Bypass -File backend\verify_gateway.ps1
$ErrorActionPreference = "Stop"
$envPath = Join-Path $PSScriptRoot "..\.env"
if (-not (Test-Path $envPath)) { Write-Host "No .env at $envPath"; exit 1 }
$kv = @{}
foreach ($line in Get-Content $envPath) {
  $t = $line.Trim()
  if ($t -and -not $t.StartsWith("#") -and $t.Contains("=")) {
    $i = $t.IndexOf("="); $kv[$t.Substring(0,$i).Trim()] = ($t.Substring($i+1) -split "#")[0].Trim()
  }
}
$key = $kv["CGU_API_KEY"]; $base = $kv["CGU_BASE_URL"]
if (-not $key -or $key -eq "PASTE_YOUR_CGU_KEY_HERE") { Write-Host "CGU_API_KEY not set in .env yet."; exit 1 }
$hdr = @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" }

Write-Host "GET $base/v1/models ..."
try { $m = Invoke-RestMethod -Uri "$base/models" -Headers $hdr -TimeoutSec 30; Write-Host ("  OK -> {0} models" -f $m.data.Count) }
catch { Write-Host ("  FAILED -> " + $_.Exception.Message); exit 1 }

foreach ($model in @("gpt-5.4-mini","gpt-oss:20b")) {
  $body = @{ model=$model; messages=@(@{role="user";content="Reply with exactly: ok"}); max_tokens=10 } | ConvertTo-Json -Depth 6
  try {
    $r = Invoke-RestMethod -Uri "$base/chat/completions" -Method Post -Headers $hdr -Body $body -TimeoutSec 90
    Write-Host ("  {0}: OK -> '{1}'" -f $model, ($r.choices[0].message.content).Trim())
  } catch { Write-Host ("  {0}: FAILED -> {1}" -f $model, $_.Exception.Message) }
}
