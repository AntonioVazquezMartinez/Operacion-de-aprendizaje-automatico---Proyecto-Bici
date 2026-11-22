# Prueba de humo del contenedor en un solo comando (Windows PowerShell 5.1 o PowerShell 7):
# build + run + espera healthy + verificación de reproducibilidad + limpieza.
#
# Uso (desde la raíz del repositorio):
#   powershell -ExecutionPolicy Bypass -File scripts\docker_smoke.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\docker_smoke.ps1 -SkipBuild `
#       -Image gusrivasmna/mibici-demand-service:1.1.0
param(
    [string]$Image = "ml-service:latest",
    [int]$Port = 8000,
    [string]$Label = "contenedor-local",
    [string]$Output = "",
    [string]$Python = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if (-not $Output) { $Output = "reports/fase_final/reproducibilidad/docker_smoke_$Label.json" }
if (-not $Python) {
    if (Test-Path ".venv\Scripts\python.exe") { $Python = ".venv\Scripts\python.exe" } else { $Python = "python" }
}
$Name = "mibici-smoke-$PID"
$exitCode = 1

try {
    if (-not $SkipBuild) {
        Write-Host "== docker build -t $Image ."
        docker build -t $Image .
        if ($LASTEXITCODE -ne 0) { throw "docker build falló (código $LASTEXITCODE)" }
    }

    Write-Host "== docker run -d -p ${Port}:8000 $Image"
    docker run -d --name $Name -p "${Port}:8000" $Image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "docker run falló (código $LASTEXITCODE)" }

    Write-Host "== esperando HEALTHCHECK"
    $status = "starting"
    for ($i = 0; $i -lt 40; $i++) {
        $status = (docker inspect --format "{{.State.Health.Status}}" $Name).Trim()
        if ($status -eq "healthy" -or $status -eq "unhealthy") { break }
        Start-Sleep -Seconds 3
    }
    Write-Host "estado: $status"
    if ($status -ne "healthy") {
        docker logs $Name
        throw "El contenedor no llegó a healthy"
    }

    Write-Host "== verificación de reproducibilidad"
    & $Python scripts/verify_reproducibility.py --url "http://localhost:$Port" --label $Label --output $Output
    $exitCode = $LASTEXITCODE
}
finally {
    try { docker rm -f $Name *> $null } catch { }
}
exit $exitCode
