# Headless screenshots of the ct2skel viewer (CT pose + saved poses) for visual verification.
#   powershell -File scripts\viewer_shots.ps1 -Url http://127.0.0.1:8001/index.html -Poses ct,sitting,arms_up -Prefix m63
param(
    [string]$Url = "http://127.0.0.1:8000/index.html",
    [string]$Poses = "ct,sitting,arms_up",      # comma-separated; "ct" = fitted pose
    [string]$Prefix = "shot",
    [string]$OutDir = $env:TEMP
)
$chrome = @("C:\Program Files\Google\Chrome\Application\chrome.exe", "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) { Write-Error "no Chrome/Edge found"; exit 1 }
foreach ($pose in ($Poses -split ",")) {
    $pose = $pose.Trim(); if ($pose -eq "ct") { $pose = "" }
    $tag = if ($pose) { $pose } else { "ct" }
    $out = Join-Path $OutDir "$Prefix`_$tag.png"
    if (Test-Path $out) { Remove-Item $out }
    $target = if ($pose) { "$Url`?pose=$pose" } else { $Url }
    $args = "--headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist --hide-scrollbars " +
            "--window-size=1500,1000 --virtual-time-budget=150000 --timeout=200000 --screenshot=`"$out`" `"$target`""
    $p = Start-Process -FilePath $chrome -ArgumentList $args -PassThru -WindowStyle Hidden
    $p.WaitForExit(260000) | Out-Null
    if (-not $p.HasExited) { $p.Kill() }
    if (Test-Path $out) { Write-Output "$tag -> $out" } else { Write-Output "$tag FAILED" }
}
