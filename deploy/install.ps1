$binDir = "$HOME\.pixi\bin"
if (!(Test-Path $binDir)) { New-Item -ItemType Directory -Force -Path $binDir }
$target = "$PWD\.pixi\envs\default\Scripts\rapida.exe"
Set-Content "$binDir\rapida.cmd" "@echo off`n`"$target`" %*"
Write-Host "Rapida global launcher installed successfully!" -ForegroundColor Green