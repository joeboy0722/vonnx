param([string]$Python = "$PSScriptRoot\venv\python.exe", [switch]$SkipDll)
$ErrorActionPreference = 'Stop'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installation = & $vswhere -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $installation) { throw 'Visual Studio 2022 C++ tools required.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$batch = Join-Path $PSScriptRoot 'build\build_vonnx.cmd'
New-Item -ItemType Directory -Force (Split-Path $batch) | Out-Null
$commands = @("@echo off", "chcp 65001 >nul", "call `"$vcvars`"", 'if errorlevel 1 exit /b %errorlevel%', 'set "DISTUTILS_USE_SDK=1"', 'set "PYTHONUTF8=1"', "cd /d `"$PSScriptRoot`"", "`"$Python`" setup.py build_ext --inplace --force", 'if errorlevel 1 exit /b %errorlevel%')
if (-not $SkipDll) { $commands += "`"$Python`" tools\build_native.py" }
$commands | Set-Content -Encoding utf8 $batch
& cmd /c $batch
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)." }
