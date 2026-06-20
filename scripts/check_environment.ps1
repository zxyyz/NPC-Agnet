#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$Root
)

$ErrorActionPreference = 'Stop'

$links = [ordered]@{
    Python = 'https://www.python.org/downloads/windows/'
    NvidiaDriver = 'https://www.nvidia.com/Download/index.aspx'
    CudaToolkit = 'https://developer.nvidia.com/cuda-downloads'
    LlamaCpp = 'https://github.com/ggml-org/llama.cpp/releases'
    VcRedist = 'https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist'
    TtsModel = 'https://huggingface.co/BricksDisplay/chatterbox-multilingual-ONNX-q4'
    GgufModels = 'https://huggingface.co/models?library=gguf'
}

$issues = New-Object System.Collections.Generic.List[object]

function Add-Issue {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Message,
        [Parameter(Mandatory)][string]$Fix,
        [Parameter(Mandatory)][string]$Url
    )

    $issues.Add([pscustomobject]@{
        Name = $Name
        Message = $Message
        Fix = $Fix
        Url = $Url
    }) | Out-Null
}

function Test-Command {
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-ProjectPython {
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }

    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    $pythonw = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($pythonw) { return $pythonw.Source }

    return $null
}

function Invoke-Python {
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$Code
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ([IO.Path]::GetFileName($Python).Equals('py.exe', [StringComparison]::OrdinalIgnoreCase)) {
            return & $Python -3 -c $Code 2>&1
        }

        return & $Python -c $Code 2>&1
    }
    finally {
        $script:ErrorActionPreference = $previousPreference
    }
}

Write-Host ''
Write-Host 'Checking runtime environment...' -ForegroundColor Cyan

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$requirementsPath = Join-Path $rootPath 'requirements.txt'
$llamaServerPath = Join-Path $rootPath 'runtime\llama.cpp\llama-server.exe'
$settingsPath = Join-Path $rootPath 'config\agent_settings.json'
$defaultModelPath = Join-Path $rootPath 'models\llm\Qwen3.5-4B-Q4_K_M-GGUF\qwen3-5-4B-Q4_K_M.gguf'
$ttsModelDir = Join-Path $rootPath 'models\tts\BricksDisplay-chatterbox-multilingual-ONNX-q4'

$python = Get-ProjectPython
if (-not $python) {
    Add-Issue `
        -Name 'Python' `
        -Message 'python.exe, pythonw.exe, or py.exe was not found in PATH.' `
        -Fix 'Install 64-bit Python and enable "Add Python to PATH".' `
        -Url $links.Python
}
else {
    $versionOutput = Invoke-Python -Python $python -Code 'import sys; print(chr(46).join(map(str, sys.version_info[:3]))); raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
    if ($LASTEXITCODE -ne 0) {
        Add-Issue `
            -Name 'Python version' `
            -Message "Python is too old or unavailable: $($versionOutput -join ' ')" `
            -Fix 'Install Python 3.10 or newer.' `
            -Url $links.Python
    }

    if (Test-Path -LiteralPath $requirementsPath) {
        $missing = New-Object System.Collections.Generic.List[string]
        foreach ($package in Get-Content -LiteralPath $requirementsPath) {
            $name = $package.Trim()
            if (-not $name -or $name.StartsWith('#')) { continue }
            $importName = switch -Regex ($name) {
                '^onnxruntime-gpu' { 'onnxruntime'; break }
                default { ($name -split '[<>=!~ ]')[0].Replace('-', '_') }
            }
            Invoke-Python -Python $python -Code "import $importName" | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $missing.Add($name) | Out-Null
            }
        }

        if ($missing.Count -gt 0) {
            Add-Issue `
                -Name 'Python packages' `
                -Message "Missing packages: $($missing -join ', ')" `
                -Fix "Run: `"$python`" -m pip install -r `"$requirementsPath`"" `
                -Url 'https://pypi.org/'
        }
    }
    else {
        Add-Issue `
            -Name 'requirements.txt' `
            -Message 'requirements.txt is missing from the project root.' `
            -Fix 'Restore requirements.txt, then start again.' `
            -Url $links.Python
    }
}

if (-not (Test-Command 'nvidia-smi.exe')) {
    Add-Issue `
        -Name 'NVIDIA driver' `
        -Message 'nvidia-smi.exe was not found. GPU/CUDA may be unavailable.' `
        -Fix 'Install or update the NVIDIA driver. Install CUDA Toolkit if local CUDA runtime files are also needed.' `
        -Url $links.NvidiaDriver
}

if ($python) {
    $providersOutput = Invoke-Python -Python $python -Code 'import onnxruntime as ort; print(ort.get_available_providers())'
    if ($LASTEXITCODE -ne 0 -or (($providersOutput -join ' ') -notmatch 'CUDAExecutionProvider')) {
        Add-Issue `
            -Name 'ONNXRuntime CUDA' `
            -Message 'onnxruntime-gpu does not expose CUDAExecutionProvider.' `
            -Fix 'Check NVIDIA driver, CUDA runtime compatibility, and onnxruntime-gpu installation.' `
            -Url $links.CudaToolkit
    }
}

if (-not (Test-Path -LiteralPath $llamaServerPath)) {
    Add-Issue `
        -Name 'llama.cpp' `
        -Message "Missing llama-server.exe: $llamaServerPath" `
        -Fix 'Download a Windows llama.cpp build and place it under runtime\llama.cpp.' `
        -Url $links.LlamaCpp
}

if (-not (Test-Path -LiteralPath (Join-Path $ttsModelDir 'onnx\language_model.onnx'))) {
    Add-Issue `
        -Name 'TTS model' `
        -Message "Missing Chatterbox ONNX model directory: $ttsModelDir" `
        -Fix 'Restore models\tts\BricksDisplay-chatterbox-multilingual-ONNX-q4.' `
        -Url $links.TtsModel
}

$modelPath = $defaultModelPath
if (Test-Path -LiteralPath $settingsPath) {
    try {
        $settings = Get-Content -Raw -LiteralPath $settingsPath | ConvertFrom-Json
        if ($settings.model_path) { $modelPath = [string]$settings.model_path }
    }
    catch {
        Add-Issue `
            -Name 'Config file' `
            -Message "Cannot parse config file: $settingsPath" `
            -Fix 'Check that agent_settings.json is valid JSON.' `
            -Url 'https://www.json.org/json-en.html'
    }
}

if (-not (Test-Path -LiteralPath $modelPath)) {
    Add-Issue `
        -Name 'LLM model' `
        -Message "Missing configured GGUF model file: $modelPath" `
        -Fix 'Place a GGUF model under models\llm and select it in the control panel settings.' `
        -Url $links.GgufModels
}

if ($issues.Count -eq 0) {
    Write-Host 'Environment check passed.' -ForegroundColor Green
    exit 0
}

Write-Host ''
Write-Host 'Environment check failed. Startup has been stopped.' -ForegroundColor Red
Write-Host ''

$index = 1
foreach ($issue in $issues) {
    Write-Host "[$index] $($issue.Name)" -ForegroundColor Yellow
    Write-Host "    Problem: $($issue.Message)"
    Write-Host "    Fix:     $($issue.Fix)"
    Write-Host "    Link:    $($issue.Url)"
    Write-Host ''
    $index++
}

Write-Host 'Official download links:' -ForegroundColor Cyan
Write-Host "  Python:        $($links.Python)"
Write-Host "  NVIDIA driver: $($links.NvidiaDriver)"
Write-Host "  CUDA Toolkit:  $($links.CudaToolkit)"
Write-Host "  llama.cpp:     $($links.LlamaCpp)"
Write-Host "  VC++ runtime:  $($links.VcRedist)"
Write-Host ''
Write-Host 'Fix the items above, then run start.bat again.'

exit 1
