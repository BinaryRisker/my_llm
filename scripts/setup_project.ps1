# LLM项目自动化设置脚本 (PowerShell版本)
# 适用于Windows环境

param(
    [string]$Environment = "development",
    [switch]$SkipVirtualEnv = $false,
    [switch]$InstallGPUSupport = $false,
    [switch]$InstallDevelopmentTools = $false
)

# 脚本配置
$ProjectName = "LLM从零实现"
$VenvName = "llm_env"
$PythonVersion = "3.8+"

# 颜色输出函数
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success($message) {
    Write-ColorOutput Green "✅ $message"
}

function Write-Info($message) {
    Write-ColorOutput Blue "ℹ️  $message"
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "⚠️  $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "❌ $message"
}

function Write-Header($message) {
    Write-Output ""
    Write-ColorOutput Cyan "🚀 $message"
    Write-Output "=" * ($message.Length + 4)
}

# 检查Python版本
function Test-PythonVersion {
    Write-Info "检查Python版本..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Python未安装或不在PATH中"
            Write-Info "请安装Python 3.8+: https://www.python.org/downloads/"
            return $false
        }
        
        $version = $pythonVersion -replace "Python ", ""
        Write-Success "找到Python版本: $version"
        
        # 简单版本检查 (至少3.8)
        $majorMinor = $version.Split('.')[0..1] -join '.'
        if ([version]$majorMinor -lt [version]"3.8") {
            Write-Error "需要Python 3.8+，当前版本: $version"
            return $false
        }
        
        return $true
    }
    catch {
        Write-Error "检查Python版本时出错: $_"
        return $false
    }
}

# 创建虚拟环境
function New-VirtualEnvironment {
    if ($SkipVirtualEnv) {
        Write-Warning "跳过虚拟环境创建"
        return $true
    }
    
    Write-Info "创建虚拟环境: $VenvName"
    
    if (Test-Path $VenvName) {
        Write-Warning "虚拟环境已存在，将重新创建"
        Remove-Item -Recurse -Force $VenvName
    }
    
    python -m venv $VenvName
    if ($LASTEXITCODE -ne 0) {
        Write-Error "创建虚拟环境失败"
        return $false
    }
    
    Write-Success "虚拟环境创建成功"
    return $true
}

# 激活虚拟环境
function Enable-VirtualEnvironment {
    if ($SkipVirtualEnv) {
        return $true
    }
    
    Write-Info "激活虚拟环境..."
    
    $activateScript = Join-Path $VenvName "Scripts\Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        Write-Error "虚拟环境激活脚本不存在: $activateScript"
        return $false
    }
    
    & $activateScript
    Write-Success "虚拟环境已激活"
    return $true
}

# 升级pip
function Update-Pip {
    Write-Info "升级pip到最新版本..."
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -eq 0) {
        Write-Success "pip升级完成"
        return $true
    } else {
        Write-Warning "pip升级失败，继续安装..."
        return $true
    }
}

# 安装依赖包
function Install-Dependencies {
    Write-Info "安装项目依赖..."
    
    # 基础依赖
    $baseDeps = @(
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "gradio>=3.0.0"
    )
    
    foreach ($dep in $baseDeps) {
        Write-Info "安装: $dep"
        pip install $dep
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "安装 $dep 失败，跳过..."
        }
    }
    
    # GPU支持 (可选)
    if ($InstallGPUSupport) {
        Write-Info "安装GPU支持..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
    
    # 开发工具 (可选)
    if ($InstallDevelopmentTools) {
        Write-Info "安装开发工具..."
        $devDeps = @(
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0"
        )
        
        foreach ($dep in $devDeps) {
            pip install $dep
        }
    }
    
    Write-Success "依赖安装完成"
}

# 创建必要目录
function New-ProjectDirectories {
    Write-Info "创建项目目录结构..."
    
    $directories = @(
        "data",
        "logs", 
        "cache",
        "checkpoints",
        "experiments",
        "outputs",
        "test_data",
        "test_logs",
        "test_cache"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
            Write-Success "创建目录: $dir"
        } else {
            Write-Info "目录已存在: $dir"
        }
    }
}

# 设置环境变量
function Set-EnvironmentVariables {
    Write-Info "设置环境变量..."
    
    # 设置项目环境
    $env:LLM_ENV = $Environment
    $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
    
    # 创建环境变量脚本
    $envScript = @"
# LLM项目环境变量
`$env:LLM_ENV = "$Environment"
`$env:PYTHONPATH = "$PWD;`$env:PYTHONPATH"

Write-Host "🚀 LLM项目环境已设置 (环境: $Environment)" -ForegroundColor Green
"@
    
    $envScript | Out-File -FilePath "set_env.ps1" -Encoding UTF8
    Write-Success "环境变量脚本创建: set_env.ps1"
}

# 运行基础测试
function Test-Installation {
    Write-Info "测试安装..."
    
    # 测试Python导入
    $testScript = @"
import sys
print(f'Python版本: {sys.version}')

try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch导入失败: {e}')

try:
    import transformers
    print(f'Transformers版本: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers导入失败: {e}')

try:
    from config import get_config
    config = get_config(env='$Environment')
    print(f'配置加载成功，项目: {config.get(\"project\", {}).get(\"name\", \"未知\")}')
except Exception as e:
    print(f'配置加载失败: {e}')

print('\\n✅ 基础测试完成！')
"@
    
    echo $testScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "安装测试通过"
        return $true
    } else {
        Write-Warning "安装测试有警告，但可以继续"
        return $true
    }
}

# 显示使用说明
function Show-Usage {
    Write-Header "设置完成！"
    
    Write-Output ""
    Write-Info "项目设置完成，以下是使用说明："
    Write-Output ""
    
    if (-not $SkipVirtualEnv) {
        Write-Output "1. 激活虚拟环境："
        Write-ColorOutput Yellow "   .\$VenvName\Scripts\Activate.ps1"
        Write-Output ""
    }
    
    Write-Output "2. 设置环境变量："
    Write-ColorOutput Yellow "   .\set_env.ps1"
    Write-Output ""
    
    Write-Output "3. 运行测试："
    Write-ColorOutput Yellow "   python test_bert.py"
    Write-Output ""
    
    Write-Output "4. 启动Web界面："
    Write-ColorOutput Yellow "   python web_interface\gradio_demo.py"
    Write-Output ""
    
    Write-Output "5. 配置文件位置："
    Write-ColorOutput Yellow "   config\$Environment.yaml"
    Write-Output ""
    
    Write-Success "🎉 $ProjectName 项目设置完成！"
    Write-Info "如需帮助，请查看 README.md 或 TODO_IMPROVEMENTS.md"
}

# 主函数
function Main {
    Write-Header "$ProjectName 自动设置"
    
    Write-Info "设置参数："
    Write-Info "  环境: $Environment"
    Write-Info "  跳过虚拟环境: $SkipVirtualEnv"
    Write-Info "  安装GPU支持: $InstallGPUSupport" 
    Write-Info "  安装开发工具: $InstallDevelopmentTools"
    Write-Output ""
    
    # 步骤1: 检查Python
    if (-not (Test-PythonVersion)) {
        Write-Error "Python检查失败，设置中止"
        exit 1
    }
    
    # 步骤2: 创建虚拟环境
    if (-not (New-VirtualEnvironment)) {
        Write-Error "虚拟环境创建失败，设置中止"
        exit 1
    }
    
    # 步骤3: 激活虚拟环境
    if (-not (Enable-VirtualEnvironment)) {
        Write-Error "虚拟环境激活失败，设置中止"
        exit 1
    }
    
    # 步骤4: 升级pip
    Update-Pip
    
    # 步骤5: 安装依赖
    Install-Dependencies
    
    # 步骤6: 创建目录
    New-ProjectDirectories
    
    # 步骤7: 设置环境变量
    Set-EnvironmentVariables
    
    # 步骤8: 测试安装
    Test-Installation
    
    # 步骤9: 显示使用说明
    Show-Usage
}

# 运行主函数
Main