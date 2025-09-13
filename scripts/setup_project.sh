#!/bin/bash

# LLM项目自动化设置脚本 (Bash版本)
# 适用于Linux/macOS环境

# 脚本配置
PROJECT_NAME="LLM从零实现"
VENV_NAME="llm_env"
PYTHON_VERSION="3.8+"

# 默认参数
ENVIRONMENT="development"
SKIP_VIRTUAL_ENV=false
INSTALL_GPU_SUPPORT=false
INSTALL_DEVELOPMENT_TOOLS=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 输出函数
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${CYAN}🚀 $1${NC}"
    echo "$(printf '=%.0s' $(seq 1 $((${#1} + 4))))"
}

# 显示帮助信息
show_help() {
    echo "LLM项目自动化设置脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -e, --environment ENV    设置环境 (development|testing|production) [默认: development]"
    echo "  -s, --skip-venv         跳过虚拟环境创建"
    echo "  -g, --gpu               安装GPU支持 (CUDA)"
    echo "  -d, --dev-tools         安装开发工具"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 基础安装"
    echo "  $0 -e production -g                  # 生产环境 + GPU支持"
    echo "  $0 -s -d                            # 跳过虚拟环境 + 开发工具"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--skip-venv)
            SKIP_VIRTUAL_ENV=true
            shift
            ;;
        -g|--gpu)
            INSTALL_GPU_SUPPORT=true
            shift
            ;;
        -d|--dev-tools)
            INSTALL_DEVELOPMENT_TOOLS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查Python版本
check_python_version() {
    print_info "检查Python版本..."
    
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "Python未安装或不在PATH中"
        print_info "请安装Python 3.8+："
        print_info "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        print_info "  macOS: brew install python3"
        return 1
    fi
    
    # 优先使用python3
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
    else
        PYTHON_CMD="python"
        PIP_CMD="pip"
    fi
    
    VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "找到Python版本: $VERSION (命令: $PYTHON_CMD)"
    
    # 检查版本是否满足要求 (至少3.8)
    MAJOR=$(echo $VERSION | cut -d. -f1)
    MINOR=$(echo $VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        print_error "需要Python 3.8+，当前版本: $VERSION"
        return 1
    fi
    
    return 0
}

# 创建虚拟环境
create_virtual_environment() {
    if [ "$SKIP_VIRTUAL_ENV" = true ]; then
        print_warning "跳过虚拟环境创建"
        return 0
    fi
    
    print_info "创建虚拟环境: $VENV_NAME"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "虚拟环境已存在，将重新创建"
        rm -rf "$VENV_NAME"
    fi
    
    $PYTHON_CMD -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        print_error "创建虚拟环境失败"
        return 1
    fi
    
    print_success "虚拟环境创建成功"
    return 0
}

# 激活虚拟环境
activate_virtual_environment() {
    if [ "$SKIP_VIRTUAL_ENV" = true ]; then
        return 0
    fi
    
    print_info "激活虚拟环境..."
    
    ACTIVATE_SCRIPT="$VENV_NAME/bin/activate"
    if [ ! -f "$ACTIVATE_SCRIPT" ]; then
        print_error "虚拟环境激活脚本不存在: $ACTIVATE_SCRIPT"
        return 1
    fi
    
    source "$ACTIVATE_SCRIPT"
    
    # 更新命令引用
    PYTHON_CMD="python"
    PIP_CMD="pip"
    
    print_success "虚拟环境已激活"
    return 0
}

# 升级pip
update_pip() {
    print_info "升级pip到最新版本..."
    $PIP_CMD install --upgrade pip
    if [ $? -eq 0 ]; then
        print_success "pip升级完成"
    else
        print_warning "pip升级失败，继续安装..."
    fi
}

# 安装依赖包
install_dependencies() {
    print_info "安装项目依赖..."
    
    # 基础依赖
    BASE_DEPS=(
        "torch>=2.0.0"
        "transformers>=4.20.0"
        "datasets>=2.0.0"
        "numpy>=1.21.0"
        "pandas>=1.3.0"
        "matplotlib>=3.5.0"
        "seaborn>=0.11.0"
        "scikit-learn>=1.1.0"
        "tqdm>=4.64.0"
        "pyyaml>=6.0"
        "gradio>=3.0.0"
    )
    
    for dep in "${BASE_DEPS[@]}"; do
        print_info "安装: $dep"
        $PIP_CMD install "$dep"
        if [ $? -ne 0 ]; then
            print_warning "安装 $dep 失败，跳过..."
        fi
    done
    
    # GPU支持 (可选)
    if [ "$INSTALL_GPU_SUPPORT" = true ]; then
        print_info "安装GPU支持..."
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # 开发工具 (可选)
    if [ "$INSTALL_DEVELOPMENT_TOOLS" = true ]; then
        print_info "安装开发工具..."
        DEV_DEPS=(
            "pytest>=7.0.0"
            "pytest-cov>=3.0.0"
            "black>=22.0.0"
            "isort>=5.10.0"
            "flake8>=4.0.0"
            "mypy>=0.950"
            "jupyter>=1.0.0"
        )
        
        for dep in "${DEV_DEPS[@]}"; do
            $PIP_CMD install "$dep"
        done
    fi
    
    print_success "依赖安装完成"
}

# 创建必要目录
create_project_directories() {
    print_info "创建项目目录结构..."
    
    DIRECTORIES=(
        "data"
        "logs"
        "cache"
        "checkpoints"
        "experiments"
        "outputs"
        "test_data"
        "test_logs"
        "test_cache"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "创建目录: $dir"
        else
            print_info "目录已存在: $dir"
        fi
    done
}

# 设置环境变量
set_environment_variables() {
    print_info "设置环境变量..."
    
    # 设置当前会话的环境变量
    export LLM_ENV="$ENVIRONMENT"
    export PYTHONPATH="$PWD:$PYTHONPATH"
    
    # 创建环境变量脚本
    cat > set_env.sh << EOF
#!/bin/bash
# LLM项目环境变量

export LLM_ENV="$ENVIRONMENT"
export PYTHONPATH="$PWD:\$PYTHONPATH"

echo -e "${GREEN}🚀 LLM项目环境已设置 (环境: $ENVIRONMENT)${NC}"
EOF
    
    chmod +x set_env.sh
    print_success "环境变量脚本创建: set_env.sh"
}

# 运行基础测试
test_installation() {
    print_info "测试安装..."
    
    # 创建测试脚本
    cat > test_install.py << 'EOF'
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
    config = get_config(env='development')
    print(f'配置加载成功，项目: {config.get("project", {}).get("name", "未知")}')
except Exception as e:
    print(f'配置加载失败: {e}')

print('\n✅ 基础测试完成！')
EOF
    
    $PYTHON_CMD test_install.py
    rm -f test_install.py
    
    if [ $? -eq 0 ]; then
        print_success "安装测试通过"
    else
        print_warning "安装测试有警告，但可以继续"
    fi
}

# 显示使用说明
show_usage() {
    print_header "设置完成！"
    
    echo ""
    print_info "项目设置完成，以下是使用说明："
    echo ""
    
    if [ "$SKIP_VIRTUAL_ENV" = false ]; then
        echo "1. 激活虚拟环境："
        echo -e "   ${YELLOW}source $VENV_NAME/bin/activate${NC}"
        echo ""
    fi
    
    echo "2. 设置环境变量："
    echo -e "   ${YELLOW}source set_env.sh${NC}"
    echo ""
    
    echo "3. 运行测试："
    echo -e "   ${YELLOW}python test_bert.py${NC}"
    echo ""
    
    echo "4. 启动Web界面："
    echo -e "   ${YELLOW}python web_interface/gradio_demo.py${NC}"
    echo ""
    
    echo "5. 配置文件位置："
    echo -e "   ${YELLOW}config/$ENVIRONMENT.yaml${NC}"
    echo ""
    
    print_success "🎉 $PROJECT_NAME 项目设置完成！"
    print_info "如需帮助，请查看 README.md 或 TODO_IMPROVEMENTS.md"
}

# 主函数
main() {
    print_header "$PROJECT_NAME 自动设置"
    
    print_info "设置参数："
    print_info "  环境: $ENVIRONMENT"
    print_info "  跳过虚拟环境: $SKIP_VIRTUAL_ENV"
    print_info "  安装GPU支持: $INSTALL_GPU_SUPPORT"
    print_info "  安装开发工具: $INSTALL_DEVELOPMENT_TOOLS"
    echo ""
    
    # 步骤1: 检查Python
    if ! check_python_version; then
        print_error "Python检查失败，设置中止"
        exit 1
    fi
    
    # 步骤2: 创建虚拟环境
    if ! create_virtual_environment; then
        print_error "虚拟环境创建失败，设置中止"
        exit 1
    fi
    
    # 步骤3: 激活虚拟环境
    if ! activate_virtual_environment; then
        print_error "虚拟环境激活失败，设置中止"
        exit 1
    fi
    
    # 步骤4: 升级pip
    update_pip
    
    # 步骤5: 安装依赖
    install_dependencies
    
    # 步骤6: 创建目录
    create_project_directories
    
    # 步骤7: 设置环境变量
    set_environment_variables
    
    # 步骤8: 测试安装
    test_installation
    
    # 步骤9: 显示使用说明
    show_usage
}

# 运行主函数
main "$@"