#!/bin/bash
# 创建GitHub仓库并推送

# 项目信息
REPO_NAME="mhf-fno"
REPO_DESC="Multi-Head Fourier Neural Operator - TransFourier-style MHF for NeuralOperator 2.0.0"

echo "=========================================="
echo " MHF-FNO GitHub 上传指南"
echo "=========================================="

echo ""
echo "项目已准备就绪！请按以下步骤上传到GitHub："
echo ""
echo "步骤1: 在GitHub上创建新仓库"
echo "  - 访问: https://github.com/new"
echo "  - 仓库名: $REPO_NAME"
echo "  - 描述: $REPO_DESC"
echo "  - 公开/私有: 选择公开"
echo "  - 不要初始化README（已有）"
echo ""
echo "步骤2: 添加远程仓库并推送"
echo ""

# 显示当前仓库状态
cd /root/.openclaw/workspace/memory/projects/tianyuan-fft
echo "当前分支: $(git branch --show-current)"
echo "提交数量: $(git rev-list --count HEAD)"
echo "文件数量: $(git ls-files | wc -l)"
echo ""
echo "运行以下命令推送:"
echo ""
echo "  git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo "  git push -u origin main"
echo ""