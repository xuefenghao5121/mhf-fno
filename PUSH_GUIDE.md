# GitHub 推送指南

## 方法 1: 使用 GitHub CLI

```bash
gh auth login
gh repo create mhf-fno --public --source=. --push
```

## 方法 2: 手动推送

```bash
git remote add origin https://github.com/xuefenghao5121/mhf-fno.git
git push -u origin main
```