# Data 目录

本目录存放训练和测试数据。

## 数据生成

使用 `benchmark/generate_data.py` 生成数据：

```bash
cd benchmark

# Darcy Flow 2D
python generate_data.py --dataset darcy --n_train 500 --n_test 100

# Burgers 1D
python generate_data.py --dataset burgers --n_train 500 --n_test 100

# Navier-Stokes 2D
python generate_data.py --dataset navier_stokes --n_train 200 --n_test 50
```

数据会自动保存到 `data/` 目录。

## 注意

- 数据文件不提交到 Git (已在 .gitignore 中配置)
- 运行测试前请先生成数据