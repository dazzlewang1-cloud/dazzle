# Streamlit Cloud 部署步骤

现在这个目录已经整理成 Streamlit Community Cloud 可直接发布的结构了。别人以后打开你的网页链接，就能直接在浏览器里使用，不需要下载代码。

## 你现在手里有什么

目录：

- `app.py`
- `predictor.py`
- `requirements.txt`
- `packages.txt`
- `README.md`
- `assets/`

其中：

- `requirements.txt` 是 Python 依赖
- `packages.txt` 是 Streamlit Cloud 上需要的系统依赖
- `assets/` 已经是自包含模型文件，不再依赖外部目录

## 我为什么改成 Streamlit Community Cloud

你的模型是已经保存好的 `best_model.pkl`。

这类 sklearn / joblib 模型最怕部署环境版本漂移，尤其是 Python 和依赖版本不一致。这里最关键的是部署时把 Python 版本选成 `3.10`，因为这套模型我已经在本地 `Python 3.10` 下实际验证通过了。

另外我已经补了 `packages.txt`，里面放了 `libgomp1`，这是 `scikit-learn` 在 Linux 环境里常见需要的系统库。

## 你整理完后应该怎么做

你现在只需要做下面这几步。

### 第 1 步：把这个目录内容放到 GitHub 仓库根目录

不是上传整个 `/home/wlz/code`，只上传 `extratrees_online_calculator` 目录里的内容。

上传后仓库根目录应该直接看到：

- `app.py`
- `predictor.py`
- `requirements.txt`
- `packages.txt`
- `assets/`

### 第 2 步：打开 Streamlit Community Cloud

打开：

`https://share.streamlit.io/`

用 GitHub 账号登录后，点击新建应用。

### 第 3 步：选择仓库和入口文件

部署时这样填：

- Repository：你的 GitHub 仓库
- Branch：`main`
- Main file path：`app.py`

### 第 4 步：设置 Python 版本

打开 `Advanced settings`，把 Python 版本改成：

```text
3.10
```

这一点很关键。

### 第 5 步：点击 Deploy

正常情况下，平台会自动读取：

- `requirements.txt`
- `packages.txt`
- `app.py`
- `assets/`

## 构建成功后你会得到什么

你会得到一个公开网页链接。

别人打开后可以直接：

- 填写患者变量
- 点击计算
- 看到 `MCE` 概率
- 看到 `3个月不良功能结局` 概率
- 看到 `联合模型` 概率

整个过程都在网页里完成，不需要别人安装 Python，也不需要别人下载模型。

## 如果构建失败，优先检查这几项

1. 部署时 `Main file path` 是否填的是 `app.py`
2. 部署时 Python 版本是否选了 `3.10`
3. 根目录里是否直接有 `requirements.txt`
4. `assets/` 是否完整上传
5. 上传的是否是 `extratrees_online_calculator` 目录内容本身，而不是外面又多包了一层目录
6. `packages.txt` 是否也在仓库根目录

## 你下一步最简操作

如果你想最快上线，现在就按这个顺序做：

1. 把代码推到 GitHub
2. 打开 `https://share.streamlit.io/`
3. 选择仓库
4. 把入口文件设成 `app.py`
5. 把 Python 版本设成 `3.10`
6. 点击 `Deploy`

## 很重要：如果你已经用错 Python 版本部署过一次

Streamlit 官方说明里写得很明确：应用一旦创建后，不能直接修改 Python 主版本。

如果你第一次部署时没有选对版本，正确做法是：

1. 删除当前这个 app
2. 重新从 GitHub 仓库创建
3. 在创建时就把 Python 版本选成 `3.10`
