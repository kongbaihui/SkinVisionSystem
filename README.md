# 智肤研析 · SkinVision

面向医务人员的皮肤镜图像辅助分析工作台。医生上传图像后，可以对照原图、病灶分割和热力图，查看七分类候选，复核并修改报告草稿，最后保存和导出。系统用于教学与研究，分类分数未经临床校准，报告需要人工复核。[选题报告](docs/智肤研析-选题报告final.docx)

## 本地安装

Python 支持 3.10–3.12。以下示例从仓库根目录运行，使用 Windows PowerShell、Python 3.12、Node.js 22 和 CPU 版 PyTorch。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/runtime.txt
Copy-Item env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

将最后一条命令生成的值填入本机 `.env` 的 `SKINVISION_AUTH_SECRET`，保留 `SKINVISION_ENABLE_AUTH=true`。已有部署沿用原来的密钥、数据库和图像目录。Linux 使用 `source .venv/bin/activate` 和 `cp env.example .env`。需要训练与实验时另装 `python -m pip install -r requirements/training.txt`，需要 pytest 和 ONNX 验证时装 `python -m pip install -r requirements/dev.txt`。CUDA 用户按版本表安装匹配的 torch 和 torchvision，纯 PyTorch 路径可直接运行。[PyTorch 安装矩阵](https://pytorch.org/get-started/previous-versions/)

Windows 的 Python 3.12 CPU 环境也可按完整锁文件安装运行、训练与测试依赖。创建并激活虚拟环境后执行以下命令，再按前文设置本机 `.env`。

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/lock-windows-py312-cpu.txt
```

### 配置病灶分割

有两种取得 LMDAUNet 权重的方法。

1. **使用项目发布的权重。** 从 v1.0.0 下载 `best-epoch45-loss1.6684.pth`，放在本机 `weights/segmentation/`。文件为 93,488,239 字节，SHA-256 为 `a6dac925e6924c99267b2796130d2ff84b367e4976da798bdcdb66abadf65567`。PowerShell 可用下列命令下载和验证。[v1.0.0 Release](https://github.com/kongbaihui/SkinVisionSystem/releases/tag/v1.0.0)
2. **用公开分割数据自行训练。** 按下文“数据准备、训练和评估”生成 `best-epoch*-loss*.pth`。将其真实路径填入同一个 `SKINVISION_WEIGHTS` 配置项，重启服务。模型和数据分别下载，Git 克隆只包含代码与选题报告。

```powershell
New-Item -ItemType Directory -Force weights/segmentation | Out-Null
$weights = 'weights/segmentation/best-epoch45-loss1.6684.pth'
Invoke-WebRequest 'https://github.com/kongbaihui/SkinVisionSystem/releases/download/v1.0.0/best-epoch45-loss1.6684.pth' -OutFile $weights
if ((Get-FileHash $weights -Algorithm SHA256).Hash -ne 'a6dac925e6924c99267b2796130d2ff84b367e4976da798bdcdb66abadf65567') { throw '分割权重 SHA-256 不匹配' }
```

发布权重对应的 `.env` 设置如下。该归一化方式由这份权重在 8 张校准图上的平均 Dice 选出，更换权重后需要在单独的验证集上复核。模型输出取四个分割头 sigmoid 的均值，默认以 0.5 二值化。缺少或无法加载权重时，任务使用 Otsu 阈值分割，并记录 `fallback`。此时热力图和模型可靠性评估不可用。

```dotenv
SKINVISION_WEIGHTS=./weights/segmentation/best-epoch45-loss1.6684.pth
SKINVISION_NORMALIZATION=train_minmax_255
```

### 配置分类和报告

```powershell
python -m scripts.data_preparation.download_research_assets classifier
```

该命令下载并校验 `adeIaide/skin-lesion-7class-efficientnet` 的配置、模型卡和 `model.safetensors`，固定修订为 `b5ebe73c3b89baa90ee4f991afa318385414a973`，权重 SHA-256 为 `fed0c3cb0a15823f4f6a82d4dd2725a0b8e6068a971664e1504403fba5b54562`。默认 `.env` 已指定 `SKINVISION_CLASSIFICATION_BACKEND=local` 和目录 `./weights/classification/efficientnet-b0-research`。本地推理按模型卡顺序解释 MEL、NV、BCC、AKIEC、BKL、DF、VASC，显示排名靠前的候选，适用许可为 CC BY-NC 4.0。[固定版本模型卡](https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet/tree/b5ebe73c3b89baa90ee4f991afa318385414a973)

开启在线多模态报告时，在后端 `.env` 填入 `DASHSCOPE_API_KEY` 或设置 `ALIYUN_API_KEY`，默认模型为 `qwen3-vl-plus`。开启后，所选报告模式会将相关图像和填写的临床信息发送给服务提供方。也可将 `SKINVISION_CLASSIFICATION_BACKEND` 改为 `api`，单独使用在线分类。离线使用时留空 API 密钥，系统生成本地模板草稿，本地七分类仍可工作。

## 启动与医生操作

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
python -m alembic upgrade head
python -m uvicorn backend.skinvision.app:app --host 127.0.0.1 --port 8000
```

打开 [医生工作台](http://127.0.0.1:8000/) 。前端开发可在另一个终端运行 `npm --prefix frontend run dev`，浏览 [Vite 开发入口](http://localhost:5173/) 。后端服务监听 8000 端口，开发代理转发 API、流式输出与图像请求。服务状态及认证开关见 [健康检查](http://127.0.0.1:8000/health)。

1. **注册并登录。** 医生账号使用 3–20 位字母、数字或下划线作为用户名，密码至少 8 位且包含字母和数字。
2. **上传并分析。** 在工作台选择皮肤镜图像，按已有资料填写患者编号、年龄、症状与病程。点击“开始分析”。验证分割时看结果中的模型来源和运行状态，确认使用 LMDAUNet。
3. **复核结果。** 在图像对照中切换原图、分割叠加和热力图，再查看类别候选与量化特征。热力图表示模型关注区域。规则提示、七分类候选和分割稳定性各有独立含义。
4. **编辑并导出。** 查看报告草稿，编辑并预览，点击“保存修改”，再导出 Markdown 或 PDF。报告生成中断或失败时可重试。历史页可重新打开本账号的记录，查看已保存版本。旧记录缺少的原图或分析项会显示不可用。

PDF 默认使用 ReportLab。Windows 自动查找宋体，Linux 可安装 `fonts-wqy-zenhei`，也可用 `SKINVISION_PDF_FONT` 指定支持中文的字体文件。默认 SQLite 位于 `data/skinvision.db`，生成的分析图像位于 `outputs/service/`。升级已有实例前请备份数据库和图像目录，再运行同一迁移命令。原实例若使用 `data/temp_outputs` 或系统临时目录，应保留原 `SKINVISION_TEMP_DIR` 指向，以便查看旧任务的图像。

### 用公开图像试用

从 ISIC 2016 Task 1 的测试图像包中取 `ISBI2016_ISIC_Part1_Test_Data/ISIC_0000003.jpg`，上传这张图即可走完医生流程。要对照人工轮廓，在测试掩膜包中取 `ISBI2016_ISIC_Part1_Test_GroundTruth/ISIC_0000003_Segmentation.png`。来源为 International Skin Imaging Collaboration，Gutman 等，ISBI 2016，Task 1 以 CC0 发布。图像包约 232 MB，掩膜包约 2 MB。[官方数据与引用说明](https://challenge.isic-archive.com/data/) · [测试图像 ZIP](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_Data.zip) · [测试掩膜 ZIP](https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip)

## 数据准备、训练和评估

以下示例从官方 ISIC 2016 Task 1 图像及二值掩膜开始。下载清单与原始压缩包写入 `data/downloads/`，解压到 `data/research/`。准备脚本要求使用新的输出目录，写入图像、掩膜和含划分、输入哈希的清单。

```powershell
python -m pip install -r requirements/training.txt
python -m scripts.data_preparation.download_research_assets segmentation
python -m scripts.data_preparation.import_research_data segmentation
python -m scripts.data_preparation.prepare_isic2016 --root data/research/isic2016 --output data/prepared/isic2016 --size 256
python -m backend.skinvision.training.train --data-path data/prepared/isic2016 --work-dir outputs/training/isic2016 --device auto --epochs 200 --batch-size 8 --num-workers 0
```

`data_{train,val,test}.npy` 保存 NHWC RGB 图像，`mask_{train,val,test}.npy` 保存 NHW 二值掩膜（0/255）。同一套 CLI 还提供 `prepare_isic2017`、`prepare_isic2018` 和 `prepare_ph2`。使用 `--groups` 传入含 `image_id,patient_id,lesion_id` 的 CSV，可以按患者或病灶分组；未提供时仅能确认图像级划分。训练输出含验证损失最低的权重和用于恢复训练的 `latest.pth`，在同一工作目录再次运行可续训。CPU 上先用 `python -m scripts.tools.smoke_training --outdir outputs/train-smoke` 检查训练和评估链路，它以合成数据运行一个小批次。

```powershell
$trained = (Get-ChildItem outputs/training/isic2016/checkpoints/best-epoch*-loss*.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python -m backend.skinvision.training.test --device auto --data-path data/prepared/isic2016 --test-weights $trained --work-dir outputs/eval/isic2016 --num-workers 0
```

评估日志在 `outputs/eval/isic2016/log/`，包含 Dice、IoU 和混淆矩阵。要加载自训模型，把 `SKINVISION_WEIGHTS` 改为上一步 `$trained` 指向的文件，再重启后端并运行一条新分析。服务、评估与单图工具使用同一套严格的参数加载逻辑。

## 独立工具与实验

以下工具可在后端服务关闭时运行。示例使用前述公开样本和发布权重，若使用自训权重，将 `$weights` 改为对应文件。

```powershell
$image = 'data/research/isic2016/ISBI2016_ISIC_Part1_Test_Data/ISIC_0000003.jpg'
$mask = 'data/research/isic2016/ISBI2016_ISIC_Part1_Test_GroundTruth/ISIC_0000003_Segmentation.png'
$weights = 'weights/segmentation/best-epoch45-loss1.6684.pth'
python -m scripts.tools.run_preprocess_demo --input $image --outdir outputs/preprocess
python -m scripts.tools.run_metrics_demo --image $image --mask $mask
python -m scripts.tools.infer_single_image --image $image --weights $weights --normalization train_minmax_255 --outdir outputs/infer --cpu
```

预处理工具写出增强图、毛发掩膜、修复图和缩放图。量化工具输出面积占比、轮廓不规则度等 JSON 特征；单图推理保存掩膜、浮点概率数组和模型信息。安装 `requirements/dev.txt` 后还可导出和验证固定空间尺寸的 ONNX。下例校验 batch 1、32×32 的第四个分割头 logits，服务显示的是四头融合概率。

```powershell
python -m pip install -r requirements/dev.txt
python -m scripts.tools.export_onnx --weights $weights --output outputs/model32.onnx --height 32 --width 32 --cpu --verify
python -m pytest -q
```

在线接口回归可在独立测试数据库的服务上运行 `python test_api.py`。它将创建临时医生账号并测试任务的创建、访问隔离、导出与删除。默认连接 `http://localhost:8000`，也可用 `API_BASE_URL` 和 `TEST_IMAGE_PATH` 指定服务及测试图像。

实验模块使用 ISIC 2016 分割数据、ISIC 2018 Task 3 七分类数据和上文配置的权重。在已导入 2016 数据的基础上，以下命令下载 2018 数据、冻结实验样本，并依次运行归一化对照、全图与病灶区域分类对照、可靠性评估。协议与逐图结果均位于 `outputs/research-v1/`。数据许可、署名及使用条件见 [ISIC 官方数据页](https://challenge.isic-archive.com/data/)。

```powershell
python -m scripts.data_preparation.download_research_assets classification
python -m scripts.data_preparation.import_research_data classification
python -m scripts.experiments.prepare_protocol
python -m scripts.experiments.segmentation_study
python -m scripts.experiments.classification_study
python -m scripts.experiments.reliability_study
python -m scripts.experiments.summarize_results
```

配置在线模型后可另运行 `python -m scripts.experiments.report_study`，它以 8 个案例和 3 种条件发出 24 次报告请求。随后运行 `python -m scripts.experiments.audit_reports` 和 `python -m scripts.experiments.summarize_results`，查看技术性报告审计、指标汇总和图表。数据划分、逐例预测、模型运行状态与来源应随研究结果一起解释，小样本对照的数值仅适用于对应实验。
