# Azure-native vs Azure Databricks for the Nemotron cookbook — West Europe & North Europe

| | |
|---|---|
| Date | 2026-11-12 |
| Scope | Fine-tune and serve the five cookbook models (page-elements-v3, table-structure-v1, graphic-elements-v1, ocr-v1, nano-12b-v2-vl) in Azure `westeurope` and `northeurope` |
| Basis | Public Microsoft and NVIDIA documentation as of the date above (see Sources), plus what the cookbook showed on the AWS test workspace. Nothing here has been tested on Azure yet. |

**Short version:** Azure Databricks is the better fit for *fine-tuning plus governed serving*. Training runs on serverless GPUs in the EU with no quota tickets, and three recipes already work end to end. Azure-native wins if the bank needs *reserved, guaranteed GPU capacity* or wants NVIDIA's supported NIM containers unchanged. But on the native side, the off-the-shelf options for these specific models in these two regions are thin.

## Region facts (Nov 2026)

| | Azure-native | Azure Databricks |
|---|---|---|
| **Fine-tuning GPUs** | Azure ML compute clusters on A100 or H100 VMs (NC A100 v4, NCads H100 v5, ND H100 v5). Needs a per-subscription quota and the right VM types in the region. Public availability for West/North Europe **could not be confirmed**; it needs `az vm list-skus` on the bank's subscription. Microsoft says new NC capacity is going **only to H100 v5**. | AI Runtime **1×A10 / 1×H100 / 8×H100**, natively available in the **EU Data Boundary** (no cross-Geo opt-in), with no quota requests. It may use GPUs in another region at peak demand. |
| **Managed fine-tuning** | Foundry's serverless fine-tuning covers its own catalog models (Llama, Phi, Mistral…), **not these NVIDIA models** | Not needed: the recipes run as AI Runtime jobs |
| **Serverless GPU serving** | Azure Container Apps: **West Europe has T4 only, North Europe has none**. A100 serverless is only in Sweden Central and Italy North; North Europe has *dedicated* A100 profiles via a support request. | Classic GPU Model Serving in **both regions**: T4 16 GB, A100 80 GB (×1/2/4), with scale-to-zero |
| **Managed NVIDIA containers (NIMs)** | Foundry has **no first-party catalog entry for page-elements or OCR**. The Nano VL 8B NIM is being retired, and Nano 12B v2 VL isn't in the catalog. AKS can run any NIM, but you operate it. | No NIMs. The cookbook serves the open Hugging Face weights itself (pyfunc, or vLLM). |
| **OpenAI-style LLM serving** | Azure ML managed online endpoint with your own vLLM container: works with custom weights, needs a GPU quota in the region | Custom LLM Serving is **US-only per the docs**. In West/North Europe, Nano VL goes through the classic pyfunc path (`VLLMChatModel`, A100 80 GB). |

## Azure-native

**Pros**
- **Reserved capacity.** You can buy reserved instances or capacity reservations for specific GPU VMs. In capacity-tight West Europe, that's a real advantage. Databricks serverless doesn't guarantee capacity (the AWS test workspace waited 74 min for an L40S).
- **Full control of the stack.** You pick the drivers, CUDA version and container images. That avoids opaque platform failures such as the `env_pack` deployments (SPEC OQ-5/OQ-17) or an unknown serving-host driver.
- **NVIDIA-supported NIMs on AKS**, if the bank wants NVIDIA's own containers and support (NVIDIA AI Enterprise licensing for production).
- **Native Azure security and ops tooling** (Entra ID, Defender, Azure Monitor), which the bank's platform team likely already runs.

**Cons**
- **No managed path for these models in these regions.** You'd assemble it yourself: Azure ML or AKS clusters, your own Docker images (including compiling the OCR CUDA extension and `causal-conv1d`), job orchestration, evaluation, a model registry, and endpoint deployment. That is what this cookbook already gives you on Databricks.
- **Quota and SKU uncertainty in West/North Europe.** A100/H100 VM availability and quota must be requested per subscription.
- **No serverless option for Nano VL** in either region (Container Apps has no A100 there). So it's a dedicated GPU VM that is paid for even when idle, or AKS.
- **NIMs don't fit a fine-tuning story.** Catalog NIMs serve NVIDIA's weights. Serving *fine-tuned* detectors/OCR would mean custom containers anyway; whether NIM accepts custom weights for these extraction models still needs verifying. Also, NVIDIA's latest extraction stack has moved on (`nemotron-object-detection` 2.0.1, `nemotron-ocr-v2` in NeMo Retriever 26.08.1), which matters if the bank wants NIM parity.

## Azure Databricks

**Pros**
- **It works today (on AWS).** page-elements and OCR are trained, registered and served. Nano VL is trained (CORD test field F1 0.39 → 0.86), with serving in progress, all on the stock `databricks_ai_v6` environment with no custom images.
- **One governed lifecycle.** Unity Catalog links each training run to a model version and an endpoint. Documents in Volumes or Delta feed training directly. Batch extraction over the document estate can run next to the data.
- **Serverless economics.** No idle GPUs, scale-to-zero on serving, no quota tickets. Azure Databricks is a first-party Azure service, so the spend typically counts toward an existing Microsoft commitment (worth confirming for the bank).
- **EU residency for training** without enabling cross-Geo processing.

**Cons and risks**
- **Custom LLM Serving isn't in West/North Europe**, so Nano VL relies on the classic pyfunc path, with its own request format rather than the OpenAI API (no AI Playground, no vLLM metrics tab).
- **Nano VL needs the A100 there.** On AWS (2026-11-13), the cookbook serves Nano VL in FP8 on one A10G 24 GB (`GPU_MEDIUM`; field F1 0.847 vs 0.858 in bf16), because larger GPUs had no capacity. Azure serving has no A10G tier, and FP8 weights (14.5 GiB) don't fit a T4 16 GB with room for the KV cache. So in West/North Europe it is `GPU_LARGE` (A100 80 GB), in bf16 or FP8.
- **No guaranteed GPU capacity** for either AI Runtime or serving scale-up. Use provisioned concurrency and no scale-to-zero for production Nano VL.
- **Less control over the serving container.** The `env_pack` failures and the unknown driver version are examples. The cookbook pins `torch==2.11.0+cu130` and builds the OCR extension for CUDA 13, which needs driver ≥ 580; Azure serving hosts are unverified. Several pieces are still in Preview.
- **Network egress.** The same as native Azure: Hugging Face, PyPI and `download.pytorch.org` must be mirrored for a locked-down bank network. Fix: pre-load models and datasets onto Volumes (SPEC OQ-13) and use a private PyPI mirror.
- **The OCR extension doesn't support A100 yet.** It's built for sm_75/86/89/90, not the A100 (sm_80). Fine on T4, but add `8.0` before serving OCR on `GPU_LARGE`.

## Complexity (code and operations)

### What the cookbook is made of today (measured in this repo)

| Part | Size | Platform-specific? |
|---|---|---|
| ML core: YOLOX loss/trainer, detection data and metrics, OCR recognizer training, VLM LoRA/merge/eval/compat patches (`yolox_train`, `detection`, `ocr`, `vlm`) | ~2,200 lines of Python | **No.** Plain PyTorch, Hugging Face, PEFT, vLLM, MLflow. Ports to either platform as is. |
| Recipe entry points (`recipes/*/train.py`) | ~350 lines | Lightly: they read AIR parameters and write to a Volume |
| Databricks layer: register/deploy/query (`lifecycle`), parameter, path and MLflow-run helpers, `recipes/*/register.py` | ~750 lines | **Yes.** Unity Catalog registration, `MLFLOW_RUN_ID`, Model Serving SDK |
| AIR workload definitions (`workload.yaml`, `register.yaml`) | ~200 lines of YAML | Yes |
| Dockerfiles, infrastructure-as-code, cluster/node-pool config | **none** | Not needed: stock `databricks_ai_v6`, serverless GPUs, managed serving |
| Tests (CPU) | ~480 lines | No |

The hard parts (porting the YOLOX loss, OCR per-position CE, the 13 Nano VL compatibility patches, the shard merge, the vLLM eval) are **the same work on either platform**. The platforms differ in everything around that core.

### Azure Databricks

- **What you write:** the ML code, one YAML per stage, and a ~30-line register script per recipe. Unity Catalog provides the model registry, lineage (training run → model version → endpoint) and permissions. Serving is one SDK call (`nemotron_dbx.lifecycle.deploy`).
- **What you operate:** almost nothing. There are no clusters, images, drivers or node pools to maintain. CUDA extensions are compiled once inside the stock environment and cached on a Volume (`common/wheels.py`).
- **Where the complexity actually showed up (AWS test workspace):** opaque platform failures you can't debug from outside (`env_pack` deployments, SPEC OQ-5/OQ-17). GPU capacity waits that were invisible from the CLI (OQ-16). CLI quirks (OQ-15, OQ-18). Little control over the serving image (driver version, base libraries).

### Azure-native (Azure ML / AKS)

- **What you'd keep:** the ML core and the MLflow logging (Azure ML supports MLflow tracking and its model registry).
- **What you'd rewrite:** the ~750-line Databricks layer and the YAML, as Azure ML command jobs, Azure ML registry calls and online-endpoint deployments. The amount of code is comparable.
- **What you'd add, which Databricks doesn't need:**
  1. **Container images.** A training image with CUDA 13, torch, the OCR extension, `causal-conv1d` and `mamba-ssm`; one or more serving images (vLLM, the OCR runtime). Plus an Azure Container Registry, and rebuilds on every dependency bump.
  2. **Compute.** An Azure ML workspace, a compute cluster per GPU type, per-region quota requests for A10/A100/H100. Or, on AKS, GPU node pools with the NVIDIA GPU operator and driver management.
  3. **Serving plumbing.** Scoring scripts or a vLLM container with liveness/readiness probes, autoscale rules (Azure Monitor), and blue/green traffic shifts.
  4. **Networking and identity.** VNet, private endpoints for storage, registry and endpoints, managed identities and registry pull permissions.
  5. **Infrastructure-as-code** (Bicep/Terraform) for all of the above, plus CI to build and scan images.
  6. **Governance glue.** Lineage from documents to model to endpoint isn't automatic; it means Purview or custom tags.
- **The upside of that work:** full control. You pick the driver/CUDA version, inspect container logs and reserve capacity, so the failure modes above become debuggable.
- **Rough size (my estimate, not measured):** a few engineer-weeks to reach parity with this cookbook, plus ongoing image and infrastructure maintenance. On Databricks, that part was zero.

## Cost

List prices in USD, pay-as-you-go, **West Europe** unless noted, checked 2026-11-12. These are estimates; confirm with the Azure pricing calculator and the account's negotiated rates.

### Unit prices

| GPU | Azure Databricks | Azure-native VM (compute only) |
|---|---|---|
| **Training** | AI Runtime: **A10 $4.90 / GPU-h**, **H100 $7.00 / GPU-h** (Microsoft's published US East reference; West Europe not confirmed). The VM is included. | A10: `Standard_NV36ads_A10_v5` **$4.16/h**. H100: `Standard_NC40ads_H100_v5` (H100 NVL 94 GB) **$9.08/h**. Azure ML adds no surcharge. |
| **Serving T4** | `GPU_SMALL`: 10.48 DBU/h × $0.084/DBU = **$0.88/h** per instance | `Standard_NC4as_T4_v3` **$0.66/h** |
| **Serving A100 80 GB** | `GPU_LARGE`: 78.60 DBU/h × $0.084/DBU = **$6.60/h** per instance | `Standard_NC24ads_A100_v4` **$4.78/h** |

On top of that: Databricks needs a Premium workspace. Azure-native adds container registry, storage, Log Analytics, networking and the engineering time above. Both can be discounted (Databricks commit; Azure reservations/savings plans), and Azure Databricks spend typically counts toward a Microsoft commitment.

### Training: a few dollars per run on either platform

Billed times are the measured run durations from this cookbook.

| Run | Measured duration | Azure Databricks | Azure-native |
|---|---|---|---|
| page-elements-v3 fine-tune | 22 min, 1×A10 | ~$1.80 | ~$1.50, plus cluster spin-up |
| ocr-v1 fine-tune | 8.6 min, 1×A10 | ~$0.70 | ~$0.60, plus spin-up |
| nano-12b-v2-vl LoRA + evals + merge | 31 min, 1×H100 | ~$3.60 | ~$4.70, plus spin-up |
| nano-12b-v2-vl registration (vLLM smoke test + packaging) | 15–48 min, 1×H100 | ~$1.75–5.60 | similar |

The whole M0–M4 development effort (about 25 GPU jobs) cost **tens of dollars of GPU time** either way. **Training cost doesn't decide this; serving does.**

### Serving: the cost that matters

Monthly cost per endpoint with one instance. "Business hours" means ~220 h/month (10 h × 22 days), which only Databricks scale-to-zero achieves. Azure ML managed online endpoints can't scale to zero, and Azure Container Apps' serverless T4 exists in West Europe only (no A100, nothing in North Europe).

| Endpoint | Databricks, 24/7 | Databricks, business hours | Azure-native VM, 24/7 |
|---|---|---|---|
| T4 detector or OCR (×4 in the full cookbook) | ~$640 | ~$195 | ~$480 |
| A100 Nano VL | ~$4,820 | ~$1,450 | ~$3,490 |
| **Full cookbook (4 × T4 + 1 × A100)** | **~$7,390** | **~$2,230** | **~$5,410** |

- **Break-even:** per GPU-hour, Azure-native list compute is **~25–28 % cheaper** (T4 $0.66 vs $0.88; A100 $4.78 vs $6.60). Databricks bills only active time and includes the managed platform. So **an endpoint up less than ~72–75 % of the time is cheaper on Databricks**; one that must be up 24/7 is cheaper as a raw VM, before engineering and operations costs.
- **Cold starts limit scale-to-zero for Nano VL.** Loading vLLM and the 25 GB model takes minutes (the pyfunc smoke test took 217 s including vLLM start on H100). Scale-to-zero suits dev, demos and batch, not an interactive production SLA. So production Nano VL is likely always-on, where Azure-native list compute is cheaper.
- **Batch is the cheapest pattern on both.** For a document backlog, running extraction as a batch job (AI Runtime, or an Azure ML cluster that scales to zero) instead of keeping endpoints up costs only the job's GPU time. At the measured ~4.7 receipts/s for Nano VL on one H100, 100k pages is ~6 GPU-hours: ~$40 on AI Runtime.

## Recommendation

Fine-tune and run batch extraction on **Azure Databricks**. Serve the detectors and OCR on Databricks T4 endpoints, and Nano VL on the A100 pyfunc. Keep **AKS with reserved A100/H100** as the option if Nano VL needs hard latency or capacity SLAs that serverless can't promise. Cost points the same way: an always-on A100 endpoint is ~$1.3k/month cheaper as a raw VM at list price, which has to be weighed against the engineering and operations work in the Complexity section. The merged weights are plain Hugging Face/vLLM checkpoints, so moving them later is cheap. For bulk document processing, prefer batch jobs over always-on endpoints on either platform.

**Before committing, check:**
1. `az vm list-skus --all` and GPU quotas on the bank's subscription in `westeurope` and `northeurope`.
2. With the account team: AI Runtime GPU placement in the EU (and whether it crosses regions), A100/T4 serving capacity per region, and the roadmap for Custom LLM Serving in Europe.
3. A validation run of the existing recipes on an Azure Databricks workspace in one of the two regions (an Azure FEVM workspace). It should cover the driver version, the pip-built serving containers and the Nano VL A100 pyfunc.

## Sources

- Azure Databricks Designated Services (AI Runtime availability by Geo): https://learn.microsoft.com/en-us/azure/databricks/resources/designated-services
- Azure Databricks features with limited regional availability (serverless, Model Serving CPU/GPU): https://learn.microsoft.com/en-us/azure/databricks/resources/feature-region-support
- Azure Databricks AI Runtime overview (accelerators, cross-region GPUs, limitations): https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/overview
- Azure Databricks Custom LLM Serving (GPU types, regions): https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/serve-custom-llms
- Azure Databricks custom model serving GPU workload types: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints
- Azure NC A100 v4 series (capacity note): https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series
- Azure Container Apps serverless GPUs (regions): https://learn.microsoft.com/en-us/azure/container-apps/gpu-serverless-overview
- Azure AI Foundry managed compute / NVIDIA NIM: https://learn.microsoft.com/en-us/azure/foundry-classic/how-to/deploy-models-managed-pay-go, https://learn.microsoft.com/en-us/azure/foundry/how-to/deploy-models-managed
- Azure Databricks AI Runtime pricing guide (A10/H100 per GPU-hour, US East reference): https://learn.microsoft.com/fil-ph/azure/Databricks/machine-learning/ai-runtime/guides/
- Azure Databricks Model Serving DBU rates (GPU_SMALL 10.48, GPU_LARGE 78.60 DBU/h): https://learn.microsoft.com/azure/databricks/resources/pricing
- West Europe Serverless Real-Time Inference DBU price ($0.084), from the Azure Retail Prices API via https://www.dbuwatch.com/regions/azure/westeurope/
- West Europe Linux VM pay-as-you-go prices, from https://instances.usage.ai/azure/vm/region/westeurope (verify with https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)
- Azure ML online endpoints (no surcharge, no scale-to-zero): https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online
- NVIDIA NeMo Retriever extraction 26.08.1 release notes: https://docs.nvidia.com/nemo/retriever/26.8.1/extraction/releasenotes/
