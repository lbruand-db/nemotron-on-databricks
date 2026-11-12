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
- **No guaranteed GPU capacity** for either AI Runtime or serving scale-up. Use provisioned concurrency and no scale-to-zero for production Nano VL.
- **Less control over the serving container.** The `env_pack` failures and the unknown driver version are examples. The cookbook pins `torch==2.11.0+cu130` and builds the OCR extension for CUDA 13, which needs driver ≥ 580; Azure serving hosts are unverified. Several pieces are still in Preview.
- **Network egress.** The same as native Azure: Hugging Face, PyPI and `download.pytorch.org` must be mirrored for a locked-down bank network. Fix: pre-load models and datasets onto Volumes (SPEC OQ-13) and use a private PyPI mirror.
- **The OCR extension doesn't support A100 yet.** It's built for sm_75/86/89/90, not the A100 (sm_80). Fine on T4, but add `8.0` before serving OCR on `GPU_LARGE`.

## Recommendation

Fine-tune and run batch extraction on **Azure Databricks**. Serve the detectors and OCR on Databricks T4 endpoints, and Nano VL on the A100 pyfunc. Keep **AKS with reserved A100/H100** as the option if Nano VL needs hard latency or capacity SLAs that serverless can't promise. The merged weights are plain Hugging Face/vLLM checkpoints, so moving them later is cheap.

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
- NVIDIA NeMo Retriever extraction 26.08.1 release notes: https://docs.nvidia.com/nemo/retriever/26.8.1/extraction/releasenotes/
