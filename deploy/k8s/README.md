# Minimal Kubernetes deployment

Use the same manifests on an existing EKS, AKS or on-prem Kubernetes cluster. This deploys the FastAPI app and its lifecycle-managed Studio worker together, one Python process per pod. HPA keeps 2–4 pods; Service distributes HTTP requests without sticky sessions. PostgreSQL is shared action authority, Chroma is the shared approved corpus, and Studio coordinates workflows. Request guards stay local to each activity.

This is a deployment starter for the synthetic demo. Cluster rollout, multi-worker Studio recovery, container runtime compatibility and throughput remain unverified until run in your environment.

## Prerequisites

- An existing cluster, `kubectl` context, namespace write access and Metrics Server for CPU HPA.
- A registry the cluster can pull from. Configure its node/workload pull permissions or `imagePullSecrets` on the Deployment.
- Reachable private PostgreSQL with a dedicated action database and schema permissions; private Chroma 1.5.5 with persistent storage; private Ollama with `ministral-3:14b` and `nomic-embed-text:latest` already loaded. These dependencies are not installed by this script. Do not point pods at your laptop's loopback addresses. The sample DNS names are placeholders for existing services.
- A dedicated Studio development key and permitted outbound access for its API and discovered workflow/Temporal endpoints. Studio is an external control plane; an air-gapped on-prem cluster cannot run this mode. Inference remains on your private infrastructure.

## 1. Build and publish

From the repository root, choose your registry image tag or digest:

```sh
docker buildx build --platform linux/amd64 -t REGISTRY/hdfc-support:demo-1 --push .
```

Alternatively, `./deploy/k8s/build.sh REGISTRY/hdfc-support:demo-1 linux/amd64` builds and loads the image locally; publish it afterwards with `docker push REGISTRY/hdfc-support:demo-1`.

Use `linux/arm64` for ARM nodes or a multi-platform build if both node types are used. EKS can pull from ECR, AKS from ACR, and on-prem from your private registry. The image contains the locked Python dependencies, application and approved synthetic documents. Secret files and local artifacts are excluded. The editable package installation preserves the existing `/app` document-root lookup; the container runs as UID 10001 with a read-only root filesystem in Kubernetes.

## 2. Configure and deploy

```sh
cp deploy/k8s/config.env.example deploy/k8s/config.env
cp deploy/k8s/secrets.env.example deploy/k8s/secrets.env
chmod 600 deploy/k8s/secrets.env
# Edit both local files: private service addresses, dedicated Studio key,
# PostgreSQL connection, distinct demo bearer tokens and reviewer webhook.
./deploy/k8s/deploy.sh YOUR_CONTEXT REGISTRY/hdfc-support:demo-1 \
  deploy/k8s/config.env deploy/k8s/secrets.env
```

Optional fifth argument changes the namespace from `hdfc-support`. The script always selects the supplied context explicitly. It creates a ConfigMap and Secret from these files, sets the image on the API and ledger initializer, applies the Deployment/ClusterIP Service/HPA/PDB and waits for rollout. It does not provision a cloud cluster or change a public load balancer. Actual `*.env` files are ignored by Git. Never put the OCR key in these files or image. Prefer your platform's secret manager for a production deployment; native Kubernetes Secrets must be encrypted and access-controlled by cluster operators.

`INFERENCE_ALLOWED_HOSTS` permits only exact operator-configured private inference hosts. Its default is empty for local development. Set both inference origins and the matching allowlist; URLs must have no `/v1`, path, credentials or query. Use HTTPS where your private inference gateway supports TLS. Configure VPC/firewall/CNI egress to the actual private dependencies and Studio destinations; an application hostname allowlist is not a network firewall. The manifests deliberately leave provider-specific network policy and ingress configuration to your existing platform.

Every pod's init container calls the idempotent PostgreSQL schema initializer before API startup. Its existing advisory lock serializes concurrent initializers; it does not reset data. For production, replace this demo startup initializer with your approved migration process and reduce the runtime role's DDL permissions. Corpus ingestion stays explicit:

```sh
kubectl --context YOUR_CONTEXT -n hdfc-support exec deployment/hdfc-support -- \
  hdfc-mistral-usecase ingest
```

Run ingestion once after deploying against a new Chroma collection. Repeating it keeps stable chunks. This uses reviewed synthetic fixtures, with no hosted OCR call. Do not run concurrent corpus updates during a demonstration.

## 3. Demonstrate privately

```sh
kubectl --context YOUR_CONTEXT -n hdfc-support port-forward service/hdfc-support 8021:80
curl -sS http://127.0.0.1:8021/health/ready
curl -sS http://127.0.0.1:8021/v1/customer-support \
  -H 'Content-Type: application/json' \
  -d '{"question":"Is Riverside open on Saturdays?"}'
curl -N 'http://127.0.0.1:8021/v1/customer-support?stream=true' \
  -H 'Content-Type: application/json' \
  -d '{"question":"Is Riverside open on Saturdays?"}'
kubectl --context YOUR_CONTEXT -n hdfc-support get pods,hpa,pdb
```

`/health/live` checks that the event loop responds. `/health/ready` reports ready while the application accepts requests and returns 503 during shutdown/draining. It also reports `workflow_worker_running`, a local task indicator that does not prove remote registration. A Studio outage leaves the API ready so urgent support and the independent FAQ route stay reachable; support workflow calls may return safe 503. Neither probe performs model/database calls or certifies dependencies. Liveness does not restart pods simply because a dependency is down.

## Cloud and on-prem differences

| Environment | Supply through the existing platform |
| --- | --- |
| EKS / private VPC | ECR pull permissions, private RDS/PostgreSQL and model/data endpoints, internal ALB/NLB or existing ingress with TLS |
| AKS / private VNet | ACR pull permissions, private PostgreSQL and model/data endpoints, internal Azure load balancer or existing ingress with TLS |
| On-prem | Registry credentials, private PostgreSQL/model/data services, existing ingress or MetalLB if an internal load balancer is needed |

The app Service is `ClusterIP` by default, so the same base does not create a public endpoint in any environment. Route your existing internal ingress to service `hdfc-support:80`, preserve `POST` SSE streaming, disable response buffering and set its idle timeout above 60 seconds. Set `DEMO_REVIEW_CALLBACK_BASE_URL` to the real TLS API origin reachable by the reviewer application; its authenticated callback must reach this same Service. Port forwarding alone is not a reviewer callback address.

## Scaling and rollout limits

Resource requests/limits are starter values, not measured sizing. HPA needs Metrics Server and CPU capacity; CPU scaling will not necessarily detect a queue waiting on inference. Model/GPU capacity, Chroma and PostgreSQL must be sized independently. The per-request six-generation/twenty-second limits remain unchanged. HPA does not establish a latency or concurrency guarantee.

Each replica starts one worker using the same dedicated Studio workspace/configuration. Rolling updates permit an extra pod and keep current ready pods; PDB keeps at least one pod during voluntary disruption when capacity permits. A 90-second termination grace period allows HTTP draining and SDK cleanup. Studio-managed human waits and the PostgreSQL ledger must recover across worker replacement; live multi-pod workflow recovery, callback replay and scale-down tests remain a deployment acceptance gate. No real banking writes are enabled.

Configuration updates rerun rollout via `deploy.sh`. No cluster or dependency data is deleted by that script. Inspect failures with `kubectl --context YOUR_CONTEXT -n hdfc-support describe pod POD` and API/init-container logs; do not dump Secrets into troubleshooting output.

References: [Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/), [probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/), [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/), [Mistral workflow worker](https://docs.mistral.ai/studio/workflows/quickstarts/your-first-workflow).
