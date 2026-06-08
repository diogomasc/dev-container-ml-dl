# 🚀 Dev Container — ML & DL Sandbox (GPU Enabled)

> Container de desenvolvimento seguro, isolado e acelerado por GPU (NVIDIA CUDA) para **Python (Deep Learning)**, **Node.js**, e **Jupyter Notebook**.
> Projetado para uso com **VS Code Remote Containers**, mantendo o seu sistema host (Ubuntu) 100% limpo e padronizado.

---

## 🧠 Entendendo a Arquitetura (Host vs. Container)

O maior diferencial deste ambiente é a separação rigorosa de responsabilidades. Para garantir reprodutibilidade total, o código de Machine Learning **ignora** as bibliotecas do seu computador físico.

A arquitetura funciona em 3 camadas:
1. **O Host (Sua Máquina):** Roda apenas o Ubuntu e o **Driver da NVIDIA**. Ele atua puramente como a "ponte" para o hardware físico.
2. **O Middleware (NVIDIA Container Toolkit):** Pega o Driver da NVIDIA e o injeta com segurança dentro do Docker.
3. **O Sandbox (Dev Container):** Roda a imagem oficial `nvidia/cuda` e instala o PyTorch. O PyTorch já traz embutido o seu próprio CUDA Runtime e cuDNN. Isso significa que o modelo treina de forma idêntica em qualquer máquina, independentemente de qual versão do CUDA Toolkit esteja instalada no sistema operacional anfitrião.

---

## 📋 Passo a Passo: Preparação do Host (Ubuntu 24.04)

Antes de iniciar o container, a sua máquina física precisa estar preparada com os drivers e ferramentas de comunicação.

### 1. Instalar o Driver NVIDIA
Utilize a ferramenta oficial do Ubuntu para detectar e instalar o driver proprietário adequado (recomendado `595` ou superior).

```bash
# Instalar a ferramenta de detecção
sudo apt install -y ubuntu-drivers-common

# Listar drivers disponíveis para a placa de vídeo
ubuntu-drivers devices

# Instalar a versão proprietária compatível (Exemplo: 595)
sudo apt install -y nvidia-driver-595

# REINICIE O SISTEMA para carregar o novo driver
sudo reboot
```

### 2. Instalar o CUDA Toolkit 13.2 (Ambiente Host)

*Opcional para o Docker, mas essencial para desenvolvimento bare-metal local.*
Siga as [instruções oficiais da NVIDIA para Ubuntu 24.04](https://developer.nvidia.com/cuda-13-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local):

```bash
wget [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin)
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget [https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda-repo-ubuntu2404-13-2-local_13.2.0-595.45.04-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda-repo-ubuntu2404-13-2-local_13.2.0-595.45.04-1_amd64.deb)
sudo dpkg -i cuda-repo-ubuntu2404-13-2-local_13.2.0-595.45.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2

# Instalar drivers em modo proprietário (foco em estabilidade)
sudo apt-get install -y cuda-drivers
```

> **Nota sobre Kernel Modules:** O comando acima prioriza a estabilidade instalando os módulos proprietários. Caso necessite trocar para a versão open-source no futuro, basta executar `sudo apt-get install -y nvidia-open`.

### 3. Expor o CUDA no Host (Variáveis de Ambiente)

Adicione as seguintes linhas ao final do seu ficheiro `~/.zshrc` ou `~/.bashrc` no host para garantir que o sistema encontre os binários:

```bash
# ── CUDA & NVIDIA ──────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
```

### 4. NVIDIA Container Toolkit (A Ponte do Docker)

Esta é a ferramenta que permite ao Docker enxergar a sua RTX 3060:

```bash
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 📂 Estrutura do Projeto (Dev Container)

```text
dev-container-ml-dp/
├── docker-compose.yml        # Orquestração, limites de RAM/Shm e injeção da GPU
├── Dockerfile                # Receita baseada no CUDA 13.2-devel + Tools (uv, fnm)
├── entrypoint.sh             # Script de inicialização (HMR, permissões, socket)
├── .devcontainer/
│   └── devcontainer.json     # Integração direta com o VS Code
└── workspace/                # 📍 SEUS PROJETOS FICAM AQUI (Sincronizado)
```

## 🔐 Configuração obrigatória antes de rodar

Antes do primeiro build, defina a identidade do Git que será aplicada dentro do container.

1. Copie o arquivo de exemplo para o arquivo local:

	cp .env.example .env

2. Edite o arquivo `.env` e ajuste os campos:

	GIT_USER_NAME=Seu Nome
	GIT_USER_EMAIL=seu.email@exemplo.com

3. Suba o ambiente com rebuild para aplicar no image build:

	docker compose up -d --build

---

## ⚡ Instalação e Uso via VS Code

Este ambiente foi desenhado para ser "Plug & Play".

1. Abra o **VS Code**.
2. Arraste a pasta `dev-container-ml-dp` para dentro do editor.
3. Um pop-up verde aparecerá no canto inferior direito. Clique em **Reopen in Container** (ou pressione `F1` > digite *Dev Containers: Reopen in Container*).
4. Aguarde o build.

Quando o terminal integrado abrir, você estará dentro de um ambiente Zsh com acesso root virtual (`devuser`), Python 3.12 gerenciado via `uv`, e a sua GPU pronta para uso pesado.
---

## 🐳 Gerenciamento Manual (Docker Compose)

Caso necessite reiniciar o ambiente pelo terminal do seu computador (Host), rode os comandos dentro da pasta `dev-container-ml-dp`:

```bash
# Subir o ambiente em background
docker compose up -d

# Verificar se está ativo
docker compose ps

# Reconstruir do zero (após alterar o Dockerfile)
docker compose build --no-cache && docker compose up -d
```

---

## 📓 Fluxo de Trabalho AI / Machine Learning

O ambiente foi tunado para evitar gargalos de memória em DataLoaders pesados:

* **Shm Size:** `8gb` (Evita erros de *bus error* no PyTorch).
* **Tmpfs:** `512MB` em `/scratch` para checkpoints rápidos.

### Iniciando um Projeto PyTorch Isolado

Sempre crie um ambiente virtual (`venv`) para cada novo projeto dentro da pasta `/workspace`. O `uv` fará isso em milissegundos:

```bash
cd /workspace
uv venv meu_projeto_ai
source meu_projeto_ai/bin/activate

# Instalar PyTorch (Usando wheel otimizada)
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# Verificar a aceleração de hardware
torch-check
```

### Serviços Nativos (JupyterLab e TensorBoard)

O container já exporta as portas essenciais para o seu navegador.

* **JupyterLab (Porta 8888):** Rode o alias `jlab` no terminal e acesse `http://localhost:8888`.
* **TensorBoard (Porta 6006):** Rode `tensorboard --logdir=runs/ --bind_all` e acesse `http://localhost:6006`.
---

## 🛠️ Comandos Úteis do Ambiente Interno

| Comando / Alias | Ação |
| --- | --- |
| `gpu` | Exibe o status em tempo real da RTX 3060 (`nvidia-smi`) |
| `torch-check` | Valida a versão do PyTorch e a disponibilidade da GPU |
| `jlab` | Inicia o servidor do JupyterLab interativo |
| `uv pip install X` | Instala pacotes Python instantaneamente via Rust |
| `fnm use 22` | Alterna o Node.js para a versão LTS |
| `docker ps` | Lista e gerencia outros containers rodando no host (Ex: LocalStack) |

## 🩺 Verificação de Saúde e Comunicação (Host ⇄ Container)

Para garantir que o isolamento foi feito corretamente e que o container consegue se comunicar com o hardware e os serviços do host, execute a bateria de testes abaixo dentro do terminal do Dev Container.

### 1. Validação do Hardware (Ponte de GPU)
Verifica se o container está herdando o Driver do host e enxergando a placa de vídeo física.
```bash
gpu
```
*Ou execute o comando nativo:* `nvidia-smi`


**Resultado esperado:** A tabela oficial da NVIDIA deve ser impressa na tela, mostrando o modelo `NVIDIA GeForce RTX 3060`, o driver `595.71.05` e a versão do CUDA `13.2`.

### 2. Validação do Ecossistema de Containers (Docker-in-Docker)

Verifica se o socket do Docker foi montado corretamente, permitindo gerenciar outros containers (como o LocalStack da infraestrutura satélite) a partir de dentro do ambiente de desenvolvimento.

```bash
docker ps
```

* **Resultado esperado:** A listagem dos containers ativos no seu computador host. O container `dev-container-ml` deve aparecer na lista.

### 3. Validação dos Gerenciadores de Runtime (Foco em Performance)

```bash
uv --version
fnm --version
node -v
```

* **Resultado esperado:**
* Versão estável do gerenciador `uv`.
* Versão ativa do gerenciador `fnm`.
* Versão do Node.js `v22.x.x` (LTS carregado dinamicamente).

### 4. O Teste de Fogo (Cálculo Numérico em GPU)

O teste definitivo. Inicializa o interpretador do Python 3.12 mapeado no `/opt/venv`, carrega a biblioteca do PyTorch compilada com suporte a hardware e faz uma chamada direta aos núcleos tensores da placa.

```bash
torch-check

```

*Ou execute o comando nativo:

```
python3 -c "import torch; print(f'GPU Disponível: {torch.cuda.is_available()}'); print(f'Placa: {torch.cuda.get_device_name(0)}')"
```

* **Resultado esperado:**

```text
PyTorch 2.11.0 | CUDA disponível: True | GPU: NVIDIA GeForce RTX 3060
```

Se todos os quatro blocos responderem conforme o esperado, o ambiente está saudável, estável e pronto para a execução de pipelines de Deep Learning distribuídos e reprodutíveis.
