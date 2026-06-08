# ══════════════════════════════════════════════════════════════
# Dev Container — Ambiente de Desenvolvimento Sandboxed para ML/DL
# GPU: NVIDIA RTX 3060 (via nvidia/cuda 13.2 base)
# Stack: Node 22 LTS (fnm) + Python 3.12 (uv) + JupyterLab + Docker CLI
# Shell: Zsh + Starship + plugins (autocomplete, syntax-highlight)
# ══════════════════════════════════════════════════════════════

# Usa a imagem oficial da NVIDIA com CUDA 13.2, compilador nvcc e cuDNN embutidos.
# Baseada no Ubuntu 24.04 para garantir o Python 3.12 nativo.
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

# ── Metadados (Padrão OCI) ──────────────────────────────────
LABEL org.opencontainers.image.title="dev-container-ml"
LABEL org.opencontainers.image.description="Container de dev completo com GPU: CUDA 13.2 + Node 22 LTS + Python 3.12 + JupyterLab + Zsh"
LABEL org.opencontainers.image.authors="Diogo Mascarenhas <diogomascarenhas0574@gmail.com>"

# ── Configurações de Ambiente ────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── Pacotes do Sistema (Camada única e mínima) ────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    wget \
    git \
    sudo \
    unzip \
    gzip \
    tar \
    zsh \
    bash \
    gnupg \
    lsb-release \
    software-properties-common \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    make \
    cmake \
    llvm \
    zlib1g-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libbz2-dev \
    libncurses-dev \
    libopenblas-dev \
    gfortran \
    btop \
    direnv \
    procps \
    openssh-client \
    tree \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ── Docker CLI (Montagem via socket para o host) ──────────────
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ── Usuário Comum (Non-root) ──────────────────────────────────
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
ARG GIT_USER_NAME="Dev User"
ARG GIT_USER_EMAIL="devuser@example.com"

RUN userdel -f -r ubuntu 2>/dev/null || true \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/zsh ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && groupadd -f docker \
    && usermod -aG docker ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# ── fnm + Node 22 LTS ────────────────────────────────────────
RUN curl -fsSL https://fnm.vercel.app/install | bash \
    && export PATH="$HOME/.local/share/fnm:$PATH" \
    && eval "$(fnm env)" \
    && fnm install 22 \
    && fnm default 22 \
    && eval "$(fnm env)" \
    && npm config set fund false \
    && npm config set audit false \
    && npm cache clean --force

# ── uv ───────────────────────────────────────────────────────
# Instala o uv — gerenciador de pacotes Python ultra-rápido (Rust)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /tmp/* ~/.cache/uv

# ── Venv global em /opt/venv ─────────────────────────────────
# Ubuntu 24.04 implementa PEP 668: bloqueia instalações --system no
# Python gerenciado pelo APT. A solução correta é um venv dedicado
# em /opt/venv — acessível globalmente, sem precisar de `source activate`.
# O PATH é configurado logo abaixo para que todos os binários
# (jupyter, python, etc.) sejam encontrados automaticamente.
USER root
RUN python3 -m venv /opt/venv \
    && chown -R devuser:devuser /opt/venv
USER devuser

# ── Stack ML/DL via uv + /opt/venv ──────────────────────────
# NOTA SOBRE O ÍNDICE CUDA:
# O PyTorch não publica wheels para cu132. O índice cu126 (CUDA 12.6)
# é o mais recente com suporte completo a Python 3.12 no Linux.
# Isso não é um problema: drivers NVIDIA são retrocompatíveis —
# o driver 595.71.05 (CUDA 13.2) roda sem fricção qualquer toolkit ≤ 13.2.
# Versões pinadas garantem reprodutibilidade entre membros da equipe.
ENV VIRTUAL_ENV=/opt/venv
RUN export PATH="$HOME/.local/bin:$PATH" \
    && uv pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.11.0 \
    && uv pip install \
    torch-geometric \
    && uv pip install \
    jupyterlab \
    ipywidgets \
    jupyterlab-widgets \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    tqdm \
    && rm -rf /tmp/* ~/.cache/uv

# ── Poetry (Gerenciador de dependências Python) ──────────────
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && export PATH="$HOME/.local/bin:$PATH" \
    && poetry config virtualenvs.in-project true \
    && rm -rf /tmp/*

# ── Prompt Starship ──────────────────────────────────────────
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y

# ── Plugins Zsh ──────────────────────────────────────────────
RUN mkdir -p ~/.zsh/plugins \
    && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions.git ~/.zsh/plugins/zsh-autosuggestions \
    && git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.zsh/plugins/zsh-syntax-highlighting \
    && git clone --depth=1 https://github.com/zsh-users/zsh-completions.git ~/.zsh/plugins/zsh-completions \
    && git clone --depth=1 https://github.com/zsh-users/zsh-history-substring-search.git ~/.zsh/plugins/zsh-history-substring-search \
    && rm -rf ~/.zsh/plugins/*/.git

# ── Configuração do Starship (Alinhado com o setup do Diogo no host) ──
RUN mkdir -p ~/.config && cat <<'EOF' > ~/.config/starship.toml
format = """
\\[ [$time](bold white) \\] $username$directory$git_branch$git_status$nodejs$python$docker_context$line_break$character
"""

add_newline = true

[character]
success_symbol = "[❯](bold green)"
error_symbol   = "[❯](bold red)"

[time]
disabled    = false
format      = "$time"
time_format = "%H:%M"
style       = "bold white"

[username]
show_always = true
style_user  = "bold yellow"
style_root  = "bold red"
format      = "[$user]($style) "

[directory]
truncation_length = 2
truncate_to_repo  = false
truncation_symbol = "../"
format            = "on [$path]($style)[$read_only]($read_only_style) "
style             = "bold cyan"

[git_branch]
symbol = "󰘬 "
format = "on [$symbol$branch]($style) "
style  = "bold purple"

[git_status]
format = "([\\[$all_status$ahead_behind\\]]($style) )"
style  = "red"

[nodejs]
symbol = "󰎙 "
format = "via [$symbol($version)]($style) "
style  = "bold green"

[python]
symbol = "󰌠 "
format = "via [$symbol($version)]($style) "
style  = "bold yellow"

[docker_context]
symbol = "󰡨 "
format = "on [$symbol$context]($style) "
style  = "bold blue"
EOF

# ── Configuração do Zshrc ────────────────────────────────────
RUN cat <<'ZSHRC' > ~/.zshrc
HISTFILE=~/.zsh_history
HISTSIZE=50000
SAVEHIST=50000
setopt EXTENDED_HISTORY
setopt HIST_EXPIRE_DUPS_FIRST
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE
setopt HIST_VERIFY
setopt SHARE_HISTORY
setopt APPEND_HISTORY
setopt AUTO_CD
setopt AUTO_PUSHD
setopt PUSHD_IGNORE_DUPS
setopt CORRECT
setopt INTERACTIVE_COMMENTS
setopt NO_BEEP

autoload -Uz compinit
fpath=(~/.zsh/plugins/zsh-completions/src $fpath)
compinit -C

zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}'
zstyle ':completion:*' list-colors '${(s.:.)LS_COLORS}'
zstyle ':completion:*' group-name ''
zstyle ':completion:*:descriptions' format '%F{yellow}── %d ──%f'
zstyle ':completion:*:warnings' format '%F{red}Nenhum resultado encontrado%f'
zstyle ':completion:*' squeeze-slashes true
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path ~/.zsh/cache

bindkey -e
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
bindkey '^[[1;5C' forward-word
bindkey '^[[1;5D' backward-word
bindkey '^[[3~' delete-char
bindkey '^[[H' beginning-of-line
bindkey '^[[F' end-of-line

# ── PATH consolidado ────────────────────────────────────────
# /opt/venv/bin primeiro: garante que python, jupyter, torch etc.
# do venv ML sejam encontrados antes de qualquer outro Python.
export PATH="/opt/venv/bin:$HOME/.local/bin:$HOME/.local/share/fnm:$PATH"
export VIRTUAL_ENV=/opt/venv

# ── Inicializações de runtime ────────────────────────────────
eval "$(fnm env --use-on-cd --shell zsh)"
eval "$(starship init zsh)"

# ── Plugins Zsh ─────────────────────────────────────────────
source ~/.zsh/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh
source ~/.zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source ~/.zsh/plugins/zsh-history-substring-search/zsh-history-substring-search.zsh

ZSH_AUTOSUGGEST_STRATEGY=(history completion)
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=8'
ZSH_AUTOSUGGEST_BUFFER_MAX_SIZE=20

ZSH_HIGHLIGHT_HIGHLIGHTERS=(main brackets pattern)
typeset -A ZSH_HIGHLIGHT_STYLES
ZSH_HIGHLIGHT_STYLES[command]='fg=green,bold'
ZSH_HIGHLIGHT_STYLES[builtin]='fg=green,bold'
ZSH_HIGHLIGHT_STYLES[alias]='fg=green,bold'
ZSH_HIGHLIGHT_STYLES[unknown-token]='fg=red,bold'
ZSH_HIGHLIGHT_STYLES[path]='fg=cyan,underline'
ZSH_HIGHLIGHT_STYLES[globbing]='fg=magenta'
ZSH_HIGHLIGHT_STYLES[single-quoted-argument]='fg=yellow'
ZSH_HIGHLIGHT_STYLES[double-quoted-argument]='fg=yellow'

HISTORY_SUBSTRING_SEARCH_HIGHLIGHT_FOUND='bg=green,fg=black,bold'
HISTORY_SUBSTRING_SEARCH_HIGHLIGHT_NOT_FOUND='bg=red,fg=white,bold'

alias ll='ls -lah --color=auto --group-directories-first'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias gs='git status'
alias gd='git diff'
alias gl='git log --oneline -15'
alias gco='git checkout'
alias gcm='git commit -m'
alias gp='git push'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias mkdir='mkdir -pv'
alias grep='grep --color=auto'
alias df='df -h'
alias du='du -h'
alias free='free -h'

# ── Atalhos ML úteis ─────────────────────────────────────────
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=""'
alias gpu='nvidia-smi'
alias torch-check='python3 -c "import torch; print(f\"PyTorch {torch.__version__} | CUDA disponível: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"'

export npm_config_ignore_scripts=false
ZSHRC

# ── Bashrc mínimo ────────────────────────────────────────────
RUN cat <<'BASHRC' > ~/.bashrc
export PATH="$HOME/.local/bin:$HOME/.local/share/fnm:$PATH"
eval "$(fnm env --use-on-cd)" 2>/dev/null
eval "$(starship init bash)" 2>/dev/null
alias ll='ls -lah --color=auto --group-directories-first'
BASHRC

# ── Configuração do Git ──────────────────────────────────────
RUN git config --global user.name "${GIT_USER_NAME}" \
    && git config --global user.email "${GIT_USER_EMAIL}" \
    && git config --global init.defaultBranch main \
    && git config --global core.autocrlf input \
    && git config --global pull.rebase true

# ── Consolidação do PATH (ENV para processos não-interativos) ─
# Necessário para que scripts e o VS Code encontrem jupyter, uv, fnm
# sem depender do .zshrc (que só carrega em shells interativos).
ENV PATH="/opt/venv/bin:/home/${USERNAME}/.local/bin:/home/${USERNAME}/.local/share/fnm:${PATH}"

# ── Entrypoint ───────────────────────────────────────────────
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# ── Espaço de Trabalho ───────────────────────────────────────
WORKDIR /workspace

# ── Portas ───────────────────────────────────────────────────
EXPOSE 8888 6006

# ── Healthcheck ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD pgrep -x "sleep" > /dev/null || exit 1

# ── Execução ─────────────────────────────────────────────────
ENTRYPOINT ["entrypoint.sh"]
CMD ["sleep", "infinity"]
