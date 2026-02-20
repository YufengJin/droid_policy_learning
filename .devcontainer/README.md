# Dev Container 配置说明

本目录用于 [Dev Container](https://containers.dev/) / Cursor「在容器中重新打开」时自动应用配置。

## 与现有 Docker 的关系

- **Compose 文件**：使用项目内 `docker/docker-compose.headless.yaml`，与当前 `droid-dev-headless` 容器一致。
- **工作目录**：容器内工作目录为 `/workspace/droid_policy_learning`。

## 预装扩展

在 `devcontainer.json` 的 `customizations.vscode.extensions` 中配置了以下扩展（在容器内自动安装）：

| 扩展 ID | 用途 |
|--------|------|
| `ms-python.python` | Python 语言支持 |
| `ms-python.vscode-pylance` | Python 语言服务（补全、类型检查） |
| `ms-python.debugpy` | Python 调试（与 `.vscode/launch.json` 中的 debugpy 配置对应） |
| `redhat.vscode-yaml` | YAML 编辑 |
| `ms-azuretools.vscode-docker` | Docker 文件与容器管理 |

若提示某扩展需在 Remote Extension Host（容器内）运行，请将其 **扩展 ID** 加入上述列表（以及下面的「附加用」配置）后重新打开/附加容器。扩展 ID 可在扩展详情页或 [VS Code Marketplace](https://marketplace.visualstudio.com/) 查看。

---

## 为什么容器里还是不能安装？

常见原因有两点，对应两种使用方式。

### 1. 你是用「附加到已运行容器」(Attach to Running Container)

**原因**：附加时 Cursor/VS Code **不会**使用仓库里的 `.devcontainer/devcontainer.json`，而是使用本机上的「附加用容器配置」。该配置由 Cursor 在第一次附加时创建，默认可能是空的，所以扩展不会自动安装。

**做法**（任选其一）：

- **在附加用配置里加上扩展**  
  1. 已附加到容器后，按 **F1**，执行：**Dev Containers: Open Container Configuration File**（或 **Open Named Configuration File** 按容器名配置）。  
  2. 在打开的 JSON 里加上 `extensions` 数组（可参考本目录下的 `attached-container-config.example.json`），保存。  
  3. 断开后重新「附加到同一容器/镜像」，扩展会在下次连接时安装。

- **手动在容器内安装**  
  在扩展视图里找到对应扩展，点击 **「Install in Container」**；若安装失败，可重载窗口 (Reload Window) 后再试。

### 2. 你是用「在容器中重新打开」(Reopen in Container)

**原因**：此时会用到本仓库的 `devcontainer.json`，但 Cursor 存在已知问题：有时不会把 `customizations.vscode.extensions` 里的扩展自动装到容器内。

**做法**：

- 仍可尝试一次「Reopen in Container」看是否生效。  
- 若未生效，改用上面的「附加到已运行容器」，并按 **1** 在「Open Container Configuration File」里添加 `extensions`，或对每个扩展使用「Install in Container」。

### 3. 报错「End of central directory record signature not found / zip truncated」

**原因**：Cursor 使用 `OVSX_REGISTRY_URL` 时，会从 Open VSX 拉取扩展并校验 `.sigzip` 签名。若缺少 `node-ovsx-sign` 或校验失败，会报此错。

**本仓库已做的修复**：

- 在 `devcontainer.json` 与 `docker-compose.headless.yaml` 中设置 `OVSX_REGISTRY_URL=`，强制使用官方 Marketplace 而非 Open VSX。  
- 若已启动容器，需**重新构建/启动**（Reopen in Container 或 `docker compose up -d --force-recreate`）后环境变量才会生效。

**若仍未解决，可手动安装**：在容器内执行  
`curl -sL --compressed -o /tmp/debugpy.vsix "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-python/vsextensions/debugpy/latest/vspackage"`  
然后在扩展面板 → 右上角 `...` →「从 VSIX 安装…」→ 选择 `/tmp/debugpy.vsix`。

### 4. 其他可能原因

- **网络**：容器内无法访问扩展市场（如公司代理/防火墙），会安装失败；需在容器或主机配置代理。  
- **权限**：Cursor 在容器内写入的目录（如 `~/.cursor-server`）不可写时也会失败；当前镜像以 root 运行，一般无此问题。

---

## 使用方式

1. **Reopen in Container**：打开本项目后，F1 → **Dev Containers: Reopen in Container**，会按本目录的 `devcontainer.json` 构建/启动并尝试安装扩展（可能受 Cursor 已知问题影响）。
2. **Attach to Running Container**：若已用 `docker compose -f docker/docker-compose.headless.yaml up -d` 启动容器，用 **Attach to Running Container** 连到 `droid-dev-headless`。扩展**不会**从 `devcontainer.json` 安装，需按上面「为什么容器里还是不能安装」在 **Open Container Configuration File** 里添加 `extensions`，或手动「Install in Container」。

本仓库提供的 `attached-container-config.example.json` 仅作参考，列出建议在「附加用」配置里填写的 `workspaceFolder`、`extensions`、`remoteUser`，可复制到 Open Container Configuration File 中使用。

## 可选：增加扩展

- **Reopen in Container**：编辑 `devcontainer.json`，在 `customizations.vscode.extensions` 数组中追加扩展 ID。  
- **Attach**：在 **Open Container Configuration File** 的 `extensions` 数组中追加扩展 ID。

保存后，Reopen 需重新打开/构建容器；Attach 需断开后重新附加到同一镜像/容器名。
