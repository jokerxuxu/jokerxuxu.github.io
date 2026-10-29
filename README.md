# jokerxuxu.github.io

苏中玉（某不科学的虚）的个人主页 + 技术博客。

- 框架：**Hugo**（extended）
- 主题：**PaperMod**（git submodule）
- 数学公式：**KaTeX**（已本地托管在 `static/katex/`，不依赖 CDN）
- 部署：**GitHub Pages** + GitHub Actions

线上地址（部署后）：<https://jokerxuxu.github.io/>

---

## 1. 本地预览

本机已经带了一个 Hugo 二进制在 `.tools/hugo`（不纳入 git），可以直接跑：

```bash
./scripts/serve.sh
# 打开 http://localhost:1313
```

如果换了机器，二选一：

```bash
# macOS + Homebrew
brew install hugo

# 或从 GitHub Releases 下载 extended 版本
# https://github.com/gohugoio/hugo/releases
```

> 必须是 **extended** 版本，且 Hugo ≥ 0.158（PaperMod 的要求）。

首次克隆仓库时记得带上主题子模块：

```bash
git clone --recurse-submodules https://github.com/jokerxuxu/jokerxuxu.github.io.git
# 已经克隆过的话：
git submodule update --init --recursive
```

---

## 2. 目录结构

```text
.
├── hugo.yaml                  # 站点配置：标题、导航、社交链接、首页信息
├── content/
│   ├── about.md               # About 页
│   ├── cv.md                  # CV 页
│   ├── publications.md        # 论文列表
│   ├── projects/              # 项目（每个项目一个 md）
│   └── posts/                 # 博客文章
├── assets/
│   ├── avatar.png             # 头像（会自动裁剪成圆形）
│   └── css/extended/custom.css# 少量样式微调
├── static/
│   ├── assets/                # 文章里用到的图片、favicon
│   └── katex/                 # KaTeX 本地文件
├── layouts/_partials/
│   └── extend_head.html       # 按需加载 KaTeX
├── .github/workflows/deploy.yml
├── scripts/                   # serve.sh / new.sh
└── themes/PaperMod            # 主题（submodule，不要手动改）
```

---

## 3. 改成你自己的内容

按重要性排序，全部细节见 [TODO.md](TODO.md)。

| 想改什么 | 改哪里 |
| --- | --- |
| 名字、副标题、按钮 | `hugo.yaml` → `params.profileMode` |
| 头像 | 替换 `assets/avatar.png`（正方形，建议 ≥ 400px） |
| 导航栏 | `hugo.yaml` → `menu.main` |
| GitHub / Scholar / 邮箱 | `hugo.yaml` → `params.socialIcons`，以及 `content/about.md`、`content/cv.md` |
| 首页简介 | `content/about.md` |
| 项目 | `content/projects/*.md` |
| 论文 | `content/publications.md` |
| 站点域名 | `hugo.yaml` → `baseURL` |

正文里需要写公式的页面，在 front matter 加 `math: true`：

```yaml
---
title: "某篇笔记"
date: 2026-11-01
math: true
---
```

行内公式 `$r_i(q)$`，块公式：

```markdown
$$
\tau_i = r_i(q_i)\,T
$$
```

---

## 4. 写一篇新文章

```bash
./scripts/new.sh mujoco-tendon-routing
```

会生成 `content/posts/mujoco-tendon-routing.md`，默认 `draft: true`。
本地预览加 `-D` 才看得到草稿（`serve.sh` 已经默认带上），写成 `draft: false` 才会发布。

图片统一放 `static/assets/`，Markdown 里这样引用：

```markdown
![描述](/assets/your-image.png)
```

---

## 5. 部署到 GitHub Pages

> ⚠️ **先读这段**：`jokerxuxu.github.io` 这个域名和同名仓库不是空的。它现在跑着一个 Hexo「虚的小屋」旧站
> （仓库 `jokerxuxu/jokerxuxu.github.io`，默认分支 `master`，最后一次提交 2024-07），归档里有 4 篇 2024 年的笔记：
> 理论力学（物理类）、数学物理方法、微积分、个人简介。
>
> 往那个仓库 push 会**直接覆盖旧站**。所以先决定旧站怎么处理：

| 方案 | 最终地址 | 说明 |
| --- | --- | --- |
| **A. 迁移旧文再替换**（推荐） | `https://jokerxuxu.github.io/` | 把 4 篇旧文改写成 Hugo 文章，用 `aliases` 保住旧链接；旧仓库先打 tag / 归档备份 |
| **B. 新站放子路径** | `https://jokerxuxu.github.io/home/` | 旧站完全不动；新仓库叫 `home`，`baseURL` 改成 `https://jokerxuxu.github.io/home/` |
| **C. 直接替换、不迁移** | `https://jokerxuxu.github.io/` | 最快；旧内容只留在 git 历史里，线上不再可见 |

### 部署步骤

1. 仓库：方案 A/C 用 `jokerxuxu.github.io`；方案 B 新建 `home`。
2. 推送：

   ```bash
   git remote add origin https://github.com/jokerxuxu/jokerxuxu.github.io.git
   git add .
   git commit -m "init personal site"
   git push -u origin main
   ```

   > 旧仓库的默认分支是 `master`。继续用这个仓库的话，要么在
   > **Settings → Branches** 把默认分支改成 `main`，要么把 `.github/workflows/deploy.yml`
   > 里的 `branches: [main]` 改成 `master`。

3. 仓库 **Settings → Pages → Build and deployment → Source** 选 **GitHub Actions**。
   如果旧站之前是 “Deploy from a branch”，这一步会停掉旧站。
4. 之后每次 `git push` 自动构建发布，通常 1–2 分钟。

> 方案 B 记得把 `hugo.yaml` 的 `baseURL` 改成 `https://jokerxuxu.github.io/home/`，
> 否则站内绝对链接会指到根目录。

### 自定义域名（可选）

在 `static/` 下新建 `CNAME` 文件，内容写你的域名（如 `suzhongyu.com`），然后在域名商处把 DNS 指向 GitHub Pages，最后在仓库 Pages 设置里填上域名并开启 HTTPS。

---

## 6. 常见坑

- **`themes/PaperMod` 是空的**：没初始化子模块，跑 `git submodule update --init --recursive`。
- **本地样式正常、线上 404**：`baseURL` 和仓库路径不一致。
- **公式显示成 `$$...$$` 原文**：该页 front matter 少了 `math: true`。
- **中文文章阅读时间/字数有点怪**：PaperMod 按空格分词，中文统计会偏少，不影响功能。
- **改了主题文件**：不要直接改 `themes/PaperMod`，升级会被覆盖；自定义放 `layouts/` 和 `assets/css/extended/`。
- **升级主题**：

  ```bash
  git submodule update --remote themes/PaperMod
  ```

---

## 7. 致谢

- [Hugo](https://gohugo.io/)
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod)
- [KaTeX](https://katex.org/)
