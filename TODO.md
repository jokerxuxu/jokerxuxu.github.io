# 待替换的占位符

按顺序改完这些，站点就是你的了。改完一项勾一项。

## 先决定

- [ ] **旧站怎么办**：`jokerxuxu.github.io` 现在是一个 Hexo 旧站（4 篇 2024 年的课程笔记 + 个人简介）。迁移 / 放子路径 / 直接替换，见 README 第 5 节

## 必改

- [ ] **邮箱**：`hugo.yaml` → `params.socialIcons` 里 `mailto:YOUR_EMAIL@example.com`
- [ ] **邮箱**：`content/about.md` 底部 `Elsewhere` 部分
- [ ] **邮箱**：`content/cv.md` 底部 `Contact` 部分
- [ ] **Google Scholar**：`hugo.yaml` 和 `content/about.md` 里的 `YOUR_SCHOLAR_ID`
- [ ] **学校与时间**：`content/about.md` 和 `content/cv.md` 的 Education 部分
- [ ] **头像**：替换 `assets/avatar.png`（正方形照片，建议 ≥ 400px；会自动裁圆）
- [ ] **favicon**（可选）：`static/assets/favicon-16.png`、`favicon-32.png`、`apple-touch-icon.png`

## 内容

- [ ] **Projects**：`content/projects/` 三篇目前是模板，补上真实仓库链接、结果数字、图片
- [ ] **Publications**：`content/publications.md` 目前是空列表 + 注释模板，有论文/预印本后按模板补上
- [ ] **CV**：`content/cv.md` 里 `[Your University]`、`[Lab / group]`、`20XX`；有 PDF 的话放到 `static/cv.pdf`
- [ ] **About 中文简介**：最后一段可以按自己的语气改

## 可选

- [ ] **自定义域名**：改 `hugo.yaml` 的 `baseURL`，加 `static/CNAME`
- [ ] **中文界面**：PaperMod 支持多语言，目前站点是英文，中文段落直接写在正文里
- [ ] **统计**：需要访问统计的话，在 `hugo.yaml` 里加 `params.analytics`（PaperMod wiki 有说明）
- [ ] **删除**这句 “Manuscripts in preparation”，如果你近期没有在写的稿子

## 部署前检查

- [ ] `./scripts/serve.sh` 打开每个页面，确认没有死链
- [ ] 所有 `YOUR_`、`[Your ...]`、`example.com` 都清掉了
- [ ] 仓库名是 `jokerxuxu.github.io`
- [ ] Settings → Pages → Source 已选 GitHub Actions
