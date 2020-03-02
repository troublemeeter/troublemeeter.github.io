---
title: Hexo 博客常用操作命令
mathjax: false
date: 2019-12-20 11:44:53
tags: hexo
categories: blog
---

可参考官网详细教程

	https://hexo.io/zh-cn/docs/

<!--more-->

hexo安装以后，可以使用以下两种方式执行 Hexo：  

1. `npx hexo <command>`  
2. 将 Hexo 所在的目录下的 node_modules 添加到环境变量之中即可直接使用 `hexo <command>`：  
	`echo 'PATH="$PATH:./node_modules/.bin"' >> ~/.profile`


### new

新建内容。如果没有设置 layout 的话，默认使用 \_config.yml 中的 `default_layout` 参数代替。如果标题包含空格的话，请使用引号括起来。

	$ hexo new [layout] <title>
	$ hexo new "post title with whitespace"

### generate

生成静态文件:

	$ hexo generate
	$ hexo g

### publish

草稿发布:

	$ hexo publish [layout] <filename>

### server

开启服务:

	$ hexo server
	$ hexo s

### deploy

部署网站:

	$ hexo deploy
	$ hexo d

### 常用操作

本地开启服务：

	$ npx hexo g & npx hexo s

上传部署：

	$ npx hexo g & npx hexo s
