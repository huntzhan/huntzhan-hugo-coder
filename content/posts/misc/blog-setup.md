+++
date = 2018-10-06T17:10:49+08:00
title = "使用 Hugo + Github Pages 搭建博客"
+++

# 瞎扯一通

我已经拥有过很多个博客了。

* 一开始是自己用 Python 写的静态博客生成器，之后发现类似的工具已经有人做了，于是切到了  Jekyll。
* 开始想要搞各种酷炫的功能，比如前端特效、CDN 加速、SPA。于是切到了更容易扩展的 Ghost。
* 感觉 Ghost 也没那么好用（主要是编辑器与我想要的不一样），慢慢没有了写博客的热情（主要也没什么好写的）。

最近动极思静，想着是时候认真搞一个长期的个人博客，总结沉淀自己的知识~~以方便写在简历上~~。简单想了一下，其实我最核心的需求是：

* 高可用，内容长久可访问。
* 运营成本低（最好不用花钱）。
* 支持 MathJax，容易管理内容（图片、文件等）。
* 排版还行。
* 不折腾。

于是，找了一把，发现 Hugo + Github Pages 很好地满足了我的需求，于是就有了现在这个站点。本文主要介绍搭这个博客的方法。

# 博客搭建

参考 [Host on GitHub](https://gohugo.io/hosting-and-deployment/hosting-on-github/) 的建议，[huntzhan/huntzhan-hugo-coder](https://github.com/huntzhan/huntzhan-hugo-coder) 用于存放源文件（markdown/image files），通过 Travis CI 生成静态站点并自动部署至 [huntzhan/huntzhan.github.io](https://github.com/huntzhan/huntzhan.github.io)。

具体的 Travis CI 配置方法见：

* [.travis.yml](https://github.com/huntzhan/huntzhan-hugo-coder/blob/master/.travis.yml)
* [相关脚本](https://github.com/huntzhan/huntzhan-hugo-coder/tree/master/.travis)
* [How to set up TravisCI for projects that push back to github](https://gist.github.com/willprice/e07efd73fb7f13f917ea)

本站选用的轻量级主题：[luizdepra/hugo-coder](https://github.com/luizdepra/hugo-coder)。

整个流程走下来基本比较顺畅，推荐尝试。