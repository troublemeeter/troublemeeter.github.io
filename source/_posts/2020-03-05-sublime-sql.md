---
title: sublime中配置sqltools连接数据库
mathjax: false
date: 2020-03-05 22:51:30
tags: SQL
categories: SQL 
---
利用sublime插件sqltools，实现sublime连接数据库，在sublime中优雅的写SQL。
<!--more-->

# 安装
在package control中安装SQLTOOLS

# 配置文件
配置connection设置文件：首选项 - Package Settings - SQLTools - Connecttions:

	{
	    "connections": {
	        "my_local": {
	            "type"    : "mysql",
	            "host"    : "127.0.0.1",
	            "port"    : "3306",
	            "username": "root",
	            "password": "0823",
	            "database": "myemployees",
	            "encoding": "utf-8"
	        },
	    },
	    "default": "my_local"
	}

配置setting设置文件：首选项 - Package Settings - SQLTools - Settings:

	{
		"cli":
		{
			"mysql": "D:/Program/MySQL/bin/mysql.exe"
		}
	}

# 使用

1. 快捷键`ctrl+alt+e`进行数据库选择

2. 选择数据库后，键入sql语句，光标定位在sql语句上，快捷键`ctrl+e+e`，运行语句

# 美化

安装SqlBeautifier插件美化SQL语句，命令：`ctrl+k` + `ctrl+f` 