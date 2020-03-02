---
title: MySQL 8.0.19 Windows以及Navicat15破解版安装教程
mathjax: false
date: 2020-03-02 20:54:24
tags: mysql
categories: SQL
---

Windows10系统
MySQL安装版本：8.0.19
Navicat安装版本：premium 15 for mysql

Hint:

	SQL为结构化查询语句
	MySQL, SQL Server, Oracle等为数据库管理系统
	Navicat, SQLyog等为数据库管理工具

<!--more-->
# MySQL
## 下载
官网下载地址，选择64位安装包：mysql-8.0.19-winx64.zip。

	https://dev.mysql.com/downloads/mysql/

然而下载速度很慢，可以切换国内镜像。搜狐镜像也比较慢，网易镜像速度飞快。

	http://mirrors.163.com/mysql/Downloads/MySQL-8.0/
	http://mirrors.sohu.com/mysql/MySQL-8.0/

## 安装
### 解压
解压压缩包至指定的安装目录如：`D:\Program\MySQL`。
### 添加配置文件
手动新建配置文件`my.ini`，写入内容（无需手动创建data文件夹）：

	[mysqld]
	# 设置3306端口
	port=3306
	# 设置mysql的安装目录
	basedir=D:\Program\MySQL
	# 设置mysql数据库的数据的存放目录
	datadir=D:\Program\MySQL\data
	# 允许最大连接数
	max_connections=200
	# 允许连接失败的次数。
	max_connect_errors=10
	# 服务端使用的字符集默认为utf8mb4
	character-set-server=utf8mb4
	# 创建新表时将使用的默认存储引擎
	default-storage-engine=INNODB
	# 默认使用“mysql_native_password”插件认证
	#mysql_native_password
	default_authentication_plugin=mysql_native_password
	[mysql]
	# 设置mysql客户端默认字符集
	default-character-set=utf8mb4
	[client]
	# 设置mysql客户端连接服务端时默认使用的端口
	port=3306
	default-character-set=utf8mb4

### 初始化数据库
管理员身份打开cmd窗口，执行如下命令，将生成root用户的初始密码，用作后续登录。

	mysqld --initialize --user=mysql --console

## 安装数据库

1. 执行命令：`mysqld -install `
2. 启动服务：`net start mysql`
3. 登录数据库：`mysql -u root -p`，并输入刚刚得到的密码
4. 修改密码：`alter user 'root'@'localhost' identified by 'yourpassword'`
5. 退出：`exit`

### 报错
若安装过程中报错，`VCRUNTIME140_1.dll`问题，微软官网下载适用于Visual Studio 2015、2017 和 2019 的 Microsoft Visual C++ 可再发行软件包：

	https://support.microsoft.com/zh-cn/help/2977003/the-latest-supported-visual-c-downloads

### 参考

	https://blog.csdn.net/qq_37350706/article/details/81707862

# Navicat
参考如下，玄学破解，需要多试几次。

	https://www.cnblogs.com/runw/p/12255962.html

官方使用手册

	https://www.navicat.com.cn/support/online-manual