---
title: NLTK + Stanford NLP 进行命名实体识别和词性标注
mathjax: false
date: 2019-12-20 11:44:10
tags: 
- stanford
- nltk
- ner
- pos
categories: NLP
---

NLTK 中使用 Stanford NLP 工具包进行NER和POS任务。

<!--more-->

# 安装环境

1. 下载
```
	http://nlp.stanford.edu/software/CRF-NER.html
	http://nlp.stanford.edu/software/tagger.html
```
2. 解压  
```sh
	unzip stanford-ner-2018-10-16.zip
	unzip stanford-postagger-full-2018-10-16.zip
```
3. 添加 `CLASSPATH` ，修改 `.bashrc` 文件:
```sh
	export STANFORD_NLTK_PATH=/home/haha/stanford_nltk  
	export STANFORD_NER_PATH=$STANFORD_NLTK_PATH/stanford-ner-2018-10-16
	export STANFORD_POS_PATH=$STANFORD_NLTK_PATH/stanford-postagger-full-2018-10-16
	export JAVA_HOME=/home/haha/java/jdk-13.0.1  
	export PATH=$JAVA_HOME/bin:$PATH  
	export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$STANFORD_NER_PATH/stanford-ner.jar:$STANFORD_POS_PATH/stanford-postagger.jar
```
4. 添加 `STANFORD_MODELS` ，修改 `.bashrc` 文件:
```sh
	export STANFORD_MODELS=$STANFORD_NER_PATH/classifiers:$STANFORD_POS_PATH/models
```

# 函数使用

```python
	from nltk.tag import StanfordNERTagger
	from nltk.tag import StanfordPOSTagger

	pos_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
	ner_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
	tokens = ['The', 'suspect', 'dumped', 'the', 'dead', 'body', 'into', 'a', 'local', 'reservoir', '.']
	pos = [each[1] for each in pos_tagger.tag(tokens)]
	ner = [each[1] for each in ner_tagger.tag(tokens)]
```