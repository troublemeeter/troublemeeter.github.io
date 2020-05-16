---
title: Pytorch常用操作
mathjax: false
date: 2020-05-15 17:52:27
tags: pytorch
categories: pytorch
---
Pytorch常用操作记录，pytorch文档：
```url
https://pytorch.org/docs/stable/
```
<!--more-->

# 基本数据类型

## data type

|Data type                  |dtype                                         |CPU tensor                    |GPU tensor|
|:---|:---|:---|:---|
|32-bit floating point      |``torch.float32`` or ``torch.float``          |`torch.FloatTensor`    |`torch.cuda.FloatTensor`|
|64-bit floating point      |``torch.float64`` or ``torch.double``         |`torch.DoubleTensor`   |`torch.cuda.DoubleTensor`|
|16-bit floating point      |``torch.float16`` or ``torch.half``           |`torch.HalfTensor`     |`torch.cuda.HalfTensor`|
|8-bit integer (unsigned)   |``torch.uint8``                               |`torch.ByteTensor`     |`torch.cuda.ByteTensor`|
|8-bit integer (signed)     |``torch.int8``                                |`torch.CharTensor`     |`torch.cuda.CharTensor`|
|16-bit integer (signed)    |``torch.int16`` or ``torch.short``            |`torch.ShortTensor`    |`torch.cuda.ShortTensor`|
|32-bit integer (signed)    |``torch.int32`` or ``torch.int``              |`torch.IntTensor`      |`torch.cuda.IntTensor`|
|64-bit integer (signed)    |``torch.int64`` or ``torch.long``             |`torch.LongTensor`     |`torch.cuda.LongTensor`|
|Boolean                    |``torch.bool``                                |`torch.BoolTensor`     |`torch.cuda.BoolTensor`|

## type check
```python
>>> a = torch.randn(2,3)
>>> a.type()
'torch.FloatTensor'

>>> type(a)
torch.Tensor

>>> isinstance(a, torch.FloatTensor)
True
```

## dim check
```python
>>> a = torch.rand(1,2,3)
>>> a.shape
torch.Size([1,2,3])

>>> list(a.shape)
[1,2,3]

>>> a.size(0)
1

>>> a.dim()
3
```

# 创建Tensor

## from numpy
```python
>>> a = np.array([2,3.3])
>>> torch.from_numpy(a)
tensor([2, 3])
```

## from list
```python
In [*]: torch.tensor([2,3])
Out[*]: tensor([2, 3])

In [*]: torch.FloatTensor([2,3.3])
Out[*]: tensor([2.0000, 3.3000])
```
## uninitialize
```python
In [*]: torch.Tensor(2,3)
Out[*]:
tensor([[-4.2107e-05,  4.5766e-41,  7.6038e-30],
        [ 3.0963e-41,  1.1988e-20,  4.5766e-41]])
```

## 设置默认数据类型
```python
In [*]: torch.tensor([1.2, 3]).type()
Out[*]: 'torch.FloatTensor'

In [*]: torch.set_default_tensor_type(torch.DoubleTensor)

In [*]: torch.tensor([1.2, 3]).type()
Out[*]: 'torch.DoubleTensor'
```

## 随机初始化
```python
In [*]: a = torch.rand(2,2)
In [*]: a
Out[*]: tensor([[0.0934, 0.9568],
       			 [0.3287, 0.5201]])

In [*]: torch.rand_like(a)
Out[*]: tensor([[0.1323, 0.4544],
       		     [0.3094, 0.7894]])

In [*]: torch.randint(1,10,[2,2])
Out[*]: tensor([[8, 9],
        		 [3, 1]])

In [*]: torch.randn(2,2) #标准正态分布
Out[*]: tensor([[-0.6804,  0.0677],
       			 [-0.8232, -0.8727]])

In [*]: torch.full([2,2],5) 
Out[*]: tensor([[5., 5.],
        		 [5., 5.]])
```

## 序列初始化
```python
In [*]: torch.arange(0,4)
Out[*]: tensor([0, 1, 2, 3])

In [*]: torch.arange(0,4,2)
Out[*]: tensor([0, 2])

In [*]: torch.arange(0,10)
Out[*]: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

In [*]: torch.arange(0,10,2)
Out[*]: tensor([0, 2, 4, 6, 8])

In [*]: torch.range(0,10)
/home/haha/anaconda3/envs/py36/bin/ipython:1: UserWarning: torch.range is deprecated in favor of torch.arange and will be removed in 0.5. Note that arange generates values in [start; end), not [start; end].
  #!/home/haha/anaconda3/envs/py36/bin/python
Out[*]: tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

In [*]: torch.linspace(0,10,steps=5)
Out[*]: tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])

In [*]: torch.logspace(0,1,steps=5)
Out[*]: tensor([ 1.0000,  1.7783,  3.1623,  5.6234, 10.0000])
```

## 特殊初始化
```python
In [*]: torch.ones(2,2)
Out[*]:
tensor([[1., 1.],
        [1., 1.]])

In [*]: torch.zeros(2,2)
Out[*]:
tensor([[0., 0.],
        [0., 0.]])

In [*]: torch.eye(2,3)
Out[*]:
tensor([[1., 0., 0.],
        [0., 1., 0.]])

In [*]: a = torch.zeros(3,3)

In [*]: torch.ones_like(a)
Out[*]:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])

In [*]: torch.randperm(5)
Out[*]: tensor([0, 2, 3, 1, 4])
```

# 切片和索引
## indexing
```python
In [*]: a = torch.rand(4,3,28,28)

In [*]: a[*].shape
Out[*]: torch.Size([3, 28, 28])

In [*]: a[0,0,2].shape
Out[*]: torch.Size([*])

In [*]: a[0,0,2,4]
Out[*]: tensor(0.0452)
```

## first/last N
```python 
In [*]: a[:2,1:,-1:,:].shape
Out[*]: torch.Size([2, 2, 1, 28])
```

## by step
```python
In [*]: a[:,:,0:28:2,0:28:3].shape
Out[*]: torch.Size([4, 3, 14, 10])

In [*]: a[:,:,::2,::-1].shape
-----------------------------------------------------------------------
ValueError                            Traceback (most recent call last)
<ipython-input-73-f808e66143f0> in <module>
----> 1 a[:,:,::2,::-1].shape

ValueError: negative step not yet supported

In [*]: a[:,:,::2,::3].shape
Out[*]: torch.Size([4, 3, 14, 10])
```
## by specific index
```python
In [*]: a.index_select(1,torch.arange(2)).shape
Out[*]: torch.Size([4, 2, 28, 28])

In [*]: a.index_select(3,torch.arange(7)).shape
Out[*]: torch.Size([4, 3, 28, 7])

In [*]: a[...,:2].shape
Out[*]: torch.Size([4, 3, 28, 2])

In [*]: a[1,...].shape
Out[*]: torch.Size([3, 28, 28])
```

## by mask
```python
In [*]: x = torch.randn(3,4)

In [*]: mask = x.ge(0.5)

In [*]: mask
Out[*]:
tensor([[0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0]], dtype=torch.uint8)

In [*]: torch.masked_select(x,mask)
Out[*]: tensor([0.7164, 2.1624])
```
## by flatten index
```python
In [*]: a = torch.tensor([[4,3,5],[6,8,7]])

In [*]: torch.take(a,torch.tensor([0,2,-1]))
Out[*]: tensor([4, 5, 7])
```

# 维度变换
## view/reshape
```python
In [*]: a = torch.rand(4,1,28,28
In [*]: a.view(4,28*28).shape
Out[*]: torch.Size([4, 784])
```
## squeeze/unsqueeze
### squeeze
```python
In [*]: a.shape
Out[*]: torch.Size([4, 1, 28, 28])

In [*]: a.unsqueeze(0).shape
Out[*]: torch.Size([1, 4, 1, 28, 28])

In [*]: a.unsqueeze(-1).shape
Out[*]: torch.Size([4, 1, 28, 28, 1])
```
### unsqueeze
```python
In [*]: a = torch.rand(1,32,1,1)

In [*]: a.shape
Out[*]: torch.Size([1, 32, 1, 1])

In [*]: a.squeeze().shape
Out[*]: torch.Size([*])

In [*]: a.squeeze(0).shape
Out[*]: torch.Size([32, 1, 1])

In [*]: a.squeeze(1).shape
Out[*]: torch.Size([1, 32, 1, 1])
```
## transpose/t/permute
### t
only expects a 2D tensor
```python
In [*]: a = torch.rand(3,4)

In [*]: a.t().shape
Out[*]: torch.Size([4, 3])
```
### transpose
```python
In [*]: a = torch.rand(4,3,32,32)

In [*]: a.transpose(1,3).view(4,3*32*32).shape
---------------------------------------------------------------
RuntimeError                  Traceback (most recent call last)
<ipython-input-45-bdff182163c0> in <module>
----> 1 a.transpose(1,3).view(4,3*32*32).shape

RuntimeError: invalid argument 2: view size is not compatible with input tensor size and stride (at least one dimension spans across two contiguous subspaces). Call .contiguous() before .view(). at /pytorch/aten/src/TH/generic/THTensor.cpp:203

In [*]: a1 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32)

In [*]: a2 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)

In [*]: a1.shape,a2.shape
Out[*]: (torch.Size([4, 3, 32, 32]), torch.Size([4, 3, 32, 32]))

In [*]: torch.all(torch.eq(a,a1))
Out[*]: tensor(0, dtype=torch.uint8)

In [*]: torch.all(torch.eq(a,a2))
Out[*]: tensor(1, dtype=torch.uint8)
```
### permute
```python
In [*]: a.permute(0,2,3,1).shape
Out[*]: torch.Size([4, 32, 32, 3])
```

## expand/repeat
### expand: broadcasting
```python
In [*]: a = torch.rand(1,32,1,1)

In [*]: a.expand(4,32,14,14).shape
Out[*]: torch.Size([4, 32, 14, 14])
```
### repeat: memory copied
```python
In [*]: a = torch.rand(1,32,1,1)

In [*]: a.repeat(4,32,1,1).shape
Out[*]: torch.Size([4, 1024, 1, 1])

In [*]: a.repeat(4,1,1,1).shape
Out[*]: torch.Size([4, 32, 1, 1])

In [*]: a.repeat(4,1,32,32).shape
Out[*]: torch.Size([4, 32, 32, 32])
```

# 拼接与拆分
## cat
```python
In [*]: a = torch.rand(4,32,8)

In [*]: b = torch.rand(5,32,8)

In [*]: torch.cat([a,b],dim=0).shape
Out[*]: torch.Size([9, 32, 8])
```
## stack
create new dim, 两个tensor维度相同
```python
In [*]: a.shape
Out[*]: torch.Size([4, 32, 8])

In [*]: torch.stack([a,a],dim=1).shape
Out[*]: torch.Size([4, 2, 32, 8])

In [*]: torch.stack([a,a],dim=3).shape
Out[*]: torch.Size([4, 32, 8, 2])
```
## split
by length
```python
In [*]: c = torch.rand(2,32,8)

In [*]: a,b = c.split([10,22],dim=1)

In [*]: a.shape,b.shape
Out[*]: (torch.Size([2, 10, 8]), torch.Size([2, 22, 8]))
```

```python
In [*]: a,b = c.split(16,dim=1)

In [*]: a.shape,b.shape
Out[*]: (torch.Size([2, 16, 8]), torch.Size([2, 16, 8]))

In [*]: a = c.split(2,dim=0)

In [*]: a.shape
---------------------------------------------------------------
AttributeError                Traceback (most recent call last)
<ipython-input-86-d74f1bcdd37c> in <module>
----> 1 a.shape

AttributeError: 'tuple' object has no attribute 'shape'

In [*]: len(a)
Out[*]: 1

In [*]: a[*].shape
Out[*]: torch.Size([2, 32, 8])
```
## chunk
by num
```python
In [*]: a,b = c.chunk(2,dim=0)

In [*]: a.shape,b.shape
Out[*]: (torch.Size([1, 32, 8]), torch.Size([1, 32, 8]))
```

# 数学运算
## add/minus/multiply/divide
```python
In [*]: a = torch.rand(3,4)

In [*]: b = torch.rand(4)

In [*]: a+b
Out[*]:
tensor([[0.8015, 1.1462, 1.0086, 0.3205],
        [1.2817, 1.0414, 1.3938, 0.8069],
        [0.4473, 0.4525, 1.7025, 0.9032]])

In [*]: torch.add(a,b)
Out[*]:
tensor([[0.8015, 1.1462, 1.0086, 0.3205],
        [1.2817, 1.0414, 1.3938, 0.8069],
        [0.4473, 0.4525, 1.7025, 0.9032]])
```
## matmul
torch.mm: only for 2d tensor
```python
In [*]: a
Out[*]:
tensor([[3., 3.],
        [3., 3.]])

In [*]: b = torch.ones(2,2)

In [*]: torch.mm(a,b)
Out[*]:
tensor([[6., 6.],
        [6., 6.]])

In [*]: torch.matmul(a,b)
Out[*]:
tensor([[6., 6.],
        [6., 6.]])

In [*]: a@b
Out[*]:
tensor([[6., 6.],
        [6., 6.]])
```

## floor/ceil/round/trunc/frac
```python
In [*]: a
Out[*]: tensor(3.1400)

In [*]: a.floor(),a.ceil(),a.trunc(),a.frac()
Out[*]: (tensor(3.), tensor(4.), tensor(3.), tensor(0.1400))

In [*]: torch.tensor(3.499).round()
Out[*]: tensor(3.)

In [*]: torch.tensor(3.5).round()
Out[*]: tensor(4.)
```
## clamp
```python
In [*]: a
Out[*]:
tensor([[8.0798, 8.1040, 3.2681],
        [2.9466, 3.6986, 7.1211]])

In [*]: a.clamp(3.5,7.5)
Out[*]:
tensor([[7.5000, 7.5000, 3.5000],
        [3.5000, 3.6986, 7.1211]])
```
# 统计属性
## min/max/mean/sum/argmax/argmin
```python
In [*]: a
Out[*]:
tensor([[0., 1., 2., 3.],
        [4., 5., 6., 7.]])

In [*]: a.min(),a.max(),a.mean(),a.prod(),a.sum(),a.argmax(),a.argmin()
Out[*]:
(tensor(0.),
 tensor(7.),
 tensor(3.5000),
 tensor(0.),
 tensor(28.),
 tensor(7),
 tensor(0))

In [*]: a.argmax(dim=1)
Out[*]: tensor([3, 3]) 

In [*]: a.argmax(dim=1, keepdim=True)
Out[*]:
tensor([[*],
        [*]])

In [*]: a.max(dim=1)
Out[*]:
torch.return_types.max(
values=tensor([3., 7.]),
indices=tensor([3, 3]))

In [*]: a.max(dim=1,keepdim=True)
Out[*]:
torch.return_types.max(
values=tensor([[3.],
        [7.]]),
indices=tensor([[*],
        [*]]))
```

## topk/kth-value
```python
In [*]: a
Out[*]:
tensor([[0., 1., 2., 3.],
        [4., 5., 6., 7.]])

In [*]: a.topk(2,dim=1)
Out[*]:
torch.return_types.topk(
values=tensor([[3., 2.],
        [7., 6.]]),
indices=tensor([[3, 2],
        [3, 2]]))

In [*]: a.topk(2,dim=1,largest=False)
Out[*]:
torch.return_types.topk(
values=tensor([[0., 1.],
        [4., 5.]]),
indices=tensor([[0, 1],
        [0, 1]]))

In [*]: a.kthvalue(3,dim=1)
Out[*]:
torch.return_types.kthvalue(
values=tensor([2., 6.]),
indices=tensor([2, 2]))
```
## compare
```python
In [*]: a
Out[*]:
tensor([[0., 1., 2., 3.],
        [4., 5., 6., 7.]])

In [*]: a>0
Out[*]:
tensor([[0, 1, 1, 1],
        [1, 1, 1, 1]], dtype=torch.uint8)

In [*]: torch.gt(a,0)
Out[*]:
tensor([[0, 1, 1, 1],
        [1, 1, 1, 1]], dtype=torch.uint8)

In [*]: torch.eq(a,a)
Out[*]:
tensor([[1, 1, 1, 1],
        [1, 1, 1, 1]], dtype=torch.uint8)

In [*]: torch.equal(a,a)
Out[*]: True
```

# where / gather
## where
根据逻辑值将两个tensor进行聚合
```python
In [*]: cond = torch.randn(2,2)

In [*]: cond
Out[*]:
tensor([[ 1.9013, -0.6679],
        [-1.1109, -1.0334]])

In [*]: a = torch.ones(2,2)

In [*]: a
Out[*]:
tensor([[1., 1.],
        [1., 1.]])

In [*]: b = torch.zeros(2,2)

In [*]: b
Out[*]:
tensor([[0., 0.],
        [0., 0.]])

In [*]: torch.where(cond>0,a,b)
Out[*]:
tensor([[1., 0.],
        [0., 0.]])
```
## gather
根据index选择某个tensor中的一部分
```python
In [*]: idx = torch.randint(10,[3,3])

In [*]: idx
Out[*]:
tensor([[2, 3, 7],
        [0, 4, 4],
        [6, 4, 4]])

In [*]: label = (torch.arange(10)+100).expand(3,10)

In [*]: label
Out[*]:
tensor([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])

In [*]: torch.gather(label,dim=1,index=idx)
Out[*]:
tensor([[102, 103, 107],
        [100, 104, 104],
        [106, 104, 104]])
```