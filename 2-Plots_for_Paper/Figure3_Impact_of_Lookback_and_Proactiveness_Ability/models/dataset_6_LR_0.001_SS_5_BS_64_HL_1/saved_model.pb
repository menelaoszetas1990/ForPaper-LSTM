╫а9
П▐
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758КШ7
Щ
 Adam/lstm_29/lstm_cell_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_29/lstm_cell_29/bias/v
Т
4Adam/lstm_29/lstm_cell_29/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_29/lstm_cell_29/bias/v*
_output_shapes	
:А*
dtype0
╡
,Adam/lstm_29/lstm_cell_29/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*=
shared_name.,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v
о
@Adam/lstm_29/lstm_cell_29/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
в
"Adam/lstm_29/lstm_cell_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*3
shared_name$"Adam/lstm_29/lstm_cell_29/kernel/v
Ы
6Adam/lstm_29/lstm_cell_29/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_29/lstm_cell_29/kernel/v* 
_output_shapes
:
АА*
dtype0
Щ
 Adam/lstm_28/lstm_cell_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_28/lstm_cell_28/bias/v
Т
4Adam/lstm_28/lstm_cell_28/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_28/lstm_cell_28/bias/v*
_output_shapes	
:А*
dtype0
╢
,Adam/lstm_28/lstm_cell_28/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*=
shared_name.,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v
п
@Adam/lstm_28/lstm_cell_28/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v* 
_output_shapes
:
АА*
dtype0
б
"Adam/lstm_28/lstm_cell_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_28/lstm_cell_28/kernel/v
Ъ
6Adam/lstm_28/lstm_cell_28/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_28/lstm_cell_28/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_14/kernel/v
Б
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:@*
dtype0
Щ
 Adam/lstm_29/lstm_cell_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_29/lstm_cell_29/bias/m
Т
4Adam/lstm_29/lstm_cell_29/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_29/lstm_cell_29/bias/m*
_output_shapes	
:А*
dtype0
╡
,Adam/lstm_29/lstm_cell_29/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*=
shared_name.,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m
о
@Adam/lstm_29/lstm_cell_29/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
в
"Adam/lstm_29/lstm_cell_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*3
shared_name$"Adam/lstm_29/lstm_cell_29/kernel/m
Ы
6Adam/lstm_29/lstm_cell_29/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_29/lstm_cell_29/kernel/m* 
_output_shapes
:
АА*
dtype0
Щ
 Adam/lstm_28/lstm_cell_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_28/lstm_cell_28/bias/m
Т
4Adam/lstm_28/lstm_cell_28/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_28/lstm_cell_28/bias/m*
_output_shapes	
:А*
dtype0
╢
,Adam/lstm_28/lstm_cell_28/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*=
shared_name.,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m
п
@Adam/lstm_28/lstm_cell_28/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m* 
_output_shapes
:
АА*
dtype0
б
"Adam/lstm_28/lstm_cell_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_28/lstm_cell_28/kernel/m
Ъ
6Adam/lstm_28/lstm_cell_28/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_28/lstm_cell_28/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_14/kernel/m
Б
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
Л
lstm_29/lstm_cell_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_29/lstm_cell_29/bias
Д
-lstm_29/lstm_cell_29/bias/Read/ReadVariableOpReadVariableOplstm_29/lstm_cell_29/bias*
_output_shapes	
:А*
dtype0
з
%lstm_29/lstm_cell_29/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*6
shared_name'%lstm_29/lstm_cell_29/recurrent_kernel
а
9lstm_29/lstm_cell_29/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_29/lstm_cell_29/recurrent_kernel*
_output_shapes
:	@А*
dtype0
Ф
lstm_29/lstm_cell_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*,
shared_namelstm_29/lstm_cell_29/kernel
Н
/lstm_29/lstm_cell_29/kernel/Read/ReadVariableOpReadVariableOplstm_29/lstm_cell_29/kernel* 
_output_shapes
:
АА*
dtype0
Л
lstm_28/lstm_cell_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_28/lstm_cell_28/bias
Д
-lstm_28/lstm_cell_28/bias/Read/ReadVariableOpReadVariableOplstm_28/lstm_cell_28/bias*
_output_shapes	
:А*
dtype0
и
%lstm_28/lstm_cell_28/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*6
shared_name'%lstm_28/lstm_cell_28/recurrent_kernel
б
9lstm_28/lstm_cell_28/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_28/lstm_cell_28/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
У
lstm_28/lstm_cell_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_namelstm_28/lstm_cell_28/kernel
М
/lstm_28/lstm_cell_28/kernel/Read/ReadVariableOpReadVariableOplstm_28/lstm_cell_28/kernel*
_output_shapes
:	А*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:@*
dtype0
И
serving_default_lstm_28_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
ж
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_28_inputlstm_28/lstm_cell_28/kernellstm_28/lstm_cell_28/bias%lstm_28/lstm_cell_28/recurrent_kernellstm_29/lstm_cell_29/kernellstm_29/lstm_cell_29/bias%lstm_29/lstm_cell_29/recurrent_kerneldense_14/kerneldense_14/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_235395

NoOpNoOp
л?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ц>
value▄>B┘> B╥>
┴
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
_default_save_signature
*	&call_and_return_all_conditional_losses

__call__
	optimizer

signatures*
к
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__
cell

state_spec*
к
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__
cell

state_spec*
ж
	variables
trainable_variables
regularization_losses
 	keras_api
*!&call_and_return_all_conditional_losses
"__call__

#kernel
$bias*
<
%0
&1
'2
(3
)4
*5
#6
$7*
<
%0
&1
'2
(3
)4
*5
#6
$7*
* 
░
+layer_metrics

,layers
	variables
-non_trainable_variables
.layer_regularization_losses
trainable_variables
regularization_losses
/metrics

__call__
_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*

0trace_0* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
ф

9beta_1

:beta_2
	;decay
<learning_rate
=iter#mЗ$mИ%mЙ&mК'mЛ(mМ)mН*mО#vП$vР%vС&vТ'vУ(vФ)vХ*vЦ*

>serving_default* 

%0
&1
'2*

%0
&1
'2*
* 
Я
?layer_metrics

@states

Alayers
	variables
Bnon_trainable_variables
Clayer_regularization_losses
trainable_variables
regularization_losses
Dmetrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
╠
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
*Q&call_and_return_all_conditional_losses
R__call__
S
state_size

%kernel
&recurrent_kernel
'bias*
* 

(0
)1
*2*

(0
)1
*2*
* 
Я
Tlayer_metrics

Ustates

Vlayers
	variables
Wnon_trainable_variables
Xlayer_regularization_losses
trainable_variables
regularization_losses
Ymetrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ztrace_0
[trace_1
\trace_2
]trace_3* 
6
^trace_0
_trace_1
`trace_2
atrace_3* 
╠
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
*f&call_and_return_all_conditional_losses
g__call__
h
state_size

(kernel
)recurrent_kernel
*bias*
* 

#0
$1*

#0
$1*
* 
У
ilayer_metrics

jlayers
knon_trainable_variables
	variables
llayer_regularization_losses
trainable_variables
regularization_losses
mmetrics
"__call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_28/lstm_cell_28/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_28/lstm_cell_28/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_28/lstm_cell_28/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_29/lstm_cell_29/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_29/lstm_cell_29/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_29/lstm_cell_29/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 

p0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1
'2*

%0
&1
'2*
* 
У
qlayer_metrics

rlayers
snon_trainable_variables
M	variables
tlayer_regularization_losses
Ntrainable_variables
Oregularization_losses
umetrics
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1
*2*

(0
)1
*2*
* 
У
zlayer_metrics

{layers
|non_trainable_variables
b	variables
}layer_regularization_losses
ctrainable_variables
dregularization_losses
~metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

trace_0
Аtrace_1* 

Бtrace_0
Вtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Г	variables
Д	keras_api

Еtotal

Жcount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Е0
Ж1*

Г	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_28/lstm_cell_28/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_28/lstm_cell_28/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_28/lstm_cell_28/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_29/lstm_cell_29/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_29/lstm_cell_29/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_29/lstm_cell_29/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_28/lstm_cell_28/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_28/lstm_cell_28/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_28/lstm_cell_28/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_29/lstm_cell_29/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_29/lstm_cell_29/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_29/lstm_cell_29/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Г	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biaslstm_28/lstm_cell_28/kernel%lstm_28/lstm_cell_28/recurrent_kernellstm_28/lstm_cell_28/biaslstm_29/lstm_cell_29/kernel%lstm_29/lstm_cell_29/recurrent_kernellstm_29/lstm_cell_29/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense_14/kernel/mAdam/dense_14/bias/m"Adam/lstm_28/lstm_cell_28/kernel/m,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m Adam/lstm_28/lstm_cell_28/bias/m"Adam/lstm_29/lstm_cell_29/kernel/m,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m Adam/lstm_29/lstm_cell_29/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/v"Adam/lstm_28/lstm_cell_28/kernel/v,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v Adam/lstm_28/lstm_cell_28/bias/v"Adam/lstm_29/lstm_cell_29/kernel/v,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v Adam/lstm_29/lstm_cell_29/bias/vConst*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_239253
■
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biaslstm_28/lstm_cell_28/kernel%lstm_28/lstm_cell_28/recurrent_kernellstm_28/lstm_cell_28/biaslstm_29/lstm_cell_29/kernel%lstm_29/lstm_cell_29/recurrent_kernellstm_29/lstm_cell_29/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense_14/kernel/mAdam/dense_14/bias/m"Adam/lstm_28/lstm_cell_28/kernel/m,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m Adam/lstm_28/lstm_cell_28/bias/m"Adam/lstm_29/lstm_cell_29/kernel/m,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m Adam/lstm_29/lstm_cell_29/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/v"Adam/lstm_28/lstm_cell_28/kernel/v,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v Adam/lstm_28/lstm_cell_28/bias/v"Adam/lstm_29/lstm_cell_29/kernel/v,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v Adam/lstm_29/lstm_cell_29/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_239356■№5
с7
Ж
C__inference_lstm_28_layer_call_and_return_conditional_losses_233628

inputs&
lstm_cell_28_233547:	А"
lstm_cell_28_233549:	А'
lstm_cell_28_233551:
АА
identityИв$lstm_cell_28/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask°
$lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_28_233547lstm_cell_28_233549lstm_cell_28_233551*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233501n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╗
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_28_233547lstm_cell_28_233549lstm_cell_28_233551*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_233560*
condR
while_cond_233559*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  Аl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  Аu
NoOpNoOp%^lstm_cell_28/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_28/StatefulPartitionedCall$lstm_cell_28/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
█П
╛
lstm_29_while_body_236319,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3+
'lstm_29_while_lstm_29_strided_slice_1_0g
clstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0N
:lstm_29_while_lstm_cell_29_split_readvariableop_resource_0:
ААK
<lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0:	АG
4lstm_29_while_lstm_cell_29_readvariableop_resource_0:	@А
lstm_29_while_identity
lstm_29_while_identity_1
lstm_29_while_identity_2
lstm_29_while_identity_3
lstm_29_while_identity_4
lstm_29_while_identity_5)
%lstm_29_while_lstm_29_strided_slice_1e
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorL
8lstm_29_while_lstm_cell_29_split_readvariableop_resource:
ААI
:lstm_29_while_lstm_cell_29_split_1_readvariableop_resource:	АE
2lstm_29_while_lstm_cell_29_readvariableop_resource:	@АИв)lstm_29/while/lstm_cell_29/ReadVariableOpв+lstm_29/while/lstm_cell_29/ReadVariableOp_1в+lstm_29/while/lstm_cell_29/ReadVariableOp_2в+lstm_29/while/lstm_cell_29/ReadVariableOp_3в/lstm_29/while/lstm_cell_29/split/ReadVariableOpв1lstm_29/while/lstm_cell_29/split_1/ReadVariableOpР
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╧
1lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0lstm_29_while_placeholderHlstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0l
*lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :м
/lstm_29/while/lstm_cell_29/split/ReadVariableOpReadVariableOp:lstm_29_while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0є
 lstm_29/while/lstm_cell_29/splitSplit3lstm_29/while/lstm_cell_29/split/split_dim:output:07lstm_29/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split┬
!lstm_29/while/lstm_cell_29/MatMulMatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @─
#lstm_29/while/lstm_cell_29/MatMul_1MatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @─
#lstm_29/while/lstm_cell_29/MatMul_2MatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @─
#lstm_29/while/lstm_cell_29/MatMul_3MatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @n
,lstm_29/while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_29/while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp<lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0х
"lstm_29/while/lstm_cell_29/split_1Split5lstm_29/while/lstm_cell_29/split_1/split_dim:output:09lstm_29/while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split╣
"lstm_29/while/lstm_cell_29/BiasAddBiasAdd+lstm_29/while/lstm_cell_29/MatMul:product:0+lstm_29/while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @╜
$lstm_29/while/lstm_cell_29/BiasAdd_1BiasAdd-lstm_29/while/lstm_cell_29/MatMul_1:product:0+lstm_29/while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @╜
$lstm_29/while/lstm_cell_29/BiasAdd_2BiasAdd-lstm_29/while/lstm_cell_29/MatMul_2:product:0+lstm_29/while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @╜
$lstm_29/while/lstm_cell_29/BiasAdd_3BiasAdd-lstm_29/while/lstm_cell_29/MatMul_3:product:0+lstm_29/while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Я
)lstm_29/while/lstm_cell_29/ReadVariableOpReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0
.lstm_29/while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_29/while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Б
0lstm_29/while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Є
(lstm_29/while/lstm_cell_29/strided_sliceStridedSlice1lstm_29/while/lstm_cell_29/ReadVariableOp:value:07lstm_29/while/lstm_cell_29/strided_slice/stack:output:09lstm_29/while/lstm_cell_29/strided_slice/stack_1:output:09lstm_29/while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskп
#lstm_29/while/lstm_cell_29/MatMul_4MatMullstm_29_while_placeholder_21lstm_29/while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @╡
lstm_29/while/lstm_cell_29/addAddV2+lstm_29/while/lstm_cell_29/BiasAdd:output:0-lstm_29/while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @e
 lstm_29/while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_29/while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?ж
lstm_29/while/lstm_cell_29/MulMul"lstm_29/while/lstm_cell_29/add:z:0)lstm_29/while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @м
 lstm_29/while/lstm_cell_29/Add_1AddV2"lstm_29/while/lstm_cell_29/Mul:z:0+lstm_29/while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @w
2lstm_29/while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╨
0lstm_29/while/lstm_cell_29/clip_by_value/MinimumMinimum$lstm_29/while/lstm_cell_29/Add_1:z:0;lstm_29/while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @o
*lstm_29/while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╨
(lstm_29/while/lstm_cell_29/clip_by_valueMaximum4lstm_29/while/lstm_cell_29/clip_by_value/Minimum:z:03lstm_29/while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @б
+lstm_29/while/lstm_cell_29/ReadVariableOp_1ReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_29/while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   Г
2lstm_29/while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_29/while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_29/while/lstm_cell_29/strided_slice_1StridedSlice3lstm_29/while/lstm_cell_29/ReadVariableOp_1:value:09lstm_29/while/lstm_cell_29/strided_slice_1/stack:output:0;lstm_29/while/lstm_cell_29/strided_slice_1/stack_1:output:0;lstm_29/while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_29/while/lstm_cell_29/MatMul_5MatMullstm_29_while_placeholder_23lstm_29/while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @╣
 lstm_29/while/lstm_cell_29/add_2AddV2-lstm_29/while/lstm_cell_29/BiasAdd_1:output:0-lstm_29/while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @g
"lstm_29/while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_29/while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_29/while/lstm_cell_29/Mul_1Mul$lstm_29/while/lstm_cell_29/add_2:z:0+lstm_29/while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @о
 lstm_29/while/lstm_cell_29/Add_3AddV2$lstm_29/while/lstm_cell_29/Mul_1:z:0+lstm_29/while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @y
4lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_29/while/lstm_cell_29/clip_by_value_1/MinimumMinimum$lstm_29/while/lstm_cell_29/Add_3:z:0=lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_29/while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_29/while/lstm_cell_29/clip_by_value_1Maximum6lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum:z:05lstm_29/while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @ж
 lstm_29/while/lstm_cell_29/mul_2Mul.lstm_29/while/lstm_cell_29/clip_by_value_1:z:0lstm_29_while_placeholder_3*
T0*'
_output_shapes
:         @б
+lstm_29/while/lstm_cell_29/ReadVariableOp_2ReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_29/while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_29/while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_29/while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_29/while/lstm_cell_29/strided_slice_2StridedSlice3lstm_29/while/lstm_cell_29/ReadVariableOp_2:value:09lstm_29/while/lstm_cell_29/strided_slice_2/stack:output:0;lstm_29/while/lstm_cell_29/strided_slice_2/stack_1:output:0;lstm_29/while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_29/while/lstm_cell_29/MatMul_6MatMullstm_29_while_placeholder_23lstm_29/while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @╣
 lstm_29/while/lstm_cell_29/add_4AddV2-lstm_29/while/lstm_cell_29/BiasAdd_2:output:0-lstm_29/while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @
lstm_29/while/lstm_cell_29/ReluRelu$lstm_29/while/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @╢
 lstm_29/while/lstm_cell_29/mul_3Mul,lstm_29/while/lstm_cell_29/clip_by_value:z:0-lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @з
 lstm_29/while/lstm_cell_29/add_5AddV2$lstm_29/while/lstm_cell_29/mul_2:z:0$lstm_29/while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @б
+lstm_29/while/lstm_cell_29/ReadVariableOp_3ReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_29/while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_29/while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_29/while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_29/while/lstm_cell_29/strided_slice_3StridedSlice3lstm_29/while/lstm_cell_29/ReadVariableOp_3:value:09lstm_29/while/lstm_cell_29/strided_slice_3/stack:output:0;lstm_29/while/lstm_cell_29/strided_slice_3/stack_1:output:0;lstm_29/while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_29/while/lstm_cell_29/MatMul_7MatMullstm_29_while_placeholder_23lstm_29/while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @╣
 lstm_29/while/lstm_cell_29/add_6AddV2-lstm_29/while/lstm_cell_29/BiasAdd_3:output:0-lstm_29/while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @g
"lstm_29/while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_29/while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_29/while/lstm_cell_29/Mul_4Mul$lstm_29/while/lstm_cell_29/add_6:z:0+lstm_29/while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @о
 lstm_29/while/lstm_cell_29/Add_7AddV2$lstm_29/while/lstm_cell_29/Mul_4:z:0+lstm_29/while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @y
4lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_29/while/lstm_cell_29/clip_by_value_2/MinimumMinimum$lstm_29/while/lstm_cell_29/Add_7:z:0=lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_29/while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_29/while/lstm_cell_29/clip_by_value_2Maximum6lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum:z:05lstm_29/while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @Б
!lstm_29/while/lstm_cell_29/Relu_1Relu$lstm_29/while/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @║
 lstm_29/while/lstm_cell_29/mul_5Mul.lstm_29/while/lstm_cell_29/clip_by_value_2:z:0/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @х
2lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_29_while_placeholder_1lstm_29_while_placeholder$lstm_29/while/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_29/while/addAddV2lstm_29_while_placeholderlstm_29/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_29/while/add_1AddV2(lstm_29_while_lstm_29_while_loop_counterlstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_29/while/IdentityIdentitylstm_29/while/add_1:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: К
lstm_29/while/Identity_1Identity.lstm_29_while_lstm_29_while_maximum_iterations^lstm_29/while/NoOp*
T0*
_output_shapes
: q
lstm_29/while/Identity_2Identitylstm_29/while/add:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: Ю
lstm_29/while/Identity_3IdentityBlstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_29/while/NoOp*
T0*
_output_shapes
: С
lstm_29/while/Identity_4Identity$lstm_29/while/lstm_cell_29/mul_5:z:0^lstm_29/while/NoOp*
T0*'
_output_shapes
:         @С
lstm_29/while/Identity_5Identity$lstm_29/while/lstm_cell_29/add_5:z:0^lstm_29/while/NoOp*
T0*'
_output_shapes
:         @Ё
lstm_29/while/NoOpNoOp*^lstm_29/while/lstm_cell_29/ReadVariableOp,^lstm_29/while/lstm_cell_29/ReadVariableOp_1,^lstm_29/while/lstm_cell_29/ReadVariableOp_2,^lstm_29/while/lstm_cell_29/ReadVariableOp_30^lstm_29/while/lstm_cell_29/split/ReadVariableOp2^lstm_29/while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_29_while_identity_1!lstm_29/while/Identity_1:output:0"=
lstm_29_while_identity_2!lstm_29/while/Identity_2:output:0"=
lstm_29_while_identity_3!lstm_29/while/Identity_3:output:0"=
lstm_29_while_identity_4!lstm_29/while/Identity_4:output:0"=
lstm_29_while_identity_5!lstm_29/while/Identity_5:output:0"9
lstm_29_while_identitylstm_29/while/Identity:output:0"P
%lstm_29_while_lstm_29_strided_slice_1'lstm_29_while_lstm_29_strided_slice_1_0"j
2lstm_29_while_lstm_cell_29_readvariableop_resource4lstm_29_while_lstm_cell_29_readvariableop_resource_0"z
:lstm_29_while_lstm_cell_29_split_1_readvariableop_resource<lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0"v
8lstm_29_while_lstm_cell_29_split_readvariableop_resource:lstm_29_while_lstm_cell_29_split_readvariableop_resource_0"╚
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2Z
+lstm_29/while/lstm_cell_29/ReadVariableOp_1+lstm_29/while/lstm_cell_29/ReadVariableOp_12Z
+lstm_29/while/lstm_cell_29/ReadVariableOp_2+lstm_29/while/lstm_cell_29/ReadVariableOp_22Z
+lstm_29/while/lstm_cell_29/ReadVariableOp_3+lstm_29/while/lstm_cell_29/ReadVariableOp_32V
)lstm_29/while/lstm_cell_29/ReadVariableOp)lstm_29/while/lstm_cell_29/ReadVariableOp2b
/lstm_29/while/lstm_cell_29/split/ReadVariableOp/lstm_29/while/lstm_cell_29/split/ReadVariableOp2f
1lstm_29/while/lstm_cell_29/split_1/ReadVariableOp1lstm_29/while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_29/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_29/while/loop_counter
иИ
ш
C__inference_lstm_29_layer_call_and_return_conditional_losses_234624

inputs>
*lstm_cell_29_split_readvariableop_resource:
АА;
,lstm_cell_29_split_1_readvariableop_resource:	А7
$lstm_cell_29_readvariableop_resource:	@А
identityИвlstm_cell_29/ReadVariableOpвlstm_cell_29/ReadVariableOp_1вlstm_cell_29/ReadVariableOp_2вlstm_cell_29/ReadVariableOp_3в!lstm_cell_29/split/ReadVariableOpв#lstm_cell_29/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask^
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_29/split/ReadVariableOpReadVariableOp*lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0)lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_29/split_1/ReadVariableOpReadVariableOp,lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_29/split_1Split'lstm_cell_29/split_1/split_dim:output:0+lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_29/BiasAddBiasAddlstm_cell_29/MatMul:product:0lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_1BiasAddlstm_cell_29/MatMul_1:product:0lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_2BiasAddlstm_cell_29/MatMul_2:product:0lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_3BiasAddlstm_cell_29/MatMul_3:product:0lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_29/ReadVariableOpReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_29/strided_sliceStridedSlice#lstm_cell_29/ReadVariableOp:value:0)lstm_cell_29/strided_slice/stack:output:0+lstm_cell_29/strided_slice/stack_1:output:0+lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_29/MatMul_4MatMulzeros:output:0#lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_29/addAddV2lstm_cell_29/BiasAdd:output:0lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_29/MulMullstm_cell_29/add:z:0lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_29/Add_1AddV2lstm_cell_29/Mul:z:0lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_29/clip_by_value/MinimumMinimumlstm_cell_29/Add_1:z:0-lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_29/clip_by_valueMaximum&lstm_cell_29/clip_by_value/Minimum:z:0%lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_1ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_1StridedSlice%lstm_cell_29/ReadVariableOp_1:value:0+lstm_cell_29/strided_slice_1/stack:output:0-lstm_cell_29/strided_slice_1/stack_1:output:0-lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_5MatMulzeros:output:0%lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_2AddV2lstm_cell_29/BiasAdd_1:output:0lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_1Mullstm_cell_29/add_2:z:0lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_3AddV2lstm_cell_29/Mul_1:z:0lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_cell_29/Add_3:z:0/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_1Maximum(lstm_cell_29/clip_by_value_1/Minimum:z:0'lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_29/mul_2Mul lstm_cell_29/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_2ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_2StridedSlice%lstm_cell_29/ReadVariableOp_2:value:0+lstm_cell_29/strided_slice_2/stack:output:0-lstm_cell_29/strided_slice_2/stack_1:output:0-lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_6MatMulzeros:output:0%lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_4AddV2lstm_cell_29/BiasAdd_2:output:0lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/ReluRelulstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_29/mul_3Mullstm_cell_29/clip_by_value:z:0lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_29/add_5AddV2lstm_cell_29/mul_2:z:0lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_3ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_3StridedSlice%lstm_cell_29/ReadVariableOp_3:value:0+lstm_cell_29/strided_slice_3/stack:output:0-lstm_cell_29/strided_slice_3/stack_1:output:0-lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_7MatMulzeros:output:0%lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_6AddV2lstm_cell_29/BiasAdd_3:output:0lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_4Mullstm_cell_29/add_6:z:0lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_7AddV2lstm_cell_29/Mul_4:z:0lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_cell_29/Add_7:z:0/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_2Maximum(lstm_cell_29/clip_by_value_2/Minimum:z:0'lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_29/Relu_1Relulstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_29/mul_5Mul lstm_cell_29/clip_by_value_2:z:0!lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_29_split_readvariableop_resource,lstm_cell_29_split_1_readvariableop_resource$lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_234484*
condR
while_cond_234483*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_29/ReadVariableOp^lstm_cell_29/ReadVariableOp_1^lstm_cell_29/ReadVariableOp_2^lstm_cell_29/ReadVariableOp_3"^lstm_cell_29/split/ReadVariableOp$^lstm_cell_29/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         А: : : 2>
lstm_cell_29/ReadVariableOp_1lstm_cell_29/ReadVariableOp_12>
lstm_cell_29/ReadVariableOp_2lstm_cell_29/ReadVariableOp_22>
lstm_cell_29/ReadVariableOp_3lstm_cell_29/ReadVariableOp_32:
lstm_cell_29/ReadVariableOplstm_cell_29/ReadVariableOp2F
!lstm_cell_29/split/ReadVariableOp!lstm_cell_29/split/ReadVariableOp2J
#lstm_cell_29/split_1/ReadVariableOp#lstm_cell_29/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
▌И
ъ
C__inference_lstm_29_layer_call_and_return_conditional_losses_238089
inputs_0>
*lstm_cell_29_split_readvariableop_resource:
АА;
,lstm_cell_29_split_1_readvariableop_resource:	А7
$lstm_cell_29_readvariableop_resource:	@А
identityИвlstm_cell_29/ReadVariableOpвlstm_cell_29/ReadVariableOp_1вlstm_cell_29/ReadVariableOp_2вlstm_cell_29/ReadVariableOp_3в!lstm_cell_29/split/ReadVariableOpв#lstm_cell_29/split_1/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask^
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_29/split/ReadVariableOpReadVariableOp*lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0)lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_29/split_1/ReadVariableOpReadVariableOp,lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_29/split_1Split'lstm_cell_29/split_1/split_dim:output:0+lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_29/BiasAddBiasAddlstm_cell_29/MatMul:product:0lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_1BiasAddlstm_cell_29/MatMul_1:product:0lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_2BiasAddlstm_cell_29/MatMul_2:product:0lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_3BiasAddlstm_cell_29/MatMul_3:product:0lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_29/ReadVariableOpReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_29/strided_sliceStridedSlice#lstm_cell_29/ReadVariableOp:value:0)lstm_cell_29/strided_slice/stack:output:0+lstm_cell_29/strided_slice/stack_1:output:0+lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_29/MatMul_4MatMulzeros:output:0#lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_29/addAddV2lstm_cell_29/BiasAdd:output:0lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_29/MulMullstm_cell_29/add:z:0lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_29/Add_1AddV2lstm_cell_29/Mul:z:0lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_29/clip_by_value/MinimumMinimumlstm_cell_29/Add_1:z:0-lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_29/clip_by_valueMaximum&lstm_cell_29/clip_by_value/Minimum:z:0%lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_1ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_1StridedSlice%lstm_cell_29/ReadVariableOp_1:value:0+lstm_cell_29/strided_slice_1/stack:output:0-lstm_cell_29/strided_slice_1/stack_1:output:0-lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_5MatMulzeros:output:0%lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_2AddV2lstm_cell_29/BiasAdd_1:output:0lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_1Mullstm_cell_29/add_2:z:0lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_3AddV2lstm_cell_29/Mul_1:z:0lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_cell_29/Add_3:z:0/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_1Maximum(lstm_cell_29/clip_by_value_1/Minimum:z:0'lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_29/mul_2Mul lstm_cell_29/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_2ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_2StridedSlice%lstm_cell_29/ReadVariableOp_2:value:0+lstm_cell_29/strided_slice_2/stack:output:0-lstm_cell_29/strided_slice_2/stack_1:output:0-lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_6MatMulzeros:output:0%lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_4AddV2lstm_cell_29/BiasAdd_2:output:0lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/ReluRelulstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_29/mul_3Mullstm_cell_29/clip_by_value:z:0lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_29/add_5AddV2lstm_cell_29/mul_2:z:0lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_3ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_3StridedSlice%lstm_cell_29/ReadVariableOp_3:value:0+lstm_cell_29/strided_slice_3/stack:output:0-lstm_cell_29/strided_slice_3/stack_1:output:0-lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_7MatMulzeros:output:0%lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_6AddV2lstm_cell_29/BiasAdd_3:output:0lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_4Mullstm_cell_29/add_6:z:0lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_7AddV2lstm_cell_29/Mul_4:z:0lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_cell_29/Add_7:z:0/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_2Maximum(lstm_cell_29/clip_by_value_2/Minimum:z:0'lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_29/Relu_1Relulstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_29/mul_5Mul lstm_cell_29/clip_by_value_2:z:0!lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_29_split_readvariableop_resource,lstm_cell_29_split_1_readvariableop_resource$lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_237949*
condR
while_cond_237948*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_29/ReadVariableOp^lstm_cell_29/ReadVariableOp_1^lstm_cell_29/ReadVariableOp_2^lstm_cell_29/ReadVariableOp_3"^lstm_cell_29/split/ReadVariableOp$^lstm_cell_29/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2>
lstm_cell_29/ReadVariableOp_1lstm_cell_29/ReadVariableOp_12>
lstm_cell_29/ReadVariableOp_2lstm_cell_29/ReadVariableOp_22>
lstm_cell_29/ReadVariableOp_3lstm_cell_29/ReadVariableOp_32:
lstm_cell_29/ReadVariableOplstm_cell_29/ReadVariableOp2F
!lstm_cell_29/split/ReadVariableOp!lstm_cell_29/split/ReadVariableOp2J
#lstm_cell_29/split_1/ReadVariableOp#lstm_cell_29/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  А
"
_user_specified_name
inputs_0
Ы	
├
while_cond_233559
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_233559___redundant_placeholder04
0while_while_cond_233559___redundant_placeholder14
0while_while_cond_233559___redundant_placeholder24
0while_while_cond_233559___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И
у
lstm_29_while_cond_236318,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3.
*lstm_29_while_less_lstm_29_strided_slice_1D
@lstm_29_while_lstm_29_while_cond_236318___redundant_placeholder0D
@lstm_29_while_lstm_29_while_cond_236318___redundant_placeholder1D
@lstm_29_while_lstm_29_while_cond_236318___redundant_placeholder2D
@lstm_29_while_lstm_29_while_cond_236318___redundant_placeholder3
lstm_29_while_identity
В
lstm_29/while/LessLesslstm_29_while_placeholder*lstm_29_while_less_lstm_29_strided_slice_1*
T0*
_output_shapes
: [
lstm_29/while/IdentityIdentitylstm_29/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_29_while_identitylstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_29/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_29/while/loop_counter
гЙ
·
"__inference__traced_restore_239356
file_prefix2
 assignvariableop_dense_14_kernel:@.
 assignvariableop_1_dense_14_bias:A
.assignvariableop_2_lstm_28_lstm_cell_28_kernel:	АL
8assignvariableop_3_lstm_28_lstm_cell_28_recurrent_kernel:
АА;
,assignvariableop_4_lstm_28_lstm_cell_28_bias:	АB
.assignvariableop_5_lstm_29_lstm_cell_29_kernel:
ААK
8assignvariableop_6_lstm_29_lstm_cell_29_recurrent_kernel:	@А;
,assignvariableop_7_lstm_29_lstm_cell_29_bias:	А#
assignvariableop_8_beta_1: #
assignvariableop_9_beta_2: #
assignvariableop_10_decay: +
!assignvariableop_11_learning_rate: '
assignvariableop_12_adam_iter:	 #
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_14_kernel_m:@6
(assignvariableop_16_adam_dense_14_bias_m:I
6assignvariableop_17_adam_lstm_28_lstm_cell_28_kernel_m:	АT
@assignvariableop_18_adam_lstm_28_lstm_cell_28_recurrent_kernel_m:
ААC
4assignvariableop_19_adam_lstm_28_lstm_cell_28_bias_m:	АJ
6assignvariableop_20_adam_lstm_29_lstm_cell_29_kernel_m:
ААS
@assignvariableop_21_adam_lstm_29_lstm_cell_29_recurrent_kernel_m:	@АC
4assignvariableop_22_adam_lstm_29_lstm_cell_29_bias_m:	А<
*assignvariableop_23_adam_dense_14_kernel_v:@6
(assignvariableop_24_adam_dense_14_bias_v:I
6assignvariableop_25_adam_lstm_28_lstm_cell_28_kernel_v:	АT
@assignvariableop_26_adam_lstm_28_lstm_cell_28_recurrent_kernel_v:
ААC
4assignvariableop_27_adam_lstm_28_lstm_cell_28_bias_v:	АJ
6assignvariableop_28_adam_lstm_29_lstm_cell_29_kernel_v:
ААS
@assignvariableop_29_adam_lstm_29_lstm_cell_29_recurrent_kernel_v:	@АC
4assignvariableop_30_adam_lstm_29_lstm_cell_29_bias_v:	А
identity_32ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╕
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*▐
value╘B╤ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH░
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┴
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_28_lstm_cell_28_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_28_lstm_cell_28_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_28_lstm_cell_28_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_29_lstm_cell_29_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_29_lstm_cell_29_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_29_lstm_cell_29_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_2Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_14_kernel_mIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_14_bias_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_28_lstm_cell_28_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_28_lstm_cell_28_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_28_lstm_cell_28_bias_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_29_lstm_cell_29_kernel_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_29_lstm_cell_29_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_29_lstm_cell_29_bias_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_14_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_14_bias_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_28_lstm_cell_28_kernel_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_28_lstm_cell_28_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_28_lstm_cell_28_bias_vIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_29_lstm_cell_29_kernel_vIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_29_lstm_cell_29_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_29_lstm_cell_29_bias_vIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ∙
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ц
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╥7
Ж
C__inference_lstm_29_layer_call_and_return_conditional_losses_234090

inputs'
lstm_cell_29_234009:
АА"
lstm_cell_29_234011:	А&
lstm_cell_29_234013:	@А
identityИв$lstm_cell_29/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskї
$lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_29_234009lstm_cell_29_234011lstm_cell_29_234013*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233963n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╖
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_29_234009lstm_cell_29_234011lstm_cell_29_234013*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_234022*
condR
while_cond_234021*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @u
NoOpNoOp%^lstm_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2L
$lstm_cell_29/StatefulPartitionedCall$lstm_cell_29/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
иИ
ш
C__inference_lstm_29_layer_call_and_return_conditional_losses_234947

inputs>
*lstm_cell_29_split_readvariableop_resource:
АА;
,lstm_cell_29_split_1_readvariableop_resource:	А7
$lstm_cell_29_readvariableop_resource:	@А
identityИвlstm_cell_29/ReadVariableOpвlstm_cell_29/ReadVariableOp_1вlstm_cell_29/ReadVariableOp_2вlstm_cell_29/ReadVariableOp_3в!lstm_cell_29/split/ReadVariableOpв#lstm_cell_29/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask^
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_29/split/ReadVariableOpReadVariableOp*lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0)lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_29/split_1/ReadVariableOpReadVariableOp,lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_29/split_1Split'lstm_cell_29/split_1/split_dim:output:0+lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_29/BiasAddBiasAddlstm_cell_29/MatMul:product:0lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_1BiasAddlstm_cell_29/MatMul_1:product:0lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_2BiasAddlstm_cell_29/MatMul_2:product:0lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_3BiasAddlstm_cell_29/MatMul_3:product:0lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_29/ReadVariableOpReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_29/strided_sliceStridedSlice#lstm_cell_29/ReadVariableOp:value:0)lstm_cell_29/strided_slice/stack:output:0+lstm_cell_29/strided_slice/stack_1:output:0+lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_29/MatMul_4MatMulzeros:output:0#lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_29/addAddV2lstm_cell_29/BiasAdd:output:0lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_29/MulMullstm_cell_29/add:z:0lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_29/Add_1AddV2lstm_cell_29/Mul:z:0lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_29/clip_by_value/MinimumMinimumlstm_cell_29/Add_1:z:0-lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_29/clip_by_valueMaximum&lstm_cell_29/clip_by_value/Minimum:z:0%lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_1ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_1StridedSlice%lstm_cell_29/ReadVariableOp_1:value:0+lstm_cell_29/strided_slice_1/stack:output:0-lstm_cell_29/strided_slice_1/stack_1:output:0-lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_5MatMulzeros:output:0%lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_2AddV2lstm_cell_29/BiasAdd_1:output:0lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_1Mullstm_cell_29/add_2:z:0lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_3AddV2lstm_cell_29/Mul_1:z:0lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_cell_29/Add_3:z:0/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_1Maximum(lstm_cell_29/clip_by_value_1/Minimum:z:0'lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_29/mul_2Mul lstm_cell_29/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_2ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_2StridedSlice%lstm_cell_29/ReadVariableOp_2:value:0+lstm_cell_29/strided_slice_2/stack:output:0-lstm_cell_29/strided_slice_2/stack_1:output:0-lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_6MatMulzeros:output:0%lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_4AddV2lstm_cell_29/BiasAdd_2:output:0lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/ReluRelulstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_29/mul_3Mullstm_cell_29/clip_by_value:z:0lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_29/add_5AddV2lstm_cell_29/mul_2:z:0lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_3ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_3StridedSlice%lstm_cell_29/ReadVariableOp_3:value:0+lstm_cell_29/strided_slice_3/stack:output:0-lstm_cell_29/strided_slice_3/stack_1:output:0-lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_7MatMulzeros:output:0%lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_6AddV2lstm_cell_29/BiasAdd_3:output:0lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_4Mullstm_cell_29/add_6:z:0lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_7AddV2lstm_cell_29/Mul_4:z:0lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_cell_29/Add_7:z:0/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_2Maximum(lstm_cell_29/clip_by_value_2/Minimum:z:0'lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_29/Relu_1Relulstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_29/mul_5Mul lstm_cell_29/clip_by_value_2:z:0!lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_29_split_readvariableop_resource,lstm_cell_29_split_1_readvariableop_resource$lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_234807*
condR
while_cond_234806*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_29/ReadVariableOp^lstm_cell_29/ReadVariableOp_1^lstm_cell_29/ReadVariableOp_2^lstm_cell_29/ReadVariableOp_3"^lstm_cell_29/split/ReadVariableOp$^lstm_cell_29/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         А: : : 2>
lstm_cell_29/ReadVariableOp_1lstm_cell_29/ReadVariableOp_12>
lstm_cell_29/ReadVariableOp_2lstm_cell_29/ReadVariableOp_22>
lstm_cell_29/ReadVariableOp_3lstm_cell_29/ReadVariableOp_32:
lstm_cell_29/ReadVariableOplstm_cell_29/ReadVariableOp2F
!lstm_cell_29/split/ReadVariableOp!lstm_cell_29/split/ReadVariableOp2J
#lstm_cell_29/split_1/ReadVariableOp#lstm_cell_29/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
МK
к
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233501

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А+
readvariableop_resource:
АА
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:         А]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:         А]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:         А]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:         АS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:         Аm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:         Аm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:         Аh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      э
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:         Аe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:         АJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:         А\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:         А\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         АT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:         Аi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:         АL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:         А^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:         Аj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:         Аi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:         АJ
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:         Аf
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:         АW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:         Аi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:         АL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:         А^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АL
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:         Аj
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:         А└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         А:         А: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:PL
(
_output_shapes
:         А
 
_user_specified_namestates:PL
(
_output_shapes
:         А
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╘J
к
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233761

inputs

states
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:         @\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:         @\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:         @\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ы
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskd
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:         @[
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:         @\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:         @]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:         @e
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*'
_output_shapes
:         @V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:         @]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_5:z:0*
T0*'
_output_shapes
:         @i
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*'
_output_shapes
:         @X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:         @└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         @:         @: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▌И
ш
C__inference_lstm_28_layer_call_and_return_conditional_losses_237533

inputs=
*lstm_cell_28_split_readvariableop_resource:	А;
,lstm_cell_28_split_1_readvariableop_resource:	А8
$lstm_cell_28_readvariableop_resource:
АА
identityИвlstm_cell_28/ReadVariableOpвlstm_cell_28/ReadVariableOp_1вlstm_cell_28/ReadVariableOp_2вlstm_cell_28/ReadVariableOp_3в!lstm_cell_28/split/ReadVariableOpв#lstm_cell_28/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask^
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_28/split/ReadVariableOpReadVariableOp*lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0)lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_28/split_1/ReadVariableOpReadVariableOp,lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_28/split_1Split'lstm_cell_28/split_1/split_dim:output:0+lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_28/BiasAddBiasAddlstm_cell_28/MatMul:product:0lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_1BiasAddlstm_cell_28/MatMul_1:product:0lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_2BiasAddlstm_cell_28/MatMul_2:product:0lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_3BiasAddlstm_cell_28/MatMul_3:product:0lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_28/ReadVariableOpReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_28/strided_sliceStridedSlice#lstm_cell_28/ReadVariableOp:value:0)lstm_cell_28/strided_slice/stack:output:0+lstm_cell_28/strided_slice/stack_1:output:0+lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_28/MatMul_4MatMulzeros:output:0#lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_28/addAddV2lstm_cell_28/BiasAdd:output:0lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_28/MulMullstm_cell_28/add:z:0lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_28/Add_1AddV2lstm_cell_28/Mul:z:0lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_28/clip_by_value/MinimumMinimumlstm_cell_28/Add_1:z:0-lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_28/clip_by_valueMaximum&lstm_cell_28/clip_by_value/Minimum:z:0%lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_1ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_1StridedSlice%lstm_cell_28/ReadVariableOp_1:value:0+lstm_cell_28/strided_slice_1/stack:output:0-lstm_cell_28/strided_slice_1/stack_1:output:0-lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_5MatMulzeros:output:0%lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_2AddV2lstm_cell_28/BiasAdd_1:output:0lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_1Mullstm_cell_28/add_2:z:0lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_3AddV2lstm_cell_28/Mul_1:z:0lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_cell_28/Add_3:z:0/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_1Maximum(lstm_cell_28/clip_by_value_1/Minimum:z:0'lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_28/mul_2Mul lstm_cell_28/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_2ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_2StridedSlice%lstm_cell_28/ReadVariableOp_2:value:0+lstm_cell_28/strided_slice_2/stack:output:0-lstm_cell_28/strided_slice_2/stack_1:output:0-lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_6MatMulzeros:output:0%lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_4AddV2lstm_cell_28/BiasAdd_2:output:0lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_28/ReluRelulstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_28/mul_3Mullstm_cell_28/clip_by_value:z:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_28/add_5AddV2lstm_cell_28/mul_2:z:0lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_3ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_3StridedSlice%lstm_cell_28/ReadVariableOp_3:value:0+lstm_cell_28/strided_slice_3/stack:output:0-lstm_cell_28/strided_slice_3/stack_1:output:0-lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_7MatMulzeros:output:0%lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_6AddV2lstm_cell_28/BiasAdd_3:output:0lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_4Mullstm_cell_28/add_6:z:0lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_7AddV2lstm_cell_28/Mul_4:z:0lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_cell_28/Add_7:z:0/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_2Maximum(lstm_cell_28/clip_by_value_2/Minimum:z:0'lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_28/Relu_1Relulstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_28/mul_5Mul lstm_cell_28/clip_by_value_2:z:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : №
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_28_split_readvariableop_resource,lstm_cell_28_split_1_readvariableop_resource$lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_237393*
condR
while_cond_237392*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         АЦ
NoOpNoOp^lstm_cell_28/ReadVariableOp^lstm_cell_28/ReadVariableOp_1^lstm_cell_28/ReadVariableOp_2^lstm_cell_28/ReadVariableOp_3"^lstm_cell_28/split/ReadVariableOp$^lstm_cell_28/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2>
lstm_cell_28/ReadVariableOp_1lstm_cell_28/ReadVariableOp_12>
lstm_cell_28/ReadVariableOp_2lstm_cell_28/ReadVariableOp_22>
lstm_cell_28/ReadVariableOp_3lstm_cell_28/ReadVariableOp_32:
lstm_cell_28/ReadVariableOplstm_cell_28/ReadVariableOp2F
!lstm_cell_28/split/ReadVariableOp!lstm_cell_28/split/ReadVariableOp2J
#lstm_cell_28/split_1/ReadVariableOp#lstm_cell_28/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
МK
к
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233299

inputs

states
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А+
readvariableop_resource:
АА
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:         А]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:         А]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:         А]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:         АS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:         Аm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:         Аm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:         Аh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      э
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:         Аe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:         АJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:         А\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:         А\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         АT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:         Аi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:         АL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:         А^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:         Аj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:         Аi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:         АJ
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:         Аf
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:         АW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:         Аi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:         АL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:         А^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АL
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:         Аj
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:         А└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         А:         А: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:PL
(
_output_shapes
:         А
 
_user_specified_namestates:PL
(
_output_shapes
:         А
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▄	
╔
.__inference_sequential_14_layer_call_fn_235437

inputs
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:
АА
	unknown_3:	А
	unknown_4:	@А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_14_layer_call_and_return_conditional_losses_235280o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ЫЙ
ъ
C__inference_lstm_28_layer_call_and_return_conditional_losses_237021
inputs_0=
*lstm_cell_28_split_readvariableop_resource:	А;
,lstm_cell_28_split_1_readvariableop_resource:	А8
$lstm_cell_28_readvariableop_resource:
АА
identityИвlstm_cell_28/ReadVariableOpвlstm_cell_28/ReadVariableOp_1вlstm_cell_28/ReadVariableOp_2вlstm_cell_28/ReadVariableOp_3в!lstm_cell_28/split/ReadVariableOpв#lstm_cell_28/split_1/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask^
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_28/split/ReadVariableOpReadVariableOp*lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0)lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_28/split_1/ReadVariableOpReadVariableOp,lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_28/split_1Split'lstm_cell_28/split_1/split_dim:output:0+lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_28/BiasAddBiasAddlstm_cell_28/MatMul:product:0lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_1BiasAddlstm_cell_28/MatMul_1:product:0lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_2BiasAddlstm_cell_28/MatMul_2:product:0lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_3BiasAddlstm_cell_28/MatMul_3:product:0lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_28/ReadVariableOpReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_28/strided_sliceStridedSlice#lstm_cell_28/ReadVariableOp:value:0)lstm_cell_28/strided_slice/stack:output:0+lstm_cell_28/strided_slice/stack_1:output:0+lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_28/MatMul_4MatMulzeros:output:0#lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_28/addAddV2lstm_cell_28/BiasAdd:output:0lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_28/MulMullstm_cell_28/add:z:0lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_28/Add_1AddV2lstm_cell_28/Mul:z:0lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_28/clip_by_value/MinimumMinimumlstm_cell_28/Add_1:z:0-lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_28/clip_by_valueMaximum&lstm_cell_28/clip_by_value/Minimum:z:0%lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_1ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_1StridedSlice%lstm_cell_28/ReadVariableOp_1:value:0+lstm_cell_28/strided_slice_1/stack:output:0-lstm_cell_28/strided_slice_1/stack_1:output:0-lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_5MatMulzeros:output:0%lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_2AddV2lstm_cell_28/BiasAdd_1:output:0lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_1Mullstm_cell_28/add_2:z:0lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_3AddV2lstm_cell_28/Mul_1:z:0lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_cell_28/Add_3:z:0/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_1Maximum(lstm_cell_28/clip_by_value_1/Minimum:z:0'lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_28/mul_2Mul lstm_cell_28/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_2ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_2StridedSlice%lstm_cell_28/ReadVariableOp_2:value:0+lstm_cell_28/strided_slice_2/stack:output:0-lstm_cell_28/strided_slice_2/stack_1:output:0-lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_6MatMulzeros:output:0%lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_4AddV2lstm_cell_28/BiasAdd_2:output:0lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_28/ReluRelulstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_28/mul_3Mullstm_cell_28/clip_by_value:z:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_28/add_5AddV2lstm_cell_28/mul_2:z:0lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_3ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_3StridedSlice%lstm_cell_28/ReadVariableOp_3:value:0+lstm_cell_28/strided_slice_3/stack:output:0-lstm_cell_28/strided_slice_3/stack_1:output:0-lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_7MatMulzeros:output:0%lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_6AddV2lstm_cell_28/BiasAdd_3:output:0lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_4Mullstm_cell_28/add_6:z:0lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_7AddV2lstm_cell_28/Mul_4:z:0lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_cell_28/Add_7:z:0/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_2Maximum(lstm_cell_28/clip_by_value_2/Minimum:z:0'lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_28/Relu_1Relulstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_28/mul_5Mul lstm_cell_28/clip_by_value_2:z:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : №
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_28_split_readvariableop_resource,lstm_cell_28_split_1_readvariableop_resource$lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_236881*
condR
while_cond_236880*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  Аl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  АЦ
NoOpNoOp^lstm_cell_28/ReadVariableOp^lstm_cell_28/ReadVariableOp_1^lstm_cell_28/ReadVariableOp_2^lstm_cell_28/ReadVariableOp_3"^lstm_cell_28/split/ReadVariableOp$^lstm_cell_28/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2>
lstm_cell_28/ReadVariableOp_1lstm_cell_28/ReadVariableOp_12>
lstm_cell_28/ReadVariableOp_2lstm_cell_28/ReadVariableOp_22>
lstm_cell_28/ReadVariableOp_3lstm_cell_28/ReadVariableOp_32:
lstm_cell_28/ReadVariableOplstm_cell_28/ReadVariableOp2F
!lstm_cell_28/split/ReadVariableOp!lstm_cell_28/split/ReadVariableOp2J
#lstm_cell_28/split_1/ReadVariableOp#lstm_cell_28/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Пъ
Н
__inference__traced_save_239253
file_prefix8
&read_disablecopyonread_dense_14_kernel:@4
&read_1_disablecopyonread_dense_14_bias:G
4read_2_disablecopyonread_lstm_28_lstm_cell_28_kernel:	АR
>read_3_disablecopyonread_lstm_28_lstm_cell_28_recurrent_kernel:
ААA
2read_4_disablecopyonread_lstm_28_lstm_cell_28_bias:	АH
4read_5_disablecopyonread_lstm_29_lstm_cell_29_kernel:
ААQ
>read_6_disablecopyonread_lstm_29_lstm_cell_29_recurrent_kernel:	@АA
2read_7_disablecopyonread_lstm_29_lstm_cell_29_bias:	А)
read_8_disablecopyonread_beta_1: )
read_9_disablecopyonread_beta_2: )
read_10_disablecopyonread_decay: 1
'read_11_disablecopyonread_learning_rate: -
#read_12_disablecopyonread_adam_iter:	 )
read_13_disablecopyonread_total: )
read_14_disablecopyonread_count: B
0read_15_disablecopyonread_adam_dense_14_kernel_m:@<
.read_16_disablecopyonread_adam_dense_14_bias_m:O
<read_17_disablecopyonread_adam_lstm_28_lstm_cell_28_kernel_m:	АZ
Fread_18_disablecopyonread_adam_lstm_28_lstm_cell_28_recurrent_kernel_m:
ААI
:read_19_disablecopyonread_adam_lstm_28_lstm_cell_28_bias_m:	АP
<read_20_disablecopyonread_adam_lstm_29_lstm_cell_29_kernel_m:
ААY
Fread_21_disablecopyonread_adam_lstm_29_lstm_cell_29_recurrent_kernel_m:	@АI
:read_22_disablecopyonread_adam_lstm_29_lstm_cell_29_bias_m:	АB
0read_23_disablecopyonread_adam_dense_14_kernel_v:@<
.read_24_disablecopyonread_adam_dense_14_bias_v:O
<read_25_disablecopyonread_adam_lstm_28_lstm_cell_28_kernel_v:	АZ
Fread_26_disablecopyonread_adam_lstm_28_lstm_cell_28_recurrent_kernel_v:
ААI
:read_27_disablecopyonread_adam_lstm_28_lstm_cell_28_bias_v:	АP
<read_28_disablecopyonread_adam_lstm_29_lstm_cell_29_kernel_v:
ААY
Fread_29_disablecopyonread_adam_lstm_29_lstm_cell_29_recurrent_kernel_v:	@АI
:read_30_disablecopyonread_adam_lstm_29_lstm_cell_29_bias_v:	А
savev2_const
identity_63ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_14_kernel"/device:CPU:0*
_output_shapes
 в
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_14_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_14_bias"/device:CPU:0*
_output_shapes
 в
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_14_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:И
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_lstm_28_lstm_cell_28_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_lstm_28_lstm_cell_28_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	АТ
Read_3/DisableCopyOnReadDisableCopyOnRead>read_3_disablecopyonread_lstm_28_lstm_cell_28_recurrent_kernel"/device:CPU:0*
_output_shapes
 └
Read_3/ReadVariableOpReadVariableOp>read_3_disablecopyonread_lstm_28_lstm_cell_28_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЖ
Read_4/DisableCopyOnReadDisableCopyOnRead2read_4_disablecopyonread_lstm_28_lstm_cell_28_bias"/device:CPU:0*
_output_shapes
 п
Read_4/ReadVariableOpReadVariableOp2read_4_disablecopyonread_lstm_28_lstm_cell_28_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:АИ
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_lstm_29_lstm_cell_29_kernel"/device:CPU:0*
_output_shapes
 ╢
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_lstm_29_lstm_cell_29_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААТ
Read_6/DisableCopyOnReadDisableCopyOnRead>read_6_disablecopyonread_lstm_29_lstm_cell_29_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_6/ReadVariableOpReadVariableOp>read_6_disablecopyonread_lstm_29_lstm_cell_29_recurrent_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@А*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@Аf
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	@АЖ
Read_7/DisableCopyOnReadDisableCopyOnRead2read_7_disablecopyonread_lstm_29_lstm_cell_29_bias"/device:CPU:0*
_output_shapes
 п
Read_7/ReadVariableOpReadVariableOp2read_7_disablecopyonread_lstm_29_lstm_cell_29_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аs
Read_8/DisableCopyOnReadDisableCopyOnReadread_8_disablecopyonread_beta_1"/device:CPU:0*
_output_shapes
 Ч
Read_8/ReadVariableOpReadVariableOpread_8_disablecopyonread_beta_1^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: s
Read_9/DisableCopyOnReadDisableCopyOnReadread_9_disablecopyonread_beta_2"/device:CPU:0*
_output_shapes
 Ч
Read_9/ReadVariableOpReadVariableOpread_9_disablecopyonread_beta_2^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_10/DisableCopyOnReadDisableCopyOnReadread_10_disablecopyonread_decay"/device:CPU:0*
_output_shapes
 Щ
Read_10/ReadVariableOpReadVariableOpread_10_disablecopyonread_decay^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 б
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 Э
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_adam_iter^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: t
Read_13/DisableCopyOnReadDisableCopyOnReadread_13_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_13/ReadVariableOpReadVariableOpread_13_disablecopyonread_total^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_14/DisableCopyOnReadDisableCopyOnReadread_14_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_14/ReadVariableOpReadVariableOpread_14_disablecopyonread_count^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_dense_14_kernel_m"/device:CPU:0*
_output_shapes
 ▓
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_dense_14_kernel_m^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:@Г
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_dense_14_bias_m"/device:CPU:0*
_output_shapes
 м
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_dense_14_bias_m^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:С
Read_17/DisableCopyOnReadDisableCopyOnRead<read_17_disablecopyonread_adam_lstm_28_lstm_cell_28_kernel_m"/device:CPU:0*
_output_shapes
 ┐
Read_17/ReadVariableOpReadVariableOp<read_17_disablecopyonread_adam_lstm_28_lstm_cell_28_kernel_m^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	АЫ
Read_18/DisableCopyOnReadDisableCopyOnReadFread_18_disablecopyonread_adam_lstm_28_lstm_cell_28_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 ╩
Read_18/ReadVariableOpReadVariableOpFread_18_disablecopyonread_adam_lstm_28_lstm_cell_28_recurrent_kernel_m^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААП
Read_19/DisableCopyOnReadDisableCopyOnRead:read_19_disablecopyonread_adam_lstm_28_lstm_cell_28_bias_m"/device:CPU:0*
_output_shapes
 ╣
Read_19/ReadVariableOpReadVariableOp:read_19_disablecopyonread_adam_lstm_28_lstm_cell_28_bias_m^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:АС
Read_20/DisableCopyOnReadDisableCopyOnRead<read_20_disablecopyonread_adam_lstm_29_lstm_cell_29_kernel_m"/device:CPU:0*
_output_shapes
 └
Read_20/ReadVariableOpReadVariableOp<read_20_disablecopyonread_adam_lstm_29_lstm_cell_29_kernel_m^Read_20/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЫ
Read_21/DisableCopyOnReadDisableCopyOnReadFread_21_disablecopyonread_adam_lstm_29_lstm_cell_29_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 ╔
Read_21/ReadVariableOpReadVariableOpFread_21_disablecopyonread_adam_lstm_29_lstm_cell_29_recurrent_kernel_m^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@А*
dtype0p
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@Аf
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:	@АП
Read_22/DisableCopyOnReadDisableCopyOnRead:read_22_disablecopyonread_adam_lstm_29_lstm_cell_29_bias_m"/device:CPU:0*
_output_shapes
 ╣
Read_22/ReadVariableOpReadVariableOp:read_22_disablecopyonread_adam_lstm_29_lstm_cell_29_bias_m^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_dense_14_kernel_v"/device:CPU:0*
_output_shapes
 ▓
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_dense_14_kernel_v^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:@Г
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_dense_14_bias_v"/device:CPU:0*
_output_shapes
 м
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_dense_14_bias_v^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:С
Read_25/DisableCopyOnReadDisableCopyOnRead<read_25_disablecopyonread_adam_lstm_28_lstm_cell_28_kernel_v"/device:CPU:0*
_output_shapes
 ┐
Read_25/ReadVariableOpReadVariableOp<read_25_disablecopyonread_adam_lstm_28_lstm_cell_28_kernel_v^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	АЫ
Read_26/DisableCopyOnReadDisableCopyOnReadFread_26_disablecopyonread_adam_lstm_28_lstm_cell_28_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 ╩
Read_26/ReadVariableOpReadVariableOpFread_26_disablecopyonread_adam_lstm_28_lstm_cell_28_recurrent_kernel_v^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААП
Read_27/DisableCopyOnReadDisableCopyOnRead:read_27_disablecopyonread_adam_lstm_28_lstm_cell_28_bias_v"/device:CPU:0*
_output_shapes
 ╣
Read_27/ReadVariableOpReadVariableOp:read_27_disablecopyonread_adam_lstm_28_lstm_cell_28_bias_v^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:АС
Read_28/DisableCopyOnReadDisableCopyOnRead<read_28_disablecopyonread_adam_lstm_29_lstm_cell_29_kernel_v"/device:CPU:0*
_output_shapes
 └
Read_28/ReadVariableOpReadVariableOp<read_28_disablecopyonread_adam_lstm_29_lstm_cell_29_kernel_v^Read_28/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЫ
Read_29/DisableCopyOnReadDisableCopyOnReadFread_29_disablecopyonread_adam_lstm_29_lstm_cell_29_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 ╔
Read_29/ReadVariableOpReadVariableOpFread_29_disablecopyonread_adam_lstm_29_lstm_cell_29_recurrent_kernel_v^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@А*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@Аf
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	@АП
Read_30/DisableCopyOnReadDisableCopyOnRead:read_30_disablecopyonread_adam_lstm_29_lstm_cell_29_bias_v"/device:CPU:0*
_output_shapes
 ╣
Read_30/ReadVariableOpReadVariableOp:read_30_disablecopyonread_adam_lstm_29_lstm_cell_29_bias_v^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:А╡
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*▐
value╘B╤ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHн
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *.
dtypes$
"2 	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_62Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_63IdentityIdentity_62:output:0^NoOp*
T0*
_output_shapes
: о
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_63Identity_63:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp: 

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╥7
Ж
C__inference_lstm_29_layer_call_and_return_conditional_losses_233843

inputs'
lstm_cell_29_233762:
АА"
lstm_cell_29_233764:	А&
lstm_cell_29_233766:	@А
identityИв$lstm_cell_29/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskї
$lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_29_233762lstm_cell_29_233764lstm_cell_29_233766*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233761n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╖
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_29_233762lstm_cell_29_233764lstm_cell_29_233766*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_233775*
condR
while_cond_233774*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @u
NoOpNoOp%^lstm_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2L
$lstm_cell_29/StatefulPartitionedCall$lstm_cell_29/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
▌И
ш
C__inference_lstm_28_layer_call_and_return_conditional_losses_234361

inputs=
*lstm_cell_28_split_readvariableop_resource:	А;
,lstm_cell_28_split_1_readvariableop_resource:	А8
$lstm_cell_28_readvariableop_resource:
АА
identityИвlstm_cell_28/ReadVariableOpвlstm_cell_28/ReadVariableOp_1вlstm_cell_28/ReadVariableOp_2вlstm_cell_28/ReadVariableOp_3в!lstm_cell_28/split/ReadVariableOpв#lstm_cell_28/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask^
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_28/split/ReadVariableOpReadVariableOp*lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0)lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_28/split_1/ReadVariableOpReadVariableOp,lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_28/split_1Split'lstm_cell_28/split_1/split_dim:output:0+lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_28/BiasAddBiasAddlstm_cell_28/MatMul:product:0lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_1BiasAddlstm_cell_28/MatMul_1:product:0lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_2BiasAddlstm_cell_28/MatMul_2:product:0lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_3BiasAddlstm_cell_28/MatMul_3:product:0lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_28/ReadVariableOpReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_28/strided_sliceStridedSlice#lstm_cell_28/ReadVariableOp:value:0)lstm_cell_28/strided_slice/stack:output:0+lstm_cell_28/strided_slice/stack_1:output:0+lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_28/MatMul_4MatMulzeros:output:0#lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_28/addAddV2lstm_cell_28/BiasAdd:output:0lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_28/MulMullstm_cell_28/add:z:0lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_28/Add_1AddV2lstm_cell_28/Mul:z:0lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_28/clip_by_value/MinimumMinimumlstm_cell_28/Add_1:z:0-lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_28/clip_by_valueMaximum&lstm_cell_28/clip_by_value/Minimum:z:0%lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_1ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_1StridedSlice%lstm_cell_28/ReadVariableOp_1:value:0+lstm_cell_28/strided_slice_1/stack:output:0-lstm_cell_28/strided_slice_1/stack_1:output:0-lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_5MatMulzeros:output:0%lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_2AddV2lstm_cell_28/BiasAdd_1:output:0lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_1Mullstm_cell_28/add_2:z:0lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_3AddV2lstm_cell_28/Mul_1:z:0lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_cell_28/Add_3:z:0/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_1Maximum(lstm_cell_28/clip_by_value_1/Minimum:z:0'lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_28/mul_2Mul lstm_cell_28/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_2ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_2StridedSlice%lstm_cell_28/ReadVariableOp_2:value:0+lstm_cell_28/strided_slice_2/stack:output:0-lstm_cell_28/strided_slice_2/stack_1:output:0-lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_6MatMulzeros:output:0%lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_4AddV2lstm_cell_28/BiasAdd_2:output:0lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_28/ReluRelulstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_28/mul_3Mullstm_cell_28/clip_by_value:z:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_28/add_5AddV2lstm_cell_28/mul_2:z:0lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_3ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_3StridedSlice%lstm_cell_28/ReadVariableOp_3:value:0+lstm_cell_28/strided_slice_3/stack:output:0-lstm_cell_28/strided_slice_3/stack_1:output:0-lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_7MatMulzeros:output:0%lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_6AddV2lstm_cell_28/BiasAdd_3:output:0lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_4Mullstm_cell_28/add_6:z:0lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_7AddV2lstm_cell_28/Mul_4:z:0lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_cell_28/Add_7:z:0/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_2Maximum(lstm_cell_28/clip_by_value_2/Minimum:z:0'lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_28/Relu_1Relulstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_28/mul_5Mul lstm_cell_28/clip_by_value_2:z:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : №
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_28_split_readvariableop_resource,lstm_cell_28_split_1_readvariableop_resource$lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_234221*
condR
while_cond_234220*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         АЦ
NoOpNoOp^lstm_cell_28/ReadVariableOp^lstm_cell_28/ReadVariableOp_1^lstm_cell_28/ReadVariableOp_2^lstm_cell_28/ReadVariableOp_3"^lstm_cell_28/split/ReadVariableOp$^lstm_cell_28/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2>
lstm_cell_28/ReadVariableOp_1lstm_cell_28/ReadVariableOp_12>
lstm_cell_28/ReadVariableOp_2lstm_cell_28/ReadVariableOp_22>
lstm_cell_28/ReadVariableOp_3lstm_cell_28/ReadVariableOp_32:
lstm_cell_28/ReadVariableOplstm_cell_28/ReadVariableOp2F
!lstm_cell_28/split/ReadVariableOp!lstm_cell_28/split/ReadVariableOp2J
#lstm_cell_28/split_1/ReadVariableOp#lstm_cell_28/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ип
к
'sequential_14_lstm_28_while_body_232777H
Dsequential_14_lstm_28_while_sequential_14_lstm_28_while_loop_counterN
Jsequential_14_lstm_28_while_sequential_14_lstm_28_while_maximum_iterations+
'sequential_14_lstm_28_while_placeholder-
)sequential_14_lstm_28_while_placeholder_1-
)sequential_14_lstm_28_while_placeholder_2-
)sequential_14_lstm_28_while_placeholder_3G
Csequential_14_lstm_28_while_sequential_14_lstm_28_strided_slice_1_0Г
sequential_14_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_28_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_14_lstm_28_while_lstm_cell_28_split_readvariableop_resource_0:	АY
Jsequential_14_lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0:	АV
Bsequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource_0:
АА(
$sequential_14_lstm_28_while_identity*
&sequential_14_lstm_28_while_identity_1*
&sequential_14_lstm_28_while_identity_2*
&sequential_14_lstm_28_while_identity_3*
&sequential_14_lstm_28_while_identity_4*
&sequential_14_lstm_28_while_identity_5E
Asequential_14_lstm_28_while_sequential_14_lstm_28_strided_slice_1Б
}sequential_14_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_28_tensorarrayunstack_tensorlistfromtensorY
Fsequential_14_lstm_28_while_lstm_cell_28_split_readvariableop_resource:	АW
Hsequential_14_lstm_28_while_lstm_cell_28_split_1_readvariableop_resource:	АT
@sequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource:
ААИв7sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOpв9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_1в9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_2в9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_3в=sequential_14/lstm_28/while/lstm_cell_28/split/ReadVariableOpв?sequential_14/lstm_28/while/lstm_cell_28/split_1/ReadVariableOpЮ
Msequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ф
?sequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_14_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_28_tensorarrayunstack_tensorlistfromtensor_0'sequential_14_lstm_28_while_placeholderVsequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0z
8sequential_14/lstm_28/while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╟
=sequential_14/lstm_28/while/lstm_cell_28/split/ReadVariableOpReadVariableOpHsequential_14_lstm_28_while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0Э
.sequential_14/lstm_28/while/lstm_cell_28/splitSplitAsequential_14/lstm_28/while/lstm_cell_28/split/split_dim:output:0Esequential_14/lstm_28/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitэ
/sequential_14/lstm_28/while/lstm_cell_28/MatMulMatMulFsequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_28/while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ая
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_1MatMulFsequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_28/while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ая
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_2MatMulFsequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_28/while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ая
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_3MatMulFsequential_14/lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_28/while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А|
:sequential_14/lstm_28/while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╟
?sequential_14/lstm_28/while/lstm_cell_28/split_1/ReadVariableOpReadVariableOpJsequential_14_lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0У
0sequential_14/lstm_28/while/lstm_cell_28/split_1SplitCsequential_14/lstm_28/while/lstm_cell_28/split_1/split_dim:output:0Gsequential_14/lstm_28/while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitф
0sequential_14/lstm_28/while/lstm_cell_28/BiasAddBiasAdd9sequential_14/lstm_28/while/lstm_cell_28/MatMul:product:09sequential_14/lstm_28/while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аш
2sequential_14/lstm_28/while/lstm_cell_28/BiasAdd_1BiasAdd;sequential_14/lstm_28/while/lstm_cell_28/MatMul_1:product:09sequential_14/lstm_28/while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аш
2sequential_14/lstm_28/while/lstm_cell_28/BiasAdd_2BiasAdd;sequential_14/lstm_28/while/lstm_cell_28/MatMul_2:product:09sequential_14/lstm_28/while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аш
2sequential_14/lstm_28/while/lstm_cell_28/BiasAdd_3BiasAdd;sequential_14/lstm_28/while/lstm_cell_28/MatMul_3:product:09sequential_14/lstm_28/while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         А╝
7sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOpReadVariableOpBsequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Н
<sequential_14/lstm_28/while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        П
>sequential_14/lstm_28/while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   П
>sequential_14/lstm_28/while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ║
6sequential_14/lstm_28/while/lstm_cell_28/strided_sliceStridedSlice?sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp:value:0Esequential_14/lstm_28/while/lstm_cell_28/strided_slice/stack:output:0Gsequential_14/lstm_28/while/lstm_cell_28/strided_slice/stack_1:output:0Gsequential_14/lstm_28/while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask┌
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_4MatMul)sequential_14_lstm_28_while_placeholder_2?sequential_14/lstm_28/while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         Ар
,sequential_14/lstm_28/while/lstm_cell_28/addAddV29sequential_14/lstm_28/while/lstm_cell_28/BiasAdd:output:0;sequential_14/lstm_28/while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         Аs
.sequential_14/lstm_28/while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_14/lstm_28/while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╤
,sequential_14/lstm_28/while/lstm_cell_28/MulMul0sequential_14/lstm_28/while/lstm_cell_28/add:z:07sequential_14/lstm_28/while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         А╫
.sequential_14/lstm_28/while/lstm_cell_28/Add_1AddV20sequential_14/lstm_28/while/lstm_cell_28/Mul:z:09sequential_14/lstm_28/while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         АЕ
@sequential_14/lstm_28/while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?√
>sequential_14/lstm_28/while/lstm_cell_28/clip_by_value/MinimumMinimum2sequential_14/lstm_28/while/lstm_cell_28/Add_1:z:0Isequential_14/lstm_28/while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         А}
8sequential_14/lstm_28/while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    √
6sequential_14/lstm_28/while/lstm_cell_28/clip_by_valueMaximumBsequential_14/lstm_28/while/lstm_cell_28/clip_by_value/Minimum:z:0Asequential_14/lstm_28/while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         А╛
9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_1ReadVariableOpBsequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0П
>sequential_14/lstm_28/while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   С
@sequential_14/lstm_28/while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       С
@sequential_14/lstm_28/while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ─
8sequential_14/lstm_28/while/lstm_cell_28/strided_slice_1StridedSliceAsequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_1:value:0Gsequential_14/lstm_28/while/lstm_cell_28/strided_slice_1/stack:output:0Isequential_14/lstm_28/while/lstm_cell_28/strided_slice_1/stack_1:output:0Isequential_14/lstm_28/while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▄
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_5MatMul)sequential_14_lstm_28_while_placeholder_2Asequential_14/lstm_28/while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Аф
.sequential_14/lstm_28/while/lstm_cell_28/add_2AddV2;sequential_14/lstm_28/while/lstm_cell_28/BiasAdd_1:output:0;sequential_14/lstm_28/while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         Аu
0sequential_14/lstm_28/while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_14/lstm_28/while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╫
.sequential_14/lstm_28/while/lstm_cell_28/Mul_1Mul2sequential_14/lstm_28/while/lstm_cell_28/add_2:z:09sequential_14/lstm_28/while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         А┘
.sequential_14/lstm_28/while/lstm_cell_28/Add_3AddV22sequential_14/lstm_28/while/lstm_cell_28/Mul_1:z:09sequential_14/lstm_28/while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         АЗ
Bsequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А? 
@sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1/MinimumMinimum2sequential_14/lstm_28/while/lstm_cell_28/Add_3:z:0Ksequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         А
:sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
8sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1MaximumDsequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum:z:0Csequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А╤
.sequential_14/lstm_28/while/lstm_cell_28/mul_2Mul<sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_1:z:0)sequential_14_lstm_28_while_placeholder_3*
T0*(
_output_shapes
:         А╛
9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_2ReadVariableOpBsequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0П
>sequential_14/lstm_28/while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       С
@sequential_14/lstm_28/while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  С
@sequential_14/lstm_28/while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ─
8sequential_14/lstm_28/while/lstm_cell_28/strided_slice_2StridedSliceAsequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_2:value:0Gsequential_14/lstm_28/while/lstm_cell_28/strided_slice_2/stack:output:0Isequential_14/lstm_28/while/lstm_cell_28/strided_slice_2/stack_1:output:0Isequential_14/lstm_28/while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▄
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_6MatMul)sequential_14_lstm_28_while_placeholder_2Asequential_14/lstm_28/while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Аф
.sequential_14/lstm_28/while/lstm_cell_28/add_4AddV2;sequential_14/lstm_28/while/lstm_cell_28/BiasAdd_2:output:0;sequential_14/lstm_28/while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         АЬ
-sequential_14/lstm_28/while/lstm_cell_28/ReluRelu2sequential_14/lstm_28/while/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         Ас
.sequential_14/lstm_28/while/lstm_cell_28/mul_3Mul:sequential_14/lstm_28/while/lstm_cell_28/clip_by_value:z:0;sequential_14/lstm_28/while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А╥
.sequential_14/lstm_28/while/lstm_cell_28/add_5AddV22sequential_14/lstm_28/while/lstm_cell_28/mul_2:z:02sequential_14/lstm_28/while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         А╛
9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_3ReadVariableOpBsequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0П
>sequential_14/lstm_28/while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  С
@sequential_14/lstm_28/while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        С
@sequential_14/lstm_28/while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ─
8sequential_14/lstm_28/while/lstm_cell_28/strided_slice_3StridedSliceAsequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_3:value:0Gsequential_14/lstm_28/while/lstm_cell_28/strided_slice_3/stack:output:0Isequential_14/lstm_28/while/lstm_cell_28/strided_slice_3/stack_1:output:0Isequential_14/lstm_28/while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▄
1sequential_14/lstm_28/while/lstm_cell_28/MatMul_7MatMul)sequential_14_lstm_28_while_placeholder_2Asequential_14/lstm_28/while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Аф
.sequential_14/lstm_28/while/lstm_cell_28/add_6AddV2;sequential_14/lstm_28/while/lstm_cell_28/BiasAdd_3:output:0;sequential_14/lstm_28/while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         Аu
0sequential_14/lstm_28/while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_14/lstm_28/while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╫
.sequential_14/lstm_28/while/lstm_cell_28/Mul_4Mul2sequential_14/lstm_28/while/lstm_cell_28/add_6:z:09sequential_14/lstm_28/while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         А┘
.sequential_14/lstm_28/while/lstm_cell_28/Add_7AddV22sequential_14/lstm_28/while/lstm_cell_28/Mul_4:z:09sequential_14/lstm_28/while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         АЗ
Bsequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А? 
@sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2/MinimumMinimum2sequential_14/lstm_28/while/lstm_cell_28/Add_7:z:0Ksequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         А
:sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
8sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2MaximumDsequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum:z:0Csequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АЮ
/sequential_14/lstm_28/while/lstm_cell_28/Relu_1Relu2sequential_14/lstm_28/while/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Ах
.sequential_14/lstm_28/while/lstm_cell_28/mul_5Mul<sequential_14/lstm_28/while/lstm_cell_28/clip_by_value_2:z:0=sequential_14/lstm_28/while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         АЭ
@sequential_14/lstm_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_14_lstm_28_while_placeholder_1'sequential_14_lstm_28_while_placeholder2sequential_14/lstm_28/while/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥c
!sequential_14/lstm_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ю
sequential_14/lstm_28/while/addAddV2'sequential_14_lstm_28_while_placeholder*sequential_14/lstm_28/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_14/lstm_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_14/lstm_28/while/add_1AddV2Dsequential_14_lstm_28_while_sequential_14_lstm_28_while_loop_counter,sequential_14/lstm_28/while/add_1/y:output:0*
T0*
_output_shapes
: Ы
$sequential_14/lstm_28/while/IdentityIdentity%sequential_14/lstm_28/while/add_1:z:0!^sequential_14/lstm_28/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_14/lstm_28/while/Identity_1IdentityJsequential_14_lstm_28_while_sequential_14_lstm_28_while_maximum_iterations!^sequential_14/lstm_28/while/NoOp*
T0*
_output_shapes
: Ы
&sequential_14/lstm_28/while/Identity_2Identity#sequential_14/lstm_28/while/add:z:0!^sequential_14/lstm_28/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_14/lstm_28/while/Identity_3IdentityPsequential_14/lstm_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_14/lstm_28/while/NoOp*
T0*
_output_shapes
: ╝
&sequential_14/lstm_28/while/Identity_4Identity2sequential_14/lstm_28/while/lstm_cell_28/mul_5:z:0!^sequential_14/lstm_28/while/NoOp*
T0*(
_output_shapes
:         А╝
&sequential_14/lstm_28/while/Identity_5Identity2sequential_14/lstm_28/while/lstm_cell_28/add_5:z:0!^sequential_14/lstm_28/while/NoOp*
T0*(
_output_shapes
:         А╥
 sequential_14/lstm_28/while/NoOpNoOp8^sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp:^sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_1:^sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_2:^sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_3>^sequential_14/lstm_28/while/lstm_cell_28/split/ReadVariableOp@^sequential_14/lstm_28/while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Y
&sequential_14_lstm_28_while_identity_1/sequential_14/lstm_28/while/Identity_1:output:0"Y
&sequential_14_lstm_28_while_identity_2/sequential_14/lstm_28/while/Identity_2:output:0"Y
&sequential_14_lstm_28_while_identity_3/sequential_14/lstm_28/while/Identity_3:output:0"Y
&sequential_14_lstm_28_while_identity_4/sequential_14/lstm_28/while/Identity_4:output:0"Y
&sequential_14_lstm_28_while_identity_5/sequential_14/lstm_28/while/Identity_5:output:0"U
$sequential_14_lstm_28_while_identity-sequential_14/lstm_28/while/Identity:output:0"Ж
@sequential_14_lstm_28_while_lstm_cell_28_readvariableop_resourceBsequential_14_lstm_28_while_lstm_cell_28_readvariableop_resource_0"Ц
Hsequential_14_lstm_28_while_lstm_cell_28_split_1_readvariableop_resourceJsequential_14_lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0"Т
Fsequential_14_lstm_28_while_lstm_cell_28_split_readvariableop_resourceHsequential_14_lstm_28_while_lstm_cell_28_split_readvariableop_resource_0"И
Asequential_14_lstm_28_while_sequential_14_lstm_28_strided_slice_1Csequential_14_lstm_28_while_sequential_14_lstm_28_strided_slice_1_0"А
}sequential_14_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_28_tensorarrayunstack_tensorlistfromtensorsequential_14_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_28_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2v
9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_19sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_12v
9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_29sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_22v
9sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_39sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp_32r
7sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp7sequential_14/lstm_28/while/lstm_cell_28/ReadVariableOp2~
=sequential_14/lstm_28/while/lstm_cell_28/split/ReadVariableOp=sequential_14/lstm_28/while/lstm_cell_28/split/ReadVariableOp2В
?sequential_14/lstm_28/while/lstm_cell_28/split_1/ReadVariableOp?sequential_14/lstm_28/while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_14/lstm_28/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_14/lstm_28/while/loop_counter
Ы	
├
while_cond_235084
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_235084___redundant_placeholder04
0while_while_cond_235084___redundant_placeholder14
0while_while_cond_235084___redundant_placeholder24
0while_while_cond_235084___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ы	
├
while_cond_237136
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237136___redundant_placeholder04
0while_while_cond_237136___redundant_placeholder14
0while_while_cond_237136___redundant_placeholder24
0while_while_cond_237136___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ЫЙ
ъ
C__inference_lstm_28_layer_call_and_return_conditional_losses_236765
inputs_0=
*lstm_cell_28_split_readvariableop_resource:	А;
,lstm_cell_28_split_1_readvariableop_resource:	А8
$lstm_cell_28_readvariableop_resource:
АА
identityИвlstm_cell_28/ReadVariableOpвlstm_cell_28/ReadVariableOp_1вlstm_cell_28/ReadVariableOp_2вlstm_cell_28/ReadVariableOp_3в!lstm_cell_28/split/ReadVariableOpв#lstm_cell_28/split_1/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask^
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_28/split/ReadVariableOpReadVariableOp*lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0)lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_28/split_1/ReadVariableOpReadVariableOp,lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_28/split_1Split'lstm_cell_28/split_1/split_dim:output:0+lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_28/BiasAddBiasAddlstm_cell_28/MatMul:product:0lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_1BiasAddlstm_cell_28/MatMul_1:product:0lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_2BiasAddlstm_cell_28/MatMul_2:product:0lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_3BiasAddlstm_cell_28/MatMul_3:product:0lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_28/ReadVariableOpReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_28/strided_sliceStridedSlice#lstm_cell_28/ReadVariableOp:value:0)lstm_cell_28/strided_slice/stack:output:0+lstm_cell_28/strided_slice/stack_1:output:0+lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_28/MatMul_4MatMulzeros:output:0#lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_28/addAddV2lstm_cell_28/BiasAdd:output:0lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_28/MulMullstm_cell_28/add:z:0lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_28/Add_1AddV2lstm_cell_28/Mul:z:0lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_28/clip_by_value/MinimumMinimumlstm_cell_28/Add_1:z:0-lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_28/clip_by_valueMaximum&lstm_cell_28/clip_by_value/Minimum:z:0%lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_1ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_1StridedSlice%lstm_cell_28/ReadVariableOp_1:value:0+lstm_cell_28/strided_slice_1/stack:output:0-lstm_cell_28/strided_slice_1/stack_1:output:0-lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_5MatMulzeros:output:0%lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_2AddV2lstm_cell_28/BiasAdd_1:output:0lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_1Mullstm_cell_28/add_2:z:0lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_3AddV2lstm_cell_28/Mul_1:z:0lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_cell_28/Add_3:z:0/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_1Maximum(lstm_cell_28/clip_by_value_1/Minimum:z:0'lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_28/mul_2Mul lstm_cell_28/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_2ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_2StridedSlice%lstm_cell_28/ReadVariableOp_2:value:0+lstm_cell_28/strided_slice_2/stack:output:0-lstm_cell_28/strided_slice_2/stack_1:output:0-lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_6MatMulzeros:output:0%lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_4AddV2lstm_cell_28/BiasAdd_2:output:0lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_28/ReluRelulstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_28/mul_3Mullstm_cell_28/clip_by_value:z:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_28/add_5AddV2lstm_cell_28/mul_2:z:0lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_3ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_3StridedSlice%lstm_cell_28/ReadVariableOp_3:value:0+lstm_cell_28/strided_slice_3/stack:output:0-lstm_cell_28/strided_slice_3/stack_1:output:0-lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_7MatMulzeros:output:0%lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_6AddV2lstm_cell_28/BiasAdd_3:output:0lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_4Mullstm_cell_28/add_6:z:0lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_7AddV2lstm_cell_28/Mul_4:z:0lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_cell_28/Add_7:z:0/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_2Maximum(lstm_cell_28/clip_by_value_2/Minimum:z:0'lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_28/Relu_1Relulstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_28/mul_5Mul lstm_cell_28/clip_by_value_2:z:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : №
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_28_split_readvariableop_resource,lstm_cell_28_split_1_readvariableop_resource$lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_236625*
condR
while_cond_236624*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  Аl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  АЦ
NoOpNoOp^lstm_cell_28/ReadVariableOp^lstm_cell_28/ReadVariableOp_1^lstm_cell_28/ReadVariableOp_2^lstm_cell_28/ReadVariableOp_3"^lstm_cell_28/split/ReadVariableOp$^lstm_cell_28/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2>
lstm_cell_28/ReadVariableOp_1lstm_cell_28/ReadVariableOp_12>
lstm_cell_28/ReadVariableOp_2lstm_cell_28/ReadVariableOp_22>
lstm_cell_28/ReadVariableOp_3lstm_cell_28/ReadVariableOp_32:
lstm_cell_28/ReadVariableOplstm_cell_28/ReadVariableOp2F
!lstm_cell_28/split/ReadVariableOp!lstm_cell_28/split/ReadVariableOp2J
#lstm_cell_28/split_1/ReadVariableOp#lstm_cell_28/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
иИ
ш
C__inference_lstm_29_layer_call_and_return_conditional_losses_238345

inputs>
*lstm_cell_29_split_readvariableop_resource:
АА;
,lstm_cell_29_split_1_readvariableop_resource:	А7
$lstm_cell_29_readvariableop_resource:	@А
identityИвlstm_cell_29/ReadVariableOpвlstm_cell_29/ReadVariableOp_1вlstm_cell_29/ReadVariableOp_2вlstm_cell_29/ReadVariableOp_3в!lstm_cell_29/split/ReadVariableOpв#lstm_cell_29/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask^
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_29/split/ReadVariableOpReadVariableOp*lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0)lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_29/split_1/ReadVariableOpReadVariableOp,lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_29/split_1Split'lstm_cell_29/split_1/split_dim:output:0+lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_29/BiasAddBiasAddlstm_cell_29/MatMul:product:0lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_1BiasAddlstm_cell_29/MatMul_1:product:0lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_2BiasAddlstm_cell_29/MatMul_2:product:0lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_3BiasAddlstm_cell_29/MatMul_3:product:0lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_29/ReadVariableOpReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_29/strided_sliceStridedSlice#lstm_cell_29/ReadVariableOp:value:0)lstm_cell_29/strided_slice/stack:output:0+lstm_cell_29/strided_slice/stack_1:output:0+lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_29/MatMul_4MatMulzeros:output:0#lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_29/addAddV2lstm_cell_29/BiasAdd:output:0lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_29/MulMullstm_cell_29/add:z:0lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_29/Add_1AddV2lstm_cell_29/Mul:z:0lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_29/clip_by_value/MinimumMinimumlstm_cell_29/Add_1:z:0-lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_29/clip_by_valueMaximum&lstm_cell_29/clip_by_value/Minimum:z:0%lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_1ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_1StridedSlice%lstm_cell_29/ReadVariableOp_1:value:0+lstm_cell_29/strided_slice_1/stack:output:0-lstm_cell_29/strided_slice_1/stack_1:output:0-lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_5MatMulzeros:output:0%lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_2AddV2lstm_cell_29/BiasAdd_1:output:0lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_1Mullstm_cell_29/add_2:z:0lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_3AddV2lstm_cell_29/Mul_1:z:0lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_cell_29/Add_3:z:0/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_1Maximum(lstm_cell_29/clip_by_value_1/Minimum:z:0'lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_29/mul_2Mul lstm_cell_29/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_2ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_2StridedSlice%lstm_cell_29/ReadVariableOp_2:value:0+lstm_cell_29/strided_slice_2/stack:output:0-lstm_cell_29/strided_slice_2/stack_1:output:0-lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_6MatMulzeros:output:0%lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_4AddV2lstm_cell_29/BiasAdd_2:output:0lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/ReluRelulstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_29/mul_3Mullstm_cell_29/clip_by_value:z:0lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_29/add_5AddV2lstm_cell_29/mul_2:z:0lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_3ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_3StridedSlice%lstm_cell_29/ReadVariableOp_3:value:0+lstm_cell_29/strided_slice_3/stack:output:0-lstm_cell_29/strided_slice_3/stack_1:output:0-lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_7MatMulzeros:output:0%lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_6AddV2lstm_cell_29/BiasAdd_3:output:0lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_4Mullstm_cell_29/add_6:z:0lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_7AddV2lstm_cell_29/Mul_4:z:0lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_cell_29/Add_7:z:0/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_2Maximum(lstm_cell_29/clip_by_value_2/Minimum:z:0'lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_29/Relu_1Relulstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_29/mul_5Mul lstm_cell_29/clip_by_value_2:z:0!lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_29_split_readvariableop_resource,lstm_cell_29_split_1_readvariableop_resource$lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_238205*
condR
while_cond_238204*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_29/ReadVariableOp^lstm_cell_29/ReadVariableOp_1^lstm_cell_29/ReadVariableOp_2^lstm_cell_29/ReadVariableOp_3"^lstm_cell_29/split/ReadVariableOp$^lstm_cell_29/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         А: : : 2>
lstm_cell_29/ReadVariableOp_1lstm_cell_29/ReadVariableOp_12>
lstm_cell_29/ReadVariableOp_2lstm_cell_29/ReadVariableOp_22>
lstm_cell_29/ReadVariableOp_3lstm_cell_29/ReadVariableOp_32:
lstm_cell_29/ReadVariableOplstm_cell_29/ReadVariableOp2F
!lstm_cell_29/split/ReadVariableOp!lstm_cell_29/split/ReadVariableOp2J
#lstm_cell_29/split_1/ReadVariableOp#lstm_cell_29/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Ч	
├
while_cond_237692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237692___redundant_placeholder04
0while_while_cond_237692___redundant_placeholder14
0while_while_cond_237692___redundant_placeholder24
0while_while_cond_237692___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▌И
ш
C__inference_lstm_28_layer_call_and_return_conditional_losses_237277

inputs=
*lstm_cell_28_split_readvariableop_resource:	А;
,lstm_cell_28_split_1_readvariableop_resource:	А8
$lstm_cell_28_readvariableop_resource:
АА
identityИвlstm_cell_28/ReadVariableOpвlstm_cell_28/ReadVariableOp_1вlstm_cell_28/ReadVariableOp_2вlstm_cell_28/ReadVariableOp_3в!lstm_cell_28/split/ReadVariableOpв#lstm_cell_28/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask^
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_28/split/ReadVariableOpReadVariableOp*lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0)lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_28/split_1/ReadVariableOpReadVariableOp,lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_28/split_1Split'lstm_cell_28/split_1/split_dim:output:0+lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_28/BiasAddBiasAddlstm_cell_28/MatMul:product:0lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_1BiasAddlstm_cell_28/MatMul_1:product:0lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_2BiasAddlstm_cell_28/MatMul_2:product:0lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_3BiasAddlstm_cell_28/MatMul_3:product:0lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_28/ReadVariableOpReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_28/strided_sliceStridedSlice#lstm_cell_28/ReadVariableOp:value:0)lstm_cell_28/strided_slice/stack:output:0+lstm_cell_28/strided_slice/stack_1:output:0+lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_28/MatMul_4MatMulzeros:output:0#lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_28/addAddV2lstm_cell_28/BiasAdd:output:0lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_28/MulMullstm_cell_28/add:z:0lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_28/Add_1AddV2lstm_cell_28/Mul:z:0lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_28/clip_by_value/MinimumMinimumlstm_cell_28/Add_1:z:0-lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_28/clip_by_valueMaximum&lstm_cell_28/clip_by_value/Minimum:z:0%lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_1ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_1StridedSlice%lstm_cell_28/ReadVariableOp_1:value:0+lstm_cell_28/strided_slice_1/stack:output:0-lstm_cell_28/strided_slice_1/stack_1:output:0-lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_5MatMulzeros:output:0%lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_2AddV2lstm_cell_28/BiasAdd_1:output:0lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_1Mullstm_cell_28/add_2:z:0lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_3AddV2lstm_cell_28/Mul_1:z:0lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_cell_28/Add_3:z:0/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_1Maximum(lstm_cell_28/clip_by_value_1/Minimum:z:0'lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_28/mul_2Mul lstm_cell_28/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_2ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_2StridedSlice%lstm_cell_28/ReadVariableOp_2:value:0+lstm_cell_28/strided_slice_2/stack:output:0-lstm_cell_28/strided_slice_2/stack_1:output:0-lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_6MatMulzeros:output:0%lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_4AddV2lstm_cell_28/BiasAdd_2:output:0lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_28/ReluRelulstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_28/mul_3Mullstm_cell_28/clip_by_value:z:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_28/add_5AddV2lstm_cell_28/mul_2:z:0lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_3ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_3StridedSlice%lstm_cell_28/ReadVariableOp_3:value:0+lstm_cell_28/strided_slice_3/stack:output:0-lstm_cell_28/strided_slice_3/stack_1:output:0-lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_7MatMulzeros:output:0%lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_6AddV2lstm_cell_28/BiasAdd_3:output:0lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_4Mullstm_cell_28/add_6:z:0lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_7AddV2lstm_cell_28/Mul_4:z:0lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_cell_28/Add_7:z:0/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_2Maximum(lstm_cell_28/clip_by_value_2/Minimum:z:0'lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_28/Relu_1Relulstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_28/mul_5Mul lstm_cell_28/clip_by_value_2:z:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : №
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_28_split_readvariableop_resource,lstm_cell_28_split_1_readvariableop_resource$lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_237137*
condR
while_cond_237136*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         АЦ
NoOpNoOp^lstm_cell_28/ReadVariableOp^lstm_cell_28/ReadVariableOp_1^lstm_cell_28/ReadVariableOp_2^lstm_cell_28/ReadVariableOp_3"^lstm_cell_28/split/ReadVariableOp$^lstm_cell_28/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2>
lstm_cell_28/ReadVariableOp_1lstm_cell_28/ReadVariableOp_12>
lstm_cell_28/ReadVariableOp_2lstm_cell_28/ReadVariableOp_22>
lstm_cell_28/ReadVariableOp_3lstm_cell_28/ReadVariableOp_32:
lstm_cell_28/ReadVariableOplstm_cell_28/ReadVariableOp2F
!lstm_cell_28/split/ReadVariableOp!lstm_cell_28/split/ReadVariableOp2J
#lstm_cell_28/split_1/ReadVariableOp#lstm_cell_28/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
б~
ж	
while_body_237137
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_28_split_readvariableop_resource_0:	АC
4while_lstm_cell_28_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_28_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_28_split_readvariableop_resource:	АA
2while_lstm_cell_28_split_1_readvariableop_resource:	А>
*while_lstm_cell_28_readvariableop_resource:
ААИв!while/lstm_cell_28/ReadVariableOpв#while/lstm_cell_28/ReadVariableOp_1в#while/lstm_cell_28/ReadVariableOp_2в#while/lstm_cell_28/ReadVariableOp_3в'while/lstm_cell_28/split/ReadVariableOpв)while/lstm_cell_28/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_28/split/ReadVariableOpReadVariableOp2while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_28/split_1Split-while/lstm_cell_28/split_1/split_dim:output:01while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_28/BiasAddBiasAdd#while/lstm_cell_28/MatMul:product:0#while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_1BiasAdd%while/lstm_cell_28/MatMul_1:product:0#while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_2BiasAdd%while/lstm_cell_28/MatMul_2:product:0#while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_3BiasAdd%while/lstm_cell_28/MatMul_3:product:0#while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_28/ReadVariableOpReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_28/strided_sliceStridedSlice)while/lstm_cell_28/ReadVariableOp:value:0/while/lstm_cell_28/strided_slice/stack:output:01while/lstm_cell_28/strided_slice/stack_1:output:01while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_28/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_28/addAddV2#while/lstm_cell_28/BiasAdd:output:0%while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_28/MulMulwhile/lstm_cell_28/add:z:0!while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_28/Add_1AddV2while/lstm_cell_28/Mul:z:0#while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_28/clip_by_value/MinimumMinimumwhile/lstm_cell_28/Add_1:z:03while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_28/clip_by_valueMaximum,while/lstm_cell_28/clip_by_value/Minimum:z:0+while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_1ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_1StridedSlice+while/lstm_cell_28/ReadVariableOp_1:value:01while/lstm_cell_28/strided_slice_1/stack:output:03while/lstm_cell_28/strided_slice_1/stack_1:output:03while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_2AddV2%while/lstm_cell_28/BiasAdd_1:output:0%while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_1Mulwhile/lstm_cell_28/add_2:z:0#while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_3AddV2while/lstm_cell_28/Mul_1:z:0#while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_1/MinimumMinimumwhile/lstm_cell_28/Add_3:z:05while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_1Maximum.while/lstm_cell_28/clip_by_value_1/Minimum:z:0-while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_28/mul_2Mul&while/lstm_cell_28/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_2ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_2StridedSlice+while/lstm_cell_28/ReadVariableOp_2:value:01while/lstm_cell_28/strided_slice_2/stack:output:03while/lstm_cell_28/strided_slice_2/stack_1:output:03while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_4AddV2%while/lstm_cell_28/BiasAdd_2:output:0%while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_28/ReluReluwhile/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_28/mul_3Mul$while/lstm_cell_28/clip_by_value:z:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_28/add_5AddV2while/lstm_cell_28/mul_2:z:0while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_3ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_3StridedSlice+while/lstm_cell_28/ReadVariableOp_3:value:01while/lstm_cell_28/strided_slice_3/stack:output:03while/lstm_cell_28/strided_slice_3/stack_1:output:03while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_6AddV2%while/lstm_cell_28/BiasAdd_3:output:0%while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_4Mulwhile/lstm_cell_28/add_6:z:0#while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_7AddV2while/lstm_cell_28/Mul_4:z:0#while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_2/MinimumMinimumwhile/lstm_cell_28/Add_7:z:05while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_2Maximum.while/lstm_cell_28/clip_by_value_2/Minimum:z:0-while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_28/mul_5Mul&while/lstm_cell_28/clip_by_value_2:z:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_28/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_28/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_28/ReadVariableOp$^while/lstm_cell_28/ReadVariableOp_1$^while/lstm_cell_28/ReadVariableOp_2$^while/lstm_cell_28/ReadVariableOp_3(^while/lstm_cell_28/split/ReadVariableOp*^while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_28_readvariableop_resource,while_lstm_cell_28_readvariableop_resource_0"j
2while_lstm_cell_28_split_1_readvariableop_resource4while_lstm_cell_28_split_1_readvariableop_resource_0"f
0while_lstm_cell_28_split_readvariableop_resource2while_lstm_cell_28_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_28/ReadVariableOp_1#while/lstm_cell_28/ReadVariableOp_12J
#while/lstm_cell_28/ReadVariableOp_2#while/lstm_cell_28/ReadVariableOp_22J
#while/lstm_cell_28/ReadVariableOp_3#while/lstm_cell_28/ReadVariableOp_32F
!while/lstm_cell_28/ReadVariableOp!while/lstm_cell_28/ReadVariableOp2R
'while/lstm_cell_28/split/ReadVariableOp'while/lstm_cell_28/split/ReadVariableOp2V
)while/lstm_cell_28/split_1/ReadVariableOp)while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Э·
∙
!__inference__wrapped_model_233175
lstm_28_inputS
@sequential_14_lstm_28_lstm_cell_28_split_readvariableop_resource:	АQ
Bsequential_14_lstm_28_lstm_cell_28_split_1_readvariableop_resource:	АN
:sequential_14_lstm_28_lstm_cell_28_readvariableop_resource:
ААT
@sequential_14_lstm_29_lstm_cell_29_split_readvariableop_resource:
ААQ
Bsequential_14_lstm_29_lstm_cell_29_split_1_readvariableop_resource:	АM
:sequential_14_lstm_29_lstm_cell_29_readvariableop_resource:	@АG
5sequential_14_dense_14_matmul_readvariableop_resource:@D
6sequential_14_dense_14_biasadd_readvariableop_resource:
identityИв-sequential_14/dense_14/BiasAdd/ReadVariableOpв,sequential_14/dense_14/MatMul/ReadVariableOpв1sequential_14/lstm_28/lstm_cell_28/ReadVariableOpв3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_1в3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_2в3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_3в7sequential_14/lstm_28/lstm_cell_28/split/ReadVariableOpв9sequential_14/lstm_28/lstm_cell_28/split_1/ReadVariableOpвsequential_14/lstm_28/whileв1sequential_14/lstm_29/lstm_cell_29/ReadVariableOpв3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_1в3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_2в3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_3в7sequential_14/lstm_29/lstm_cell_29/split/ReadVariableOpв9sequential_14/lstm_29/lstm_cell_29/split_1/ReadVariableOpвsequential_14/lstm_29/whilef
sequential_14/lstm_28/ShapeShapelstm_28_input*
T0*
_output_shapes
::э╧s
)sequential_14/lstm_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_14/lstm_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_14/lstm_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_14/lstm_28/strided_sliceStridedSlice$sequential_14/lstm_28/Shape:output:02sequential_14/lstm_28/strided_slice/stack:output:04sequential_14/lstm_28/strided_slice/stack_1:output:04sequential_14/lstm_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_14/lstm_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А╡
"sequential_14/lstm_28/zeros/packedPack,sequential_14/lstm_28/strided_slice:output:0-sequential_14/lstm_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_14/lstm_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    п
sequential_14/lstm_28/zerosFill+sequential_14/lstm_28/zeros/packed:output:0*sequential_14/lstm_28/zeros/Const:output:0*
T0*(
_output_shapes
:         Аi
&sequential_14/lstm_28/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А╣
$sequential_14/lstm_28/zeros_1/packedPack,sequential_14/lstm_28/strided_slice:output:0/sequential_14/lstm_28/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_14/lstm_28/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╡
sequential_14/lstm_28/zeros_1Fill-sequential_14/lstm_28/zeros_1/packed:output:0,sequential_14/lstm_28/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аy
$sequential_14/lstm_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
sequential_14/lstm_28/transpose	Transposelstm_28_input-sequential_14/lstm_28/transpose/perm:output:0*
T0*+
_output_shapes
:         ~
sequential_14/lstm_28/Shape_1Shape#sequential_14/lstm_28/transpose:y:0*
T0*
_output_shapes
::э╧u
+sequential_14/lstm_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_14/lstm_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_14/lstm_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_14/lstm_28/strided_slice_1StridedSlice&sequential_14/lstm_28/Shape_1:output:04sequential_14/lstm_28/strided_slice_1/stack:output:06sequential_14/lstm_28/strided_slice_1/stack_1:output:06sequential_14/lstm_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_14/lstm_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
#sequential_14/lstm_28/TensorArrayV2TensorListReserve:sequential_14/lstm_28/TensorArrayV2/element_shape:output:0.sequential_14/lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ь
Ksequential_14/lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       в
=sequential_14/lstm_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_14/lstm_28/transpose:y:0Tsequential_14/lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥u
+sequential_14/lstm_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_14/lstm_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_14/lstm_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╫
%sequential_14/lstm_28/strided_slice_2StridedSlice#sequential_14/lstm_28/transpose:y:04sequential_14/lstm_28/strided_slice_2/stack:output:06sequential_14/lstm_28/strided_slice_2/stack_1:output:06sequential_14/lstm_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskt
2sequential_14/lstm_28/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╣
7sequential_14/lstm_28/lstm_cell_28/split/ReadVariableOpReadVariableOp@sequential_14_lstm_28_lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
(sequential_14/lstm_28/lstm_cell_28/splitSplit;sequential_14/lstm_28/lstm_cell_28/split/split_dim:output:0?sequential_14/lstm_28/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split╔
)sequential_14/lstm_28/lstm_cell_28/MatMulMatMul.sequential_14/lstm_28/strided_slice_2:output:01sequential_14/lstm_28/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         А╦
+sequential_14/lstm_28/lstm_cell_28/MatMul_1MatMul.sequential_14/lstm_28/strided_slice_2:output:01sequential_14/lstm_28/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         А╦
+sequential_14/lstm_28/lstm_cell_28/MatMul_2MatMul.sequential_14/lstm_28/strided_slice_2:output:01sequential_14/lstm_28/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         А╦
+sequential_14/lstm_28/lstm_cell_28/MatMul_3MatMul.sequential_14/lstm_28/strided_slice_2:output:01sequential_14/lstm_28/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аv
4sequential_14/lstm_28/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╣
9sequential_14/lstm_28/lstm_cell_28/split_1/ReadVariableOpReadVariableOpBsequential_14_lstm_28_lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Б
*sequential_14/lstm_28/lstm_cell_28/split_1Split=sequential_14/lstm_28/lstm_cell_28/split_1/split_dim:output:0Asequential_14/lstm_28/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_split╥
*sequential_14/lstm_28/lstm_cell_28/BiasAddBiasAdd3sequential_14/lstm_28/lstm_cell_28/MatMul:product:03sequential_14/lstm_28/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         А╓
,sequential_14/lstm_28/lstm_cell_28/BiasAdd_1BiasAdd5sequential_14/lstm_28/lstm_cell_28/MatMul_1:product:03sequential_14/lstm_28/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         А╓
,sequential_14/lstm_28/lstm_cell_28/BiasAdd_2BiasAdd5sequential_14/lstm_28/lstm_cell_28/MatMul_2:product:03sequential_14/lstm_28/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         А╓
,sequential_14/lstm_28/lstm_cell_28/BiasAdd_3BiasAdd5sequential_14/lstm_28/lstm_cell_28/MatMul_3:product:03sequential_14/lstm_28/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         Ао
1sequential_14/lstm_28/lstm_cell_28/ReadVariableOpReadVariableOp:sequential_14_lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0З
6sequential_14/lstm_28/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Й
8sequential_14/lstm_28/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Й
8sequential_14/lstm_28/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ь
0sequential_14/lstm_28/lstm_cell_28/strided_sliceStridedSlice9sequential_14/lstm_28/lstm_cell_28/ReadVariableOp:value:0?sequential_14/lstm_28/lstm_cell_28/strided_slice/stack:output:0Asequential_14/lstm_28/lstm_cell_28/strided_slice/stack_1:output:0Asequential_14/lstm_28/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╔
+sequential_14/lstm_28/lstm_cell_28/MatMul_4MatMul$sequential_14/lstm_28/zeros:output:09sequential_14/lstm_28/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         А╬
&sequential_14/lstm_28/lstm_cell_28/addAddV23sequential_14/lstm_28/lstm_cell_28/BiasAdd:output:05sequential_14/lstm_28/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         Аm
(sequential_14/lstm_28/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_14/lstm_28/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?┐
&sequential_14/lstm_28/lstm_cell_28/MulMul*sequential_14/lstm_28/lstm_cell_28/add:z:01sequential_14/lstm_28/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         А┼
(sequential_14/lstm_28/lstm_cell_28/Add_1AddV2*sequential_14/lstm_28/lstm_cell_28/Mul:z:03sequential_14/lstm_28/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         А
:sequential_14/lstm_28/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?щ
8sequential_14/lstm_28/lstm_cell_28/clip_by_value/MinimumMinimum,sequential_14/lstm_28/lstm_cell_28/Add_1:z:0Csequential_14/lstm_28/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аw
2sequential_14/lstm_28/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    щ
0sequential_14/lstm_28/lstm_cell_28/clip_by_valueMaximum<sequential_14/lstm_28/lstm_cell_28/clip_by_value/Minimum:z:0;sequential_14/lstm_28/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         А░
3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_1ReadVariableOp:sequential_14_lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
8sequential_14/lstm_28/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Л
:sequential_14/lstm_28/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Л
:sequential_14/lstm_28/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
2sequential_14/lstm_28/lstm_cell_28/strided_slice_1StridedSlice;sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_1:value:0Asequential_14/lstm_28/lstm_cell_28/strided_slice_1/stack:output:0Csequential_14/lstm_28/lstm_cell_28/strided_slice_1/stack_1:output:0Csequential_14/lstm_28/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╦
+sequential_14/lstm_28/lstm_cell_28/MatMul_5MatMul$sequential_14/lstm_28/zeros:output:0;sequential_14/lstm_28/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         А╥
(sequential_14/lstm_28/lstm_cell_28/add_2AddV25sequential_14/lstm_28/lstm_cell_28/BiasAdd_1:output:05sequential_14/lstm_28/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         Аo
*sequential_14/lstm_28/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_14/lstm_28/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
(sequential_14/lstm_28/lstm_cell_28/Mul_1Mul,sequential_14/lstm_28/lstm_cell_28/add_2:z:03sequential_14/lstm_28/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         А╟
(sequential_14/lstm_28/lstm_cell_28/Add_3AddV2,sequential_14/lstm_28/lstm_cell_28/Mul_1:z:03sequential_14/lstm_28/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         АБ
<sequential_14/lstm_28/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?э
:sequential_14/lstm_28/lstm_cell_28/clip_by_value_1/MinimumMinimum,sequential_14/lstm_28/lstm_cell_28/Add_3:z:0Esequential_14/lstm_28/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аy
4sequential_14/lstm_28/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    я
2sequential_14/lstm_28/lstm_cell_28/clip_by_value_1Maximum>sequential_14/lstm_28/lstm_cell_28/clip_by_value_1/Minimum:z:0=sequential_14/lstm_28/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А┬
(sequential_14/lstm_28/lstm_cell_28/mul_2Mul6sequential_14/lstm_28/lstm_cell_28/clip_by_value_1:z:0&sequential_14/lstm_28/zeros_1:output:0*
T0*(
_output_shapes
:         А░
3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_2ReadVariableOp:sequential_14_lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
8sequential_14/lstm_28/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Л
:sequential_14/lstm_28/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  Л
:sequential_14/lstm_28/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
2sequential_14/lstm_28/lstm_cell_28/strided_slice_2StridedSlice;sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_2:value:0Asequential_14/lstm_28/lstm_cell_28/strided_slice_2/stack:output:0Csequential_14/lstm_28/lstm_cell_28/strided_slice_2/stack_1:output:0Csequential_14/lstm_28/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╦
+sequential_14/lstm_28/lstm_cell_28/MatMul_6MatMul$sequential_14/lstm_28/zeros:output:0;sequential_14/lstm_28/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         А╥
(sequential_14/lstm_28/lstm_cell_28/add_4AddV25sequential_14/lstm_28/lstm_cell_28/BiasAdd_2:output:05sequential_14/lstm_28/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         АР
'sequential_14/lstm_28/lstm_cell_28/ReluRelu,sequential_14/lstm_28/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         А╧
(sequential_14/lstm_28/lstm_cell_28/mul_3Mul4sequential_14/lstm_28/lstm_cell_28/clip_by_value:z:05sequential_14/lstm_28/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А└
(sequential_14/lstm_28/lstm_cell_28/add_5AddV2,sequential_14/lstm_28/lstm_cell_28/mul_2:z:0,sequential_14/lstm_28/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         А░
3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_3ReadVariableOp:sequential_14_lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
8sequential_14/lstm_28/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  Л
:sequential_14/lstm_28/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Л
:sequential_14/lstm_28/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
2sequential_14/lstm_28/lstm_cell_28/strided_slice_3StridedSlice;sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_3:value:0Asequential_14/lstm_28/lstm_cell_28/strided_slice_3/stack:output:0Csequential_14/lstm_28/lstm_cell_28/strided_slice_3/stack_1:output:0Csequential_14/lstm_28/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╦
+sequential_14/lstm_28/lstm_cell_28/MatMul_7MatMul$sequential_14/lstm_28/zeros:output:0;sequential_14/lstm_28/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         А╥
(sequential_14/lstm_28/lstm_cell_28/add_6AddV25sequential_14/lstm_28/lstm_cell_28/BiasAdd_3:output:05sequential_14/lstm_28/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         Аo
*sequential_14/lstm_28/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_14/lstm_28/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
(sequential_14/lstm_28/lstm_cell_28/Mul_4Mul,sequential_14/lstm_28/lstm_cell_28/add_6:z:03sequential_14/lstm_28/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         А╟
(sequential_14/lstm_28/lstm_cell_28/Add_7AddV2,sequential_14/lstm_28/lstm_cell_28/Mul_4:z:03sequential_14/lstm_28/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         АБ
<sequential_14/lstm_28/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?э
:sequential_14/lstm_28/lstm_cell_28/clip_by_value_2/MinimumMinimum,sequential_14/lstm_28/lstm_cell_28/Add_7:z:0Esequential_14/lstm_28/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аy
4sequential_14/lstm_28/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    я
2sequential_14/lstm_28/lstm_cell_28/clip_by_value_2Maximum>sequential_14/lstm_28/lstm_cell_28/clip_by_value_2/Minimum:z:0=sequential_14/lstm_28/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АТ
)sequential_14/lstm_28/lstm_cell_28/Relu_1Relu,sequential_14/lstm_28/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         А╙
(sequential_14/lstm_28/lstm_cell_28/mul_5Mul6sequential_14/lstm_28/lstm_cell_28/clip_by_value_2:z:07sequential_14/lstm_28/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         АД
3sequential_14/lstm_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ·
%sequential_14/lstm_28/TensorArrayV2_1TensorListReserve<sequential_14/lstm_28/TensorArrayV2_1/element_shape:output:0.sequential_14/lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥\
sequential_14/lstm_28/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_14/lstm_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_14/lstm_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ░
sequential_14/lstm_28/whileWhile1sequential_14/lstm_28/while/loop_counter:output:07sequential_14/lstm_28/while/maximum_iterations:output:0#sequential_14/lstm_28/time:output:0.sequential_14/lstm_28/TensorArrayV2_1:handle:0$sequential_14/lstm_28/zeros:output:0&sequential_14/lstm_28/zeros_1:output:0.sequential_14/lstm_28/strided_slice_1:output:0Msequential_14/lstm_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_14_lstm_28_lstm_cell_28_split_readvariableop_resourceBsequential_14_lstm_28_lstm_cell_28_split_1_readvariableop_resource:sequential_14_lstm_28_lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_14_lstm_28_while_body_232777*3
cond+R)
'sequential_14_lstm_28_while_cond_232776*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Ч
Fsequential_14/lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   Е
8sequential_14/lstm_28/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_14/lstm_28/while:output:3Osequential_14/lstm_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0~
+sequential_14/lstm_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_14/lstm_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_14/lstm_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
%sequential_14/lstm_28/strided_slice_3StridedSliceAsequential_14/lstm_28/TensorArrayV2Stack/TensorListStack:tensor:04sequential_14/lstm_28/strided_slice_3/stack:output:06sequential_14/lstm_28/strided_slice_3/stack_1:output:06sequential_14/lstm_28/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask{
&sequential_14/lstm_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┘
!sequential_14/lstm_28/transpose_1	TransposeAsequential_14/lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_14/lstm_28/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А~
sequential_14/lstm_29/ShapeShape%sequential_14/lstm_28/transpose_1:y:0*
T0*
_output_shapes
::э╧s
)sequential_14/lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_14/lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_14/lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_14/lstm_29/strided_sliceStridedSlice$sequential_14/lstm_29/Shape:output:02sequential_14/lstm_29/strided_slice/stack:output:04sequential_14/lstm_29/strided_slice/stack_1:output:04sequential_14/lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_14/lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@╡
"sequential_14/lstm_29/zeros/packedPack,sequential_14/lstm_29/strided_slice:output:0-sequential_14/lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_14/lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    о
sequential_14/lstm_29/zerosFill+sequential_14/lstm_29/zeros/packed:output:0*sequential_14/lstm_29/zeros/Const:output:0*
T0*'
_output_shapes
:         @h
&sequential_14/lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@╣
$sequential_14/lstm_29/zeros_1/packedPack,sequential_14/lstm_29/strided_slice:output:0/sequential_14/lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_14/lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_14/lstm_29/zeros_1Fill-sequential_14/lstm_29/zeros_1/packed:output:0,sequential_14/lstm_29/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @y
$sequential_14/lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╣
sequential_14/lstm_29/transpose	Transpose%sequential_14/lstm_28/transpose_1:y:0-sequential_14/lstm_29/transpose/perm:output:0*
T0*,
_output_shapes
:         А~
sequential_14/lstm_29/Shape_1Shape#sequential_14/lstm_29/transpose:y:0*
T0*
_output_shapes
::э╧u
+sequential_14/lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_14/lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_14/lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_14/lstm_29/strided_slice_1StridedSlice&sequential_14/lstm_29/Shape_1:output:04sequential_14/lstm_29/strided_slice_1/stack:output:06sequential_14/lstm_29/strided_slice_1/stack_1:output:06sequential_14/lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_14/lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
#sequential_14/lstm_29/TensorArrayV2TensorListReserve:sequential_14/lstm_29/TensorArrayV2/element_shape:output:0.sequential_14/lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ь
Ksequential_14/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   в
=sequential_14/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_14/lstm_29/transpose:y:0Tsequential_14/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥u
+sequential_14/lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_14/lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_14/lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
%sequential_14/lstm_29/strided_slice_2StridedSlice#sequential_14/lstm_29/transpose:y:04sequential_14/lstm_29/strided_slice_2/stack:output:06sequential_14/lstm_29/strided_slice_2/stack_1:output:06sequential_14/lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskt
2sequential_14/lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
7sequential_14/lstm_29/lstm_cell_29/split/ReadVariableOpReadVariableOp@sequential_14_lstm_29_lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Л
(sequential_14/lstm_29/lstm_cell_29/splitSplit;sequential_14/lstm_29/lstm_cell_29/split/split_dim:output:0?sequential_14/lstm_29/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split╚
)sequential_14/lstm_29/lstm_cell_29/MatMulMatMul.sequential_14/lstm_29/strided_slice_2:output:01sequential_14/lstm_29/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @╩
+sequential_14/lstm_29/lstm_cell_29/MatMul_1MatMul.sequential_14/lstm_29/strided_slice_2:output:01sequential_14/lstm_29/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @╩
+sequential_14/lstm_29/lstm_cell_29/MatMul_2MatMul.sequential_14/lstm_29/strided_slice_2:output:01sequential_14/lstm_29/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @╩
+sequential_14/lstm_29/lstm_cell_29/MatMul_3MatMul.sequential_14/lstm_29/strided_slice_2:output:01sequential_14/lstm_29/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @v
4sequential_14/lstm_29/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╣
9sequential_14/lstm_29/lstm_cell_29/split_1/ReadVariableOpReadVariableOpBsequential_14_lstm_29_lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0¤
*sequential_14/lstm_29/lstm_cell_29/split_1Split=sequential_14/lstm_29/lstm_cell_29/split_1/split_dim:output:0Asequential_14/lstm_29/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split╤
*sequential_14/lstm_29/lstm_cell_29/BiasAddBiasAdd3sequential_14/lstm_29/lstm_cell_29/MatMul:product:03sequential_14/lstm_29/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @╒
,sequential_14/lstm_29/lstm_cell_29/BiasAdd_1BiasAdd5sequential_14/lstm_29/lstm_cell_29/MatMul_1:product:03sequential_14/lstm_29/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @╒
,sequential_14/lstm_29/lstm_cell_29/BiasAdd_2BiasAdd5sequential_14/lstm_29/lstm_cell_29/MatMul_2:product:03sequential_14/lstm_29/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @╒
,sequential_14/lstm_29/lstm_cell_29/BiasAdd_3BiasAdd5sequential_14/lstm_29/lstm_cell_29/MatMul_3:product:03sequential_14/lstm_29/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @н
1sequential_14/lstm_29/lstm_cell_29/ReadVariableOpReadVariableOp:sequential_14_lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0З
6sequential_14/lstm_29/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Й
8sequential_14/lstm_29/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Й
8sequential_14/lstm_29/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ъ
0sequential_14/lstm_29/lstm_cell_29/strided_sliceStridedSlice9sequential_14/lstm_29/lstm_cell_29/ReadVariableOp:value:0?sequential_14/lstm_29/lstm_cell_29/strided_slice/stack:output:0Asequential_14/lstm_29/lstm_cell_29/strided_slice/stack_1:output:0Asequential_14/lstm_29/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╚
+sequential_14/lstm_29/lstm_cell_29/MatMul_4MatMul$sequential_14/lstm_29/zeros:output:09sequential_14/lstm_29/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @═
&sequential_14/lstm_29/lstm_cell_29/addAddV23sequential_14/lstm_29/lstm_cell_29/BiasAdd:output:05sequential_14/lstm_29/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
(sequential_14/lstm_29/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_14/lstm_29/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╛
&sequential_14/lstm_29/lstm_cell_29/MulMul*sequential_14/lstm_29/lstm_cell_29/add:z:01sequential_14/lstm_29/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @─
(sequential_14/lstm_29/lstm_cell_29/Add_1AddV2*sequential_14/lstm_29/lstm_cell_29/Mul:z:03sequential_14/lstm_29/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @
:sequential_14/lstm_29/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ш
8sequential_14/lstm_29/lstm_cell_29/clip_by_value/MinimumMinimum,sequential_14/lstm_29/lstm_cell_29/Add_1:z:0Csequential_14/lstm_29/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @w
2sequential_14/lstm_29/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ш
0sequential_14/lstm_29/lstm_cell_29/clip_by_valueMaximum<sequential_14/lstm_29/lstm_cell_29/clip_by_value/Minimum:z:0;sequential_14/lstm_29/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @п
3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_1ReadVariableOp:sequential_14_lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0Й
8sequential_14/lstm_29/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   Л
:sequential_14/lstm_29/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Л
:sequential_14/lstm_29/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_14/lstm_29/lstm_cell_29/strided_slice_1StridedSlice;sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_1:value:0Asequential_14/lstm_29/lstm_cell_29/strided_slice_1/stack:output:0Csequential_14/lstm_29/lstm_cell_29/strided_slice_1/stack_1:output:0Csequential_14/lstm_29/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╩
+sequential_14/lstm_29/lstm_cell_29/MatMul_5MatMul$sequential_14/lstm_29/zeros:output:0;sequential_14/lstm_29/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @╤
(sequential_14/lstm_29/lstm_cell_29/add_2AddV25sequential_14/lstm_29/lstm_cell_29/BiasAdd_1:output:05sequential_14/lstm_29/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @o
*sequential_14/lstm_29/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_14/lstm_29/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?─
(sequential_14/lstm_29/lstm_cell_29/Mul_1Mul,sequential_14/lstm_29/lstm_cell_29/add_2:z:03sequential_14/lstm_29/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @╞
(sequential_14/lstm_29/lstm_cell_29/Add_3AddV2,sequential_14/lstm_29/lstm_cell_29/Mul_1:z:03sequential_14/lstm_29/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @Б
<sequential_14/lstm_29/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ь
:sequential_14/lstm_29/lstm_cell_29/clip_by_value_1/MinimumMinimum,sequential_14/lstm_29/lstm_cell_29/Add_3:z:0Esequential_14/lstm_29/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @y
4sequential_14/lstm_29/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
2sequential_14/lstm_29/lstm_cell_29/clip_by_value_1Maximum>sequential_14/lstm_29/lstm_cell_29/clip_by_value_1/Minimum:z:0=sequential_14/lstm_29/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @┴
(sequential_14/lstm_29/lstm_cell_29/mul_2Mul6sequential_14/lstm_29/lstm_cell_29/clip_by_value_1:z:0&sequential_14/lstm_29/zeros_1:output:0*
T0*'
_output_shapes
:         @п
3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_2ReadVariableOp:sequential_14_lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0Й
8sequential_14/lstm_29/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Л
:sequential_14/lstm_29/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   Л
:sequential_14/lstm_29/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_14/lstm_29/lstm_cell_29/strided_slice_2StridedSlice;sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_2:value:0Asequential_14/lstm_29/lstm_cell_29/strided_slice_2/stack:output:0Csequential_14/lstm_29/lstm_cell_29/strided_slice_2/stack_1:output:0Csequential_14/lstm_29/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╩
+sequential_14/lstm_29/lstm_cell_29/MatMul_6MatMul$sequential_14/lstm_29/zeros:output:0;sequential_14/lstm_29/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @╤
(sequential_14/lstm_29/lstm_cell_29/add_4AddV25sequential_14/lstm_29/lstm_cell_29/BiasAdd_2:output:05sequential_14/lstm_29/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @П
'sequential_14/lstm_29/lstm_cell_29/ReluRelu,sequential_14/lstm_29/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @╬
(sequential_14/lstm_29/lstm_cell_29/mul_3Mul4sequential_14/lstm_29/lstm_cell_29/clip_by_value:z:05sequential_14/lstm_29/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @┐
(sequential_14/lstm_29/lstm_cell_29/add_5AddV2,sequential_14/lstm_29/lstm_cell_29/mul_2:z:0,sequential_14/lstm_29/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @п
3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_3ReadVariableOp:sequential_14_lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0Й
8sequential_14/lstm_29/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   Л
:sequential_14/lstm_29/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Л
:sequential_14/lstm_29/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_14/lstm_29/lstm_cell_29/strided_slice_3StridedSlice;sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_3:value:0Asequential_14/lstm_29/lstm_cell_29/strided_slice_3/stack:output:0Csequential_14/lstm_29/lstm_cell_29/strided_slice_3/stack_1:output:0Csequential_14/lstm_29/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╩
+sequential_14/lstm_29/lstm_cell_29/MatMul_7MatMul$sequential_14/lstm_29/zeros:output:0;sequential_14/lstm_29/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @╤
(sequential_14/lstm_29/lstm_cell_29/add_6AddV25sequential_14/lstm_29/lstm_cell_29/BiasAdd_3:output:05sequential_14/lstm_29/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @o
*sequential_14/lstm_29/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_14/lstm_29/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?─
(sequential_14/lstm_29/lstm_cell_29/Mul_4Mul,sequential_14/lstm_29/lstm_cell_29/add_6:z:03sequential_14/lstm_29/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @╞
(sequential_14/lstm_29/lstm_cell_29/Add_7AddV2,sequential_14/lstm_29/lstm_cell_29/Mul_4:z:03sequential_14/lstm_29/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @Б
<sequential_14/lstm_29/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ь
:sequential_14/lstm_29/lstm_cell_29/clip_by_value_2/MinimumMinimum,sequential_14/lstm_29/lstm_cell_29/Add_7:z:0Esequential_14/lstm_29/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @y
4sequential_14/lstm_29/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
2sequential_14/lstm_29/lstm_cell_29/clip_by_value_2Maximum>sequential_14/lstm_29/lstm_cell_29/clip_by_value_2/Minimum:z:0=sequential_14/lstm_29/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @С
)sequential_14/lstm_29/lstm_cell_29/Relu_1Relu,sequential_14/lstm_29/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @╥
(sequential_14/lstm_29/lstm_cell_29/mul_5Mul6sequential_14/lstm_29/lstm_cell_29/clip_by_value_2:z:07sequential_14/lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @Д
3sequential_14/lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ·
%sequential_14/lstm_29/TensorArrayV2_1TensorListReserve<sequential_14/lstm_29/TensorArrayV2_1/element_shape:output:0.sequential_14/lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥\
sequential_14/lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_14/lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_14/lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : м
sequential_14/lstm_29/whileWhile1sequential_14/lstm_29/while/loop_counter:output:07sequential_14/lstm_29/while/maximum_iterations:output:0#sequential_14/lstm_29/time:output:0.sequential_14/lstm_29/TensorArrayV2_1:handle:0$sequential_14/lstm_29/zeros:output:0&sequential_14/lstm_29/zeros_1:output:0.sequential_14/lstm_29/strided_slice_1:output:0Msequential_14/lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_14_lstm_29_lstm_cell_29_split_readvariableop_resourceBsequential_14_lstm_29_lstm_cell_29_split_1_readvariableop_resource:sequential_14_lstm_29_lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_14_lstm_29_while_body_233029*3
cond+R)
'sequential_14_lstm_29_while_cond_233028*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ч
Fsequential_14/lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Д
8sequential_14/lstm_29/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_14/lstm_29/while:output:3Osequential_14/lstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0~
+sequential_14/lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_14/lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_14/lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
%sequential_14/lstm_29/strided_slice_3StridedSliceAsequential_14/lstm_29/TensorArrayV2Stack/TensorListStack:tensor:04sequential_14/lstm_29/strided_slice_3/stack:output:06sequential_14/lstm_29/strided_slice_3/stack_1:output:06sequential_14/lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask{
&sequential_14/lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╪
!sequential_14/lstm_29/transpose_1	TransposeAsequential_14/lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_14/lstm_29/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @в
,sequential_14/dense_14/MatMul/ReadVariableOpReadVariableOp5sequential_14_dense_14_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0┐
sequential_14/dense_14/MatMulMatMul.sequential_14/lstm_29/strided_slice_3:output:04sequential_14/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_14/dense_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_14_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_14/dense_14/BiasAddBiasAdd'sequential_14/dense_14/MatMul:product:05sequential_14/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
IdentityIdentity'sequential_14/dense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ∙
NoOpNoOp.^sequential_14/dense_14/BiasAdd/ReadVariableOp-^sequential_14/dense_14/MatMul/ReadVariableOp2^sequential_14/lstm_28/lstm_cell_28/ReadVariableOp4^sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_14^sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_24^sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_38^sequential_14/lstm_28/lstm_cell_28/split/ReadVariableOp:^sequential_14/lstm_28/lstm_cell_28/split_1/ReadVariableOp^sequential_14/lstm_28/while2^sequential_14/lstm_29/lstm_cell_29/ReadVariableOp4^sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_14^sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_24^sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_38^sequential_14/lstm_29/lstm_cell_29/split/ReadVariableOp:^sequential_14/lstm_29/lstm_cell_29/split_1/ReadVariableOp^sequential_14/lstm_29/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2^
-sequential_14/dense_14/BiasAdd/ReadVariableOp-sequential_14/dense_14/BiasAdd/ReadVariableOp2\
,sequential_14/dense_14/MatMul/ReadVariableOp,sequential_14/dense_14/MatMul/ReadVariableOp2j
3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_13sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_12j
3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_23sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_22j
3sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_33sequential_14/lstm_28/lstm_cell_28/ReadVariableOp_32f
1sequential_14/lstm_28/lstm_cell_28/ReadVariableOp1sequential_14/lstm_28/lstm_cell_28/ReadVariableOp2r
7sequential_14/lstm_28/lstm_cell_28/split/ReadVariableOp7sequential_14/lstm_28/lstm_cell_28/split/ReadVariableOp2v
9sequential_14/lstm_28/lstm_cell_28/split_1/ReadVariableOp9sequential_14/lstm_28/lstm_cell_28/split_1/ReadVariableOp2:
sequential_14/lstm_28/whilesequential_14/lstm_28/while2j
3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_13sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_12j
3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_23sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_22j
3sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_33sequential_14/lstm_29/lstm_cell_29/ReadVariableOp_32f
1sequential_14/lstm_29/lstm_cell_29/ReadVariableOp1sequential_14/lstm_29/lstm_cell_29/ReadVariableOp2r
7sequential_14/lstm_29/lstm_cell_29/split/ReadVariableOp7sequential_14/lstm_29/lstm_cell_29/split/ReadVariableOp2v
9sequential_14/lstm_29/lstm_cell_29/split_1/ReadVariableOp9sequential_14/lstm_29/lstm_cell_29/split_1/ReadVariableOp2:
sequential_14/lstm_29/whilesequential_14/lstm_29/while:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_28_input
ТР
╛
lstm_28_while_body_235553,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3+
'lstm_28_while_lstm_28_strided_slice_1_0g
clstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_28_while_lstm_cell_28_split_readvariableop_resource_0:	АK
<lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0:	АH
4lstm_28_while_lstm_cell_28_readvariableop_resource_0:
АА
lstm_28_while_identity
lstm_28_while_identity_1
lstm_28_while_identity_2
lstm_28_while_identity_3
lstm_28_while_identity_4
lstm_28_while_identity_5)
%lstm_28_while_lstm_28_strided_slice_1e
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorK
8lstm_28_while_lstm_cell_28_split_readvariableop_resource:	АI
:lstm_28_while_lstm_cell_28_split_1_readvariableop_resource:	АF
2lstm_28_while_lstm_cell_28_readvariableop_resource:
ААИв)lstm_28/while/lstm_cell_28/ReadVariableOpв+lstm_28/while/lstm_cell_28/ReadVariableOp_1в+lstm_28/while/lstm_cell_28/ReadVariableOp_2в+lstm_28/while/lstm_cell_28/ReadVariableOp_3в/lstm_28/while/lstm_cell_28/split/ReadVariableOpв1lstm_28/while/lstm_cell_28/split_1/ReadVariableOpР
?lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0lstm_28_while_placeholderHlstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0l
*lstm_28/while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
/lstm_28/while/lstm_cell_28/split/ReadVariableOpReadVariableOp:lstm_28_while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0є
 lstm_28/while/lstm_cell_28/splitSplit3lstm_28/while/lstm_cell_28/split/split_dim:output:07lstm_28/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split├
!lstm_28/while/lstm_cell_28/MatMulMatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         А┼
#lstm_28/while/lstm_cell_28/MatMul_1MatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         А┼
#lstm_28/while/lstm_cell_28/MatMul_2MatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         А┼
#lstm_28/while/lstm_cell_28/MatMul_3MatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аn
,lstm_28/while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_28/while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp<lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0щ
"lstm_28/while/lstm_cell_28/split_1Split5lstm_28/while/lstm_cell_28/split_1/split_dim:output:09lstm_28/while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_split║
"lstm_28/while/lstm_cell_28/BiasAddBiasAdd+lstm_28/while/lstm_cell_28/MatMul:product:0+lstm_28/while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         А╛
$lstm_28/while/lstm_cell_28/BiasAdd_1BiasAdd-lstm_28/while/lstm_cell_28/MatMul_1:product:0+lstm_28/while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         А╛
$lstm_28/while/lstm_cell_28/BiasAdd_2BiasAdd-lstm_28/while/lstm_cell_28/MatMul_2:product:0+lstm_28/while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         А╛
$lstm_28/while/lstm_cell_28/BiasAdd_3BiasAdd-lstm_28/while/lstm_cell_28/MatMul_3:product:0+lstm_28/while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         Аа
)lstm_28/while/lstm_cell_28/ReadVariableOpReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0
.lstm_28/while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_28/while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Б
0lstm_28/while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ї
(lstm_28/while/lstm_cell_28/strided_sliceStridedSlice1lstm_28/while/lstm_cell_28/ReadVariableOp:value:07lstm_28/while/lstm_cell_28/strided_slice/stack:output:09lstm_28/while/lstm_cell_28/strided_slice/stack_1:output:09lstm_28/while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask░
#lstm_28/while/lstm_cell_28/MatMul_4MatMullstm_28_while_placeholder_21lstm_28/while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         А╢
lstm_28/while/lstm_cell_28/addAddV2+lstm_28/while/lstm_cell_28/BiasAdd:output:0-lstm_28/while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         Аe
 lstm_28/while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_28/while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?з
lstm_28/while/lstm_cell_28/MulMul"lstm_28/while/lstm_cell_28/add:z:0)lstm_28/while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         Ан
 lstm_28/while/lstm_cell_28/Add_1AddV2"lstm_28/while/lstm_cell_28/Mul:z:0+lstm_28/while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аw
2lstm_28/while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
0lstm_28/while/lstm_cell_28/clip_by_value/MinimumMinimum$lstm_28/while/lstm_cell_28/Add_1:z:0;lstm_28/while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аo
*lstm_28/while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(lstm_28/while/lstm_cell_28/clip_by_valueMaximum4lstm_28/while/lstm_cell_28/clip_by_value/Minimum:z:03lstm_28/while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         Ав
+lstm_28/while/lstm_cell_28/ReadVariableOp_1ReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_28/while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_28/while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_28/while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_28/while/lstm_cell_28/strided_slice_1StridedSlice3lstm_28/while/lstm_cell_28/ReadVariableOp_1:value:09lstm_28/while/lstm_cell_28/strided_slice_1/stack:output:0;lstm_28/while/lstm_cell_28/strided_slice_1/stack_1:output:0;lstm_28/while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_28/while/lstm_cell_28/MatMul_5MatMullstm_28_while_placeholder_23lstm_28/while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         А║
 lstm_28/while/lstm_cell_28/add_2AddV2-lstm_28/while/lstm_cell_28/BiasAdd_1:output:0-lstm_28/while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         Аg
"lstm_28/while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_28/while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_28/while/lstm_cell_28/Mul_1Mul$lstm_28/while/lstm_cell_28/add_2:z:0+lstm_28/while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         Ап
 lstm_28/while/lstm_cell_28/Add_3AddV2$lstm_28/while/lstm_cell_28/Mul_1:z:0+lstm_28/while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аy
4lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_28/while/lstm_cell_28/clip_by_value_1/MinimumMinimum$lstm_28/while/lstm_cell_28/Add_3:z:0=lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_28/while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_28/while/lstm_cell_28/clip_by_value_1Maximum6lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum:z:05lstm_28/while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         Аз
 lstm_28/while/lstm_cell_28/mul_2Mul.lstm_28/while/lstm_cell_28/clip_by_value_1:z:0lstm_28_while_placeholder_3*
T0*(
_output_shapes
:         Ав
+lstm_28/while/lstm_cell_28/ReadVariableOp_2ReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_28/while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_28/while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_28/while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_28/while/lstm_cell_28/strided_slice_2StridedSlice3lstm_28/while/lstm_cell_28/ReadVariableOp_2:value:09lstm_28/while/lstm_cell_28/strided_slice_2/stack:output:0;lstm_28/while/lstm_cell_28/strided_slice_2/stack_1:output:0;lstm_28/while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_28/while/lstm_cell_28/MatMul_6MatMullstm_28_while_placeholder_23lstm_28/while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         А║
 lstm_28/while/lstm_cell_28/add_4AddV2-lstm_28/while/lstm_cell_28/BiasAdd_2:output:0-lstm_28/while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         АА
lstm_28/while/lstm_cell_28/ReluRelu$lstm_28/while/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         А╖
 lstm_28/while/lstm_cell_28/mul_3Mul,lstm_28/while/lstm_cell_28/clip_by_value:z:0-lstm_28/while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         Аи
 lstm_28/while/lstm_cell_28/add_5AddV2$lstm_28/while/lstm_cell_28/mul_2:z:0$lstm_28/while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         Ав
+lstm_28/while/lstm_cell_28/ReadVariableOp_3ReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_28/while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_28/while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_28/while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_28/while/lstm_cell_28/strided_slice_3StridedSlice3lstm_28/while/lstm_cell_28/ReadVariableOp_3:value:09lstm_28/while/lstm_cell_28/strided_slice_3/stack:output:0;lstm_28/while/lstm_cell_28/strided_slice_3/stack_1:output:0;lstm_28/while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_28/while/lstm_cell_28/MatMul_7MatMullstm_28_while_placeholder_23lstm_28/while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         А║
 lstm_28/while/lstm_cell_28/add_6AddV2-lstm_28/while/lstm_cell_28/BiasAdd_3:output:0-lstm_28/while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         Аg
"lstm_28/while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_28/while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_28/while/lstm_cell_28/Mul_4Mul$lstm_28/while/lstm_cell_28/add_6:z:0+lstm_28/while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         Ап
 lstm_28/while/lstm_cell_28/Add_7AddV2$lstm_28/while/lstm_cell_28/Mul_4:z:0+lstm_28/while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аy
4lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_28/while/lstm_cell_28/clip_by_value_2/MinimumMinimum$lstm_28/while/lstm_cell_28/Add_7:z:0=lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_28/while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_28/while/lstm_cell_28/clip_by_value_2Maximum6lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum:z:05lstm_28/while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АВ
!lstm_28/while/lstm_cell_28/Relu_1Relu$lstm_28/while/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         А╗
 lstm_28/while/lstm_cell_28/mul_5Mul.lstm_28/while/lstm_cell_28/clip_by_value_2:z:0/lstm_28/while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Ах
2lstm_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_28_while_placeholder_1lstm_28_while_placeholder$lstm_28/while/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_28/while/addAddV2lstm_28_while_placeholderlstm_28/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_28/while/add_1AddV2(lstm_28_while_lstm_28_while_loop_counterlstm_28/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_28/while/IdentityIdentitylstm_28/while/add_1:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: К
lstm_28/while/Identity_1Identity.lstm_28_while_lstm_28_while_maximum_iterations^lstm_28/while/NoOp*
T0*
_output_shapes
: q
lstm_28/while/Identity_2Identitylstm_28/while/add:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: Ю
lstm_28/while/Identity_3IdentityBlstm_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_28/while/NoOp*
T0*
_output_shapes
: Т
lstm_28/while/Identity_4Identity$lstm_28/while/lstm_cell_28/mul_5:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:         АТ
lstm_28/while/Identity_5Identity$lstm_28/while/lstm_cell_28/add_5:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:         АЁ
lstm_28/while/NoOpNoOp*^lstm_28/while/lstm_cell_28/ReadVariableOp,^lstm_28/while/lstm_cell_28/ReadVariableOp_1,^lstm_28/while/lstm_cell_28/ReadVariableOp_2,^lstm_28/while/lstm_cell_28/ReadVariableOp_30^lstm_28/while/lstm_cell_28/split/ReadVariableOp2^lstm_28/while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_28_while_identity_1!lstm_28/while/Identity_1:output:0"=
lstm_28_while_identity_2!lstm_28/while/Identity_2:output:0"=
lstm_28_while_identity_3!lstm_28/while/Identity_3:output:0"=
lstm_28_while_identity_4!lstm_28/while/Identity_4:output:0"=
lstm_28_while_identity_5!lstm_28/while/Identity_5:output:0"9
lstm_28_while_identitylstm_28/while/Identity:output:0"P
%lstm_28_while_lstm_28_strided_slice_1'lstm_28_while_lstm_28_strided_slice_1_0"j
2lstm_28_while_lstm_cell_28_readvariableop_resource4lstm_28_while_lstm_cell_28_readvariableop_resource_0"z
:lstm_28_while_lstm_cell_28_split_1_readvariableop_resource<lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0"v
8lstm_28_while_lstm_cell_28_split_readvariableop_resource:lstm_28_while_lstm_cell_28_split_readvariableop_resource_0"╚
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2Z
+lstm_28/while/lstm_cell_28/ReadVariableOp_1+lstm_28/while/lstm_cell_28/ReadVariableOp_12Z
+lstm_28/while/lstm_cell_28/ReadVariableOp_2+lstm_28/while/lstm_cell_28/ReadVariableOp_22Z
+lstm_28/while/lstm_cell_28/ReadVariableOp_3+lstm_28/while/lstm_cell_28/ReadVariableOp_32V
)lstm_28/while/lstm_cell_28/ReadVariableOp)lstm_28/while/lstm_cell_28/ReadVariableOp2b
/lstm_28/while/lstm_cell_28/split/ReadVariableOp/lstm_28/while/lstm_cell_28/split/ReadVariableOp2f
1lstm_28/while/lstm_cell_28/split_1/ReadVariableOp1lstm_28/while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_28/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_28/while/loop_counter
ё	
╨
.__inference_sequential_14_layer_call_fn_235320
lstm_28_input
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:
АА
	unknown_3:	А
	unknown_4:	@А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCalllstm_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_14_layer_call_and_return_conditional_losses_235280o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_28_input
Ў
ў
-__inference_lstm_cell_28_layer_call_fn_238637

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:
АА
identity

identity_1

identity_2ИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233299p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Аr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         А:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:         А
"
_user_specified_name
states_1:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю
ў
-__inference_lstm_cell_29_layer_call_fn_238866

inputs
states_0
states_1
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identity

identity_1

identity_2ИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233963o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
М
у
lstm_28_while_cond_236066,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3.
*lstm_28_while_less_lstm_28_strided_slice_1D
@lstm_28_while_lstm_28_while_cond_236066___redundant_placeholder0D
@lstm_28_while_lstm_28_while_cond_236066___redundant_placeholder1D
@lstm_28_while_lstm_28_while_cond_236066___redundant_placeholder2D
@lstm_28_while_lstm_28_while_cond_236066___redundant_placeholder3
lstm_28_while_identity
В
lstm_28/while/LessLesslstm_28_while_placeholder*lstm_28_while_less_lstm_28_strided_slice_1*
T0*
_output_shapes
: [
lstm_28/while/IdentityIdentitylstm_28/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_28_while_identitylstm_28/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_28/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_28/while/loop_counter
ы}
ж	
while_body_237949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_29_split_readvariableop_resource_0:
ААC
4while_lstm_cell_29_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_29_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_29_split_readvariableop_resource:
ААA
2while_lstm_cell_29_split_1_readvariableop_resource:	А=
*while_lstm_cell_29_readvariableop_resource:	@АИв!while/lstm_cell_29/ReadVariableOpв#while/lstm_cell_29/ReadVariableOp_1в#while/lstm_cell_29/ReadVariableOp_2в#while/lstm_cell_29/ReadVariableOp_3в'while/lstm_cell_29/split/ReadVariableOpв)while/lstm_cell_29/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_29/split/ReadVariableOpReadVariableOp2while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_29/split_1Split-while/lstm_cell_29/split_1/split_dim:output:01while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_29/BiasAddBiasAdd#while/lstm_cell_29/MatMul:product:0#while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_1BiasAdd%while/lstm_cell_29/MatMul_1:product:0#while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_2BiasAdd%while/lstm_cell_29/MatMul_2:product:0#while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_3BiasAdd%while/lstm_cell_29/MatMul_3:product:0#while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_29/ReadVariableOpReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_29/strided_sliceStridedSlice)while/lstm_cell_29/ReadVariableOp:value:0/while/lstm_cell_29/strided_slice/stack:output:01while/lstm_cell_29/strided_slice/stack_1:output:01while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_29/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_29/addAddV2#while/lstm_cell_29/BiasAdd:output:0%while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_29/MulMulwhile/lstm_cell_29/add:z:0!while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_29/Add_1AddV2while/lstm_cell_29/Mul:z:0#while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_29/clip_by_value/MinimumMinimumwhile/lstm_cell_29/Add_1:z:03while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_29/clip_by_valueMaximum,while/lstm_cell_29/clip_by_value/Minimum:z:0+while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_1ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_1StridedSlice+while/lstm_cell_29/ReadVariableOp_1:value:01while/lstm_cell_29/strided_slice_1/stack:output:03while/lstm_cell_29/strided_slice_1/stack_1:output:03while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_2AddV2%while/lstm_cell_29/BiasAdd_1:output:0%while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_1Mulwhile/lstm_cell_29/add_2:z:0#while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_3AddV2while/lstm_cell_29/Mul_1:z:0#while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_1/MinimumMinimumwhile/lstm_cell_29/Add_3:z:05while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_1Maximum.while/lstm_cell_29/clip_by_value_1/Minimum:z:0-while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_29/mul_2Mul&while/lstm_cell_29/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_2ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_2StridedSlice+while/lstm_cell_29/ReadVariableOp_2:value:01while/lstm_cell_29/strided_slice_2/stack:output:03while/lstm_cell_29/strided_slice_2/stack_1:output:03while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_4AddV2%while/lstm_cell_29/BiasAdd_2:output:0%while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_29/ReluReluwhile/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_29/mul_3Mul$while/lstm_cell_29/clip_by_value:z:0%while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_29/add_5AddV2while/lstm_cell_29/mul_2:z:0while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_3ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_3StridedSlice+while/lstm_cell_29/ReadVariableOp_3:value:01while/lstm_cell_29/strided_slice_3/stack:output:03while/lstm_cell_29/strided_slice_3/stack_1:output:03while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_6AddV2%while/lstm_cell_29/BiasAdd_3:output:0%while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_4Mulwhile/lstm_cell_29/add_6:z:0#while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_7AddV2while/lstm_cell_29/Mul_4:z:0#while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_2/MinimumMinimumwhile/lstm_cell_29/Add_7:z:05while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_2Maximum.while/lstm_cell_29/clip_by_value_2/Minimum:z:0-while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_29/mul_5Mul&while/lstm_cell_29/clip_by_value_2:z:0'while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_29/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_29/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_29/ReadVariableOp$^while/lstm_cell_29/ReadVariableOp_1$^while/lstm_cell_29/ReadVariableOp_2$^while/lstm_cell_29/ReadVariableOp_3(^while/lstm_cell_29/split/ReadVariableOp*^while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_29_readvariableop_resource,while_lstm_cell_29_readvariableop_resource_0"j
2while_lstm_cell_29_split_1_readvariableop_resource4while_lstm_cell_29_split_1_readvariableop_resource_0"f
0while_lstm_cell_29_split_readvariableop_resource2while_lstm_cell_29_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_29/ReadVariableOp_1#while/lstm_cell_29/ReadVariableOp_12J
#while/lstm_cell_29/ReadVariableOp_2#while/lstm_cell_29/ReadVariableOp_22J
#while/lstm_cell_29/ReadVariableOp_3#while/lstm_cell_29/ReadVariableOp_32F
!while/lstm_cell_29/ReadVariableOp!while/lstm_cell_29/ReadVariableOp2R
'while/lstm_cell_29/split/ReadVariableOp'while/lstm_cell_29/split/ReadVariableOp2V
)while/lstm_cell_29/split_1/ReadVariableOp)while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Т
╕
(__inference_lstm_29_layer_call_fn_237555
inputs_0
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  А
"
_user_specified_name
inputs_0
ТР
╛
lstm_28_while_body_236067,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3+
'lstm_28_while_lstm_28_strided_slice_1_0g
clstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_28_while_lstm_cell_28_split_readvariableop_resource_0:	АK
<lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0:	АH
4lstm_28_while_lstm_cell_28_readvariableop_resource_0:
АА
lstm_28_while_identity
lstm_28_while_identity_1
lstm_28_while_identity_2
lstm_28_while_identity_3
lstm_28_while_identity_4
lstm_28_while_identity_5)
%lstm_28_while_lstm_28_strided_slice_1e
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorK
8lstm_28_while_lstm_cell_28_split_readvariableop_resource:	АI
:lstm_28_while_lstm_cell_28_split_1_readvariableop_resource:	АF
2lstm_28_while_lstm_cell_28_readvariableop_resource:
ААИв)lstm_28/while/lstm_cell_28/ReadVariableOpв+lstm_28/while/lstm_cell_28/ReadVariableOp_1в+lstm_28/while/lstm_cell_28/ReadVariableOp_2в+lstm_28/while/lstm_cell_28/ReadVariableOp_3в/lstm_28/while/lstm_cell_28/split/ReadVariableOpв1lstm_28/while/lstm_cell_28/split_1/ReadVariableOpР
?lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0lstm_28_while_placeholderHlstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0l
*lstm_28/while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
/lstm_28/while/lstm_cell_28/split/ReadVariableOpReadVariableOp:lstm_28_while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0є
 lstm_28/while/lstm_cell_28/splitSplit3lstm_28/while/lstm_cell_28/split/split_dim:output:07lstm_28/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split├
!lstm_28/while/lstm_cell_28/MatMulMatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         А┼
#lstm_28/while/lstm_cell_28/MatMul_1MatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         А┼
#lstm_28/while/lstm_cell_28/MatMul_2MatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         А┼
#lstm_28/while/lstm_cell_28/MatMul_3MatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_28/while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аn
,lstm_28/while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_28/while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp<lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0щ
"lstm_28/while/lstm_cell_28/split_1Split5lstm_28/while/lstm_cell_28/split_1/split_dim:output:09lstm_28/while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_split║
"lstm_28/while/lstm_cell_28/BiasAddBiasAdd+lstm_28/while/lstm_cell_28/MatMul:product:0+lstm_28/while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         А╛
$lstm_28/while/lstm_cell_28/BiasAdd_1BiasAdd-lstm_28/while/lstm_cell_28/MatMul_1:product:0+lstm_28/while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         А╛
$lstm_28/while/lstm_cell_28/BiasAdd_2BiasAdd-lstm_28/while/lstm_cell_28/MatMul_2:product:0+lstm_28/while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         А╛
$lstm_28/while/lstm_cell_28/BiasAdd_3BiasAdd-lstm_28/while/lstm_cell_28/MatMul_3:product:0+lstm_28/while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         Аа
)lstm_28/while/lstm_cell_28/ReadVariableOpReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0
.lstm_28/while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_28/while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Б
0lstm_28/while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ї
(lstm_28/while/lstm_cell_28/strided_sliceStridedSlice1lstm_28/while/lstm_cell_28/ReadVariableOp:value:07lstm_28/while/lstm_cell_28/strided_slice/stack:output:09lstm_28/while/lstm_cell_28/strided_slice/stack_1:output:09lstm_28/while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask░
#lstm_28/while/lstm_cell_28/MatMul_4MatMullstm_28_while_placeholder_21lstm_28/while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         А╢
lstm_28/while/lstm_cell_28/addAddV2+lstm_28/while/lstm_cell_28/BiasAdd:output:0-lstm_28/while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         Аe
 lstm_28/while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_28/while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?з
lstm_28/while/lstm_cell_28/MulMul"lstm_28/while/lstm_cell_28/add:z:0)lstm_28/while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         Ан
 lstm_28/while/lstm_cell_28/Add_1AddV2"lstm_28/while/lstm_cell_28/Mul:z:0+lstm_28/while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аw
2lstm_28/while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
0lstm_28/while/lstm_cell_28/clip_by_value/MinimumMinimum$lstm_28/while/lstm_cell_28/Add_1:z:0;lstm_28/while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аo
*lstm_28/while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(lstm_28/while/lstm_cell_28/clip_by_valueMaximum4lstm_28/while/lstm_cell_28/clip_by_value/Minimum:z:03lstm_28/while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         Ав
+lstm_28/while/lstm_cell_28/ReadVariableOp_1ReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_28/while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_28/while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_28/while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_28/while/lstm_cell_28/strided_slice_1StridedSlice3lstm_28/while/lstm_cell_28/ReadVariableOp_1:value:09lstm_28/while/lstm_cell_28/strided_slice_1/stack:output:0;lstm_28/while/lstm_cell_28/strided_slice_1/stack_1:output:0;lstm_28/while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_28/while/lstm_cell_28/MatMul_5MatMullstm_28_while_placeholder_23lstm_28/while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         А║
 lstm_28/while/lstm_cell_28/add_2AddV2-lstm_28/while/lstm_cell_28/BiasAdd_1:output:0-lstm_28/while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         Аg
"lstm_28/while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_28/while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_28/while/lstm_cell_28/Mul_1Mul$lstm_28/while/lstm_cell_28/add_2:z:0+lstm_28/while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         Ап
 lstm_28/while/lstm_cell_28/Add_3AddV2$lstm_28/while/lstm_cell_28/Mul_1:z:0+lstm_28/while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аy
4lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_28/while/lstm_cell_28/clip_by_value_1/MinimumMinimum$lstm_28/while/lstm_cell_28/Add_3:z:0=lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_28/while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_28/while/lstm_cell_28/clip_by_value_1Maximum6lstm_28/while/lstm_cell_28/clip_by_value_1/Minimum:z:05lstm_28/while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         Аз
 lstm_28/while/lstm_cell_28/mul_2Mul.lstm_28/while/lstm_cell_28/clip_by_value_1:z:0lstm_28_while_placeholder_3*
T0*(
_output_shapes
:         Ав
+lstm_28/while/lstm_cell_28/ReadVariableOp_2ReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_28/while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_28/while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_28/while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_28/while/lstm_cell_28/strided_slice_2StridedSlice3lstm_28/while/lstm_cell_28/ReadVariableOp_2:value:09lstm_28/while/lstm_cell_28/strided_slice_2/stack:output:0;lstm_28/while/lstm_cell_28/strided_slice_2/stack_1:output:0;lstm_28/while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_28/while/lstm_cell_28/MatMul_6MatMullstm_28_while_placeholder_23lstm_28/while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         А║
 lstm_28/while/lstm_cell_28/add_4AddV2-lstm_28/while/lstm_cell_28/BiasAdd_2:output:0-lstm_28/while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         АА
lstm_28/while/lstm_cell_28/ReluRelu$lstm_28/while/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         А╖
 lstm_28/while/lstm_cell_28/mul_3Mul,lstm_28/while/lstm_cell_28/clip_by_value:z:0-lstm_28/while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         Аи
 lstm_28/while/lstm_cell_28/add_5AddV2$lstm_28/while/lstm_cell_28/mul_2:z:0$lstm_28/while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         Ав
+lstm_28/while/lstm_cell_28/ReadVariableOp_3ReadVariableOp4lstm_28_while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_28/while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_28/while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_28/while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_28/while/lstm_cell_28/strided_slice_3StridedSlice3lstm_28/while/lstm_cell_28/ReadVariableOp_3:value:09lstm_28/while/lstm_cell_28/strided_slice_3/stack:output:0;lstm_28/while/lstm_cell_28/strided_slice_3/stack_1:output:0;lstm_28/while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_28/while/lstm_cell_28/MatMul_7MatMullstm_28_while_placeholder_23lstm_28/while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         А║
 lstm_28/while/lstm_cell_28/add_6AddV2-lstm_28/while/lstm_cell_28/BiasAdd_3:output:0-lstm_28/while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         Аg
"lstm_28/while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_28/while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_28/while/lstm_cell_28/Mul_4Mul$lstm_28/while/lstm_cell_28/add_6:z:0+lstm_28/while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         Ап
 lstm_28/while/lstm_cell_28/Add_7AddV2$lstm_28/while/lstm_cell_28/Mul_4:z:0+lstm_28/while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аy
4lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_28/while/lstm_cell_28/clip_by_value_2/MinimumMinimum$lstm_28/while/lstm_cell_28/Add_7:z:0=lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_28/while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_28/while/lstm_cell_28/clip_by_value_2Maximum6lstm_28/while/lstm_cell_28/clip_by_value_2/Minimum:z:05lstm_28/while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АВ
!lstm_28/while/lstm_cell_28/Relu_1Relu$lstm_28/while/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         А╗
 lstm_28/while/lstm_cell_28/mul_5Mul.lstm_28/while/lstm_cell_28/clip_by_value_2:z:0/lstm_28/while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Ах
2lstm_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_28_while_placeholder_1lstm_28_while_placeholder$lstm_28/while/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_28/while/addAddV2lstm_28_while_placeholderlstm_28/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_28/while/add_1AddV2(lstm_28_while_lstm_28_while_loop_counterlstm_28/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_28/while/IdentityIdentitylstm_28/while/add_1:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: К
lstm_28/while/Identity_1Identity.lstm_28_while_lstm_28_while_maximum_iterations^lstm_28/while/NoOp*
T0*
_output_shapes
: q
lstm_28/while/Identity_2Identitylstm_28/while/add:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: Ю
lstm_28/while/Identity_3IdentityBlstm_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_28/while/NoOp*
T0*
_output_shapes
: Т
lstm_28/while/Identity_4Identity$lstm_28/while/lstm_cell_28/mul_5:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:         АТ
lstm_28/while/Identity_5Identity$lstm_28/while/lstm_cell_28/add_5:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:         АЁ
lstm_28/while/NoOpNoOp*^lstm_28/while/lstm_cell_28/ReadVariableOp,^lstm_28/while/lstm_cell_28/ReadVariableOp_1,^lstm_28/while/lstm_cell_28/ReadVariableOp_2,^lstm_28/while/lstm_cell_28/ReadVariableOp_30^lstm_28/while/lstm_cell_28/split/ReadVariableOp2^lstm_28/while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_28_while_identity_1!lstm_28/while/Identity_1:output:0"=
lstm_28_while_identity_2!lstm_28/while/Identity_2:output:0"=
lstm_28_while_identity_3!lstm_28/while/Identity_3:output:0"=
lstm_28_while_identity_4!lstm_28/while/Identity_4:output:0"=
lstm_28_while_identity_5!lstm_28/while/Identity_5:output:0"9
lstm_28_while_identitylstm_28/while/Identity:output:0"P
%lstm_28_while_lstm_28_strided_slice_1'lstm_28_while_lstm_28_strided_slice_1_0"j
2lstm_28_while_lstm_cell_28_readvariableop_resource4lstm_28_while_lstm_cell_28_readvariableop_resource_0"z
:lstm_28_while_lstm_cell_28_split_1_readvariableop_resource<lstm_28_while_lstm_cell_28_split_1_readvariableop_resource_0"v
8lstm_28_while_lstm_cell_28_split_readvariableop_resource:lstm_28_while_lstm_cell_28_split_readvariableop_resource_0"╚
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2Z
+lstm_28/while/lstm_cell_28/ReadVariableOp_1+lstm_28/while/lstm_cell_28/ReadVariableOp_12Z
+lstm_28/while/lstm_cell_28/ReadVariableOp_2+lstm_28/while/lstm_cell_28/ReadVariableOp_22Z
+lstm_28/while/lstm_cell_28/ReadVariableOp_3+lstm_28/while/lstm_cell_28/ReadVariableOp_32V
)lstm_28/while/lstm_cell_28/ReadVariableOp)lstm_28/while/lstm_cell_28/ReadVariableOp2b
/lstm_28/while/lstm_cell_28/split/ReadVariableOp/lstm_28/while/lstm_cell_28/split/ReadVariableOp2f
1lstm_28/while/lstm_cell_28/split_1/ReadVariableOp1lstm_28/while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_28/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_28/while/loop_counter
╙#
х
while_body_233560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_28_233584_0:	А*
while_lstm_cell_28_233586_0:	А/
while_lstm_cell_28_233588_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_28_233584:	А(
while_lstm_cell_28_233586:	А-
while_lstm_cell_28_233588:
ААИв*while/lstm_cell_28/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╢
*while/lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_28_233584_0while_lstm_cell_28_233586_0while_lstm_cell_28_233588_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233501▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_28/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: С
while/Identity_4Identity3while/lstm_cell_28/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         АС
while/Identity_5Identity3while/lstm_cell_28/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Аy

while/NoOpNoOp+^while/lstm_cell_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_28_233584while_lstm_cell_28_233584_0"8
while_lstm_cell_28_233586while_lstm_cell_28_233586_0"8
while_lstm_cell_28_233588while_lstm_cell_28_233588_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2X
*while/lstm_cell_28/StatefulPartitionedCall*while/lstm_cell_28/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▄	
╔
.__inference_sequential_14_layer_call_fn_235416

inputs
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:
АА
	unknown_3:	А
	unknown_4:	@А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_14_layer_call_and_return_conditional_losses_234649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
·
╢
(__inference_lstm_29_layer_call_fn_237566

inputs
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
№╕
╩	
I__inference_sequential_14_layer_call_and_return_conditional_losses_235951

inputsE
2lstm_28_lstm_cell_28_split_readvariableop_resource:	АC
4lstm_28_lstm_cell_28_split_1_readvariableop_resource:	А@
,lstm_28_lstm_cell_28_readvariableop_resource:
ААF
2lstm_29_lstm_cell_29_split_readvariableop_resource:
ААC
4lstm_29_lstm_cell_29_split_1_readvariableop_resource:	А?
,lstm_29_lstm_cell_29_readvariableop_resource:	@А9
'dense_14_matmul_readvariableop_resource:@6
(dense_14_biasadd_readvariableop_resource:
identityИвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpв#lstm_28/lstm_cell_28/ReadVariableOpв%lstm_28/lstm_cell_28/ReadVariableOp_1в%lstm_28/lstm_cell_28/ReadVariableOp_2в%lstm_28/lstm_cell_28/ReadVariableOp_3в)lstm_28/lstm_cell_28/split/ReadVariableOpв+lstm_28/lstm_cell_28/split_1/ReadVariableOpвlstm_28/whileв#lstm_29/lstm_cell_29/ReadVariableOpв%lstm_29/lstm_cell_29/ReadVariableOp_1в%lstm_29/lstm_cell_29/ReadVariableOp_2в%lstm_29/lstm_cell_29/ReadVariableOp_3в)lstm_29/lstm_cell_29/split/ReadVariableOpв+lstm_29/lstm_cell_29/split_1/ReadVariableOpвlstm_29/whileQ
lstm_28/ShapeShapeinputs*
T0*
_output_shapes
::э╧e
lstm_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_28/strided_sliceStridedSlicelstm_28/Shape:output:0$lstm_28/strided_slice/stack:output:0&lstm_28/strided_slice/stack_1:output:0&lstm_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АЛ
lstm_28/zeros/packedPacklstm_28/strided_slice:output:0lstm_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
lstm_28/zerosFilllstm_28/zeros/packed:output:0lstm_28/zeros/Const:output:0*
T0*(
_output_shapes
:         А[
lstm_28/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АП
lstm_28/zeros_1/packedPacklstm_28/strided_slice:output:0!lstm_28/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_28/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
lstm_28/zeros_1Filllstm_28/zeros_1/packed:output:0lstm_28/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аk
lstm_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_28/transpose	Transposeinputslstm_28/transpose/perm:output:0*
T0*+
_output_shapes
:         b
lstm_28/Shape_1Shapelstm_28/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_28/strided_slice_1StridedSlicelstm_28/Shape_1:output:0&lstm_28/strided_slice_1/stack:output:0(lstm_28/strided_slice_1/stack_1:output:0(lstm_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_28/TensorArrayV2TensorListReserve,lstm_28/TensorArrayV2/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       °
/lstm_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_28/transpose:y:0Flstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
lstm_28/strided_slice_2StridedSlicelstm_28/transpose:y:0&lstm_28/strided_slice_2/stack:output:0(lstm_28/strided_slice_2/stack_1:output:0(lstm_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskf
$lstm_28/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Э
)lstm_28/lstm_cell_28/split/ReadVariableOpReadVariableOp2lstm_28_lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0с
lstm_28/lstm_cell_28/splitSplit-lstm_28/lstm_cell_28/split/split_dim:output:01lstm_28/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЯ
lstm_28/lstm_cell_28/MatMulMatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Аб
lstm_28/lstm_cell_28/MatMul_1MatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Аб
lstm_28/lstm_cell_28/MatMul_2MatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Аб
lstm_28/lstm_cell_28/MatMul_3MatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аh
&lstm_28/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_28/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4lstm_28_lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╫
lstm_28/lstm_cell_28/split_1Split/lstm_28/lstm_cell_28/split_1/split_dim:output:03lstm_28/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitи
lstm_28/lstm_cell_28/BiasAddBiasAdd%lstm_28/lstm_cell_28/MatMul:product:0%lstm_28/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Ам
lstm_28/lstm_cell_28/BiasAdd_1BiasAdd'lstm_28/lstm_cell_28/MatMul_1:product:0%lstm_28/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Ам
lstm_28/lstm_cell_28/BiasAdd_2BiasAdd'lstm_28/lstm_cell_28/MatMul_2:product:0%lstm_28/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Ам
lstm_28/lstm_cell_28/BiasAdd_3BiasAdd'lstm_28/lstm_cell_28/MatMul_3:product:0%lstm_28/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АТ
#lstm_28/lstm_cell_28/ReadVariableOpReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0y
(lstm_28/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_28/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*lstm_28/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"lstm_28/lstm_cell_28/strided_sliceStridedSlice+lstm_28/lstm_cell_28/ReadVariableOp:value:01lstm_28/lstm_cell_28/strided_slice/stack:output:03lstm_28/lstm_cell_28/strided_slice/stack_1:output:03lstm_28/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЯ
lstm_28/lstm_cell_28/MatMul_4MatMullstm_28/zeros:output:0+lstm_28/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         Ад
lstm_28/lstm_cell_28/addAddV2%lstm_28/lstm_cell_28/BiasAdd:output:0'lstm_28/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А_
lstm_28/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_28/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
lstm_28/lstm_cell_28/MulMullstm_28/lstm_cell_28/add:z:0#lstm_28/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АЫ
lstm_28/lstm_cell_28/Add_1AddV2lstm_28/lstm_cell_28/Mul:z:0%lstm_28/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аq
,lstm_28/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┐
*lstm_28/lstm_cell_28/clip_by_value/MinimumMinimumlstm_28/lstm_cell_28/Add_1:z:05lstm_28/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$lstm_28/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"lstm_28/lstm_cell_28/clip_by_valueMaximum.lstm_28/lstm_cell_28/clip_by_value/Minimum:z:0-lstm_28/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АФ
%lstm_28/lstm_cell_28/ReadVariableOp_1ReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_28/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_28/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_28/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_28/lstm_cell_28/strided_slice_1StridedSlice-lstm_28/lstm_cell_28/ReadVariableOp_1:value:03lstm_28/lstm_cell_28/strided_slice_1/stack:output:05lstm_28/lstm_cell_28/strided_slice_1/stack_1:output:05lstm_28/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_28/lstm_cell_28/MatMul_5MatMullstm_28/zeros:output:0-lstm_28/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Аи
lstm_28/lstm_cell_28/add_2AddV2'lstm_28/lstm_cell_28/BiasAdd_1:output:0'lstm_28/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         Аa
lstm_28/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_28/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_28/lstm_cell_28/Mul_1Mullstm_28/lstm_cell_28/add_2:z:0%lstm_28/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЭ
lstm_28/lstm_cell_28/Add_3AddV2lstm_28/lstm_cell_28/Mul_1:z:0%lstm_28/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аs
.lstm_28/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_28/lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_28/lstm_cell_28/Add_3:z:07lstm_28/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_28/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_28/lstm_cell_28/clip_by_value_1Maximum0lstm_28/lstm_cell_28/clip_by_value_1/Minimum:z:0/lstm_28/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АШ
lstm_28/lstm_cell_28/mul_2Mul(lstm_28/lstm_cell_28/clip_by_value_1:z:0lstm_28/zeros_1:output:0*
T0*(
_output_shapes
:         АФ
%lstm_28/lstm_cell_28/ReadVariableOp_2ReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_28/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_28/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_28/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_28/lstm_cell_28/strided_slice_2StridedSlice-lstm_28/lstm_cell_28/ReadVariableOp_2:value:03lstm_28/lstm_cell_28/strided_slice_2/stack:output:05lstm_28/lstm_cell_28/strided_slice_2/stack_1:output:05lstm_28/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_28/lstm_cell_28/MatMul_6MatMullstm_28/zeros:output:0-lstm_28/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Аи
lstm_28/lstm_cell_28/add_4AddV2'lstm_28/lstm_cell_28/BiasAdd_2:output:0'lstm_28/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аt
lstm_28/lstm_cell_28/ReluRelulstm_28/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         Ае
lstm_28/lstm_cell_28/mul_3Mul&lstm_28/lstm_cell_28/clip_by_value:z:0'lstm_28/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АЦ
lstm_28/lstm_cell_28/add_5AddV2lstm_28/lstm_cell_28/mul_2:z:0lstm_28/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АФ
%lstm_28/lstm_cell_28/ReadVariableOp_3ReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_28/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_28/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_28/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_28/lstm_cell_28/strided_slice_3StridedSlice-lstm_28/lstm_cell_28/ReadVariableOp_3:value:03lstm_28/lstm_cell_28/strided_slice_3/stack:output:05lstm_28/lstm_cell_28/strided_slice_3/stack_1:output:05lstm_28/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_28/lstm_cell_28/MatMul_7MatMullstm_28/zeros:output:0-lstm_28/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Аи
lstm_28/lstm_cell_28/add_6AddV2'lstm_28/lstm_cell_28/BiasAdd_3:output:0'lstm_28/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         Аa
lstm_28/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_28/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_28/lstm_cell_28/Mul_4Mullstm_28/lstm_cell_28/add_6:z:0%lstm_28/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЭ
lstm_28/lstm_cell_28/Add_7AddV2lstm_28/lstm_cell_28/Mul_4:z:0%lstm_28/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аs
.lstm_28/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_28/lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_28/lstm_cell_28/Add_7:z:07lstm_28/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_28/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_28/lstm_cell_28/clip_by_value_2Maximum0lstm_28/lstm_cell_28/clip_by_value_2/Minimum:z:0/lstm_28/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аv
lstm_28/lstm_cell_28/Relu_1Relulstm_28/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Ай
lstm_28/lstm_cell_28/mul_5Mul(lstm_28/lstm_cell_28/clip_by_value_2:z:0)lstm_28/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аv
%lstm_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╨
lstm_28/TensorArrayV2_1TensorListReserve.lstm_28/TensorArrayV2_1/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_28/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ь
lstm_28/whileWhile#lstm_28/while/loop_counter:output:0)lstm_28/while/maximum_iterations:output:0lstm_28/time:output:0 lstm_28/TensorArrayV2_1:handle:0lstm_28/zeros:output:0lstm_28/zeros_1:output:0 lstm_28/strided_slice_1:output:0?lstm_28/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_28_lstm_cell_28_split_readvariableop_resource4lstm_28_lstm_cell_28_split_1_readvariableop_resource,lstm_28_lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_28_while_body_235553*%
condR
lstm_28_while_cond_235552*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Й
8lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   █
*lstm_28/TensorArrayV2Stack/TensorListStackTensorListStacklstm_28/while:output:3Alstm_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0p
lstm_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_28/strided_slice_3StridedSlice3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_28/strided_slice_3/stack:output:0(lstm_28/strided_slice_3/stack_1:output:0(lstm_28/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskm
lstm_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
lstm_28/transpose_1	Transpose3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_28/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Аb
lstm_29/ShapeShapelstm_28/transpose_1:y:0*
T0*
_output_shapes
::э╧e
lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_29/strided_sliceStridedSlicelstm_29/Shape:output:0$lstm_29/strided_slice/stack:output:0&lstm_29/strided_slice/stack_1:output:0&lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Л
lstm_29/zeros/packedPacklstm_29/strided_slice:output:0lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
lstm_29/zerosFilllstm_29/zeros/packed:output:0lstm_29/zeros/Const:output:0*
T0*'
_output_shapes
:         @Z
lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@П
lstm_29/zeros_1/packedPacklstm_29/strided_slice:output:0!lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
lstm_29/zeros_1Filllstm_29/zeros_1/packed:output:0lstm_29/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @k
lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          П
lstm_29/transpose	Transposelstm_28/transpose_1:y:0lstm_29/transpose/perm:output:0*
T0*,
_output_shapes
:         Аb
lstm_29/Shape_1Shapelstm_29/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_29/strided_slice_1StridedSlicelstm_29/Shape_1:output:0&lstm_29/strided_slice_1/stack:output:0(lstm_29/strided_slice_1/stack_1:output:0(lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_29/TensorArrayV2TensorListReserve,lstm_29/TensorArrayV2/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   °
/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_29/transpose:y:0Flstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
lstm_29/strided_slice_2StridedSlicelstm_29/transpose:y:0&lstm_29/strided_slice_2/stack:output:0(lstm_29/strided_slice_2/stack_1:output:0(lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskf
$lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
)lstm_29/lstm_cell_29/split/ReadVariableOpReadVariableOp2lstm_29_lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0с
lstm_29/lstm_cell_29/splitSplit-lstm_29/lstm_cell_29/split/split_dim:output:01lstm_29/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЮ
lstm_29/lstm_cell_29/MatMulMatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @а
lstm_29/lstm_cell_29/MatMul_1MatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @а
lstm_29/lstm_cell_29/MatMul_2MatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @а
lstm_29/lstm_cell_29/MatMul_3MatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @h
&lstm_29/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_29/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4lstm_29_lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
lstm_29/lstm_cell_29/split_1Split/lstm_29/lstm_cell_29/split_1/split_dim:output:03lstm_29/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitз
lstm_29/lstm_cell_29/BiasAddBiasAdd%lstm_29/lstm_cell_29/MatMul:product:0%lstm_29/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @л
lstm_29/lstm_cell_29/BiasAdd_1BiasAdd'lstm_29/lstm_cell_29/MatMul_1:product:0%lstm_29/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @л
lstm_29/lstm_cell_29/BiasAdd_2BiasAdd'lstm_29/lstm_cell_29/MatMul_2:product:0%lstm_29/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @л
lstm_29/lstm_cell_29/BiasAdd_3BiasAdd'lstm_29/lstm_cell_29/MatMul_3:product:0%lstm_29/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @С
#lstm_29/lstm_cell_29/ReadVariableOpReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0y
(lstm_29/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_29/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm_29/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"lstm_29/lstm_cell_29/strided_sliceStridedSlice+lstm_29/lstm_cell_29/ReadVariableOp:value:01lstm_29/lstm_cell_29/strided_slice/stack:output:03lstm_29/lstm_cell_29/strided_slice/stack_1:output:03lstm_29/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЮ
lstm_29/lstm_cell_29/MatMul_4MatMullstm_29/zeros:output:0+lstm_29/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @г
lstm_29/lstm_cell_29/addAddV2%lstm_29/lstm_cell_29/BiasAdd:output:0'lstm_29/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @_
lstm_29/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_29/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
lstm_29/lstm_cell_29/MulMullstm_29/lstm_cell_29/add:z:0#lstm_29/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ъ
lstm_29/lstm_cell_29/Add_1AddV2lstm_29/lstm_cell_29/Mul:z:0%lstm_29/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @q
,lstm_29/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╛
*lstm_29/lstm_cell_29/clip_by_value/MinimumMinimumlstm_29/lstm_cell_29/Add_1:z:05lstm_29/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$lstm_29/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"lstm_29/lstm_cell_29/clip_by_valueMaximum.lstm_29/lstm_cell_29/clip_by_value/Minimum:z:0-lstm_29/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @У
%lstm_29/lstm_cell_29/ReadVariableOp_1ReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_29/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   }
,lstm_29/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_29/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_29/lstm_cell_29/strided_slice_1StridedSlice-lstm_29/lstm_cell_29/ReadVariableOp_1:value:03lstm_29/lstm_cell_29/strided_slice_1/stack:output:05lstm_29/lstm_cell_29/strided_slice_1/stack_1:output:05lstm_29/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_29/lstm_cell_29/MatMul_5MatMullstm_29/zeros:output:0-lstm_29/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @з
lstm_29/lstm_cell_29/add_2AddV2'lstm_29/lstm_cell_29/BiasAdd_1:output:0'lstm_29/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @a
lstm_29/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_29/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_29/lstm_cell_29/Mul_1Mullstm_29/lstm_cell_29/add_2:z:0%lstm_29/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ь
lstm_29/lstm_cell_29/Add_3AddV2lstm_29/lstm_cell_29/Mul_1:z:0%lstm_29/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @s
.lstm_29/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_29/lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_29/lstm_cell_29/Add_3:z:07lstm_29/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_29/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_29/lstm_cell_29/clip_by_value_1Maximum0lstm_29/lstm_cell_29/clip_by_value_1/Minimum:z:0/lstm_29/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @Ч
lstm_29/lstm_cell_29/mul_2Mul(lstm_29/lstm_cell_29/clip_by_value_1:z:0lstm_29/zeros_1:output:0*
T0*'
_output_shapes
:         @У
%lstm_29/lstm_cell_29/ReadVariableOp_2ReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_29/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_29/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_29/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_29/lstm_cell_29/strided_slice_2StridedSlice-lstm_29/lstm_cell_29/ReadVariableOp_2:value:03lstm_29/lstm_cell_29/strided_slice_2/stack:output:05lstm_29/lstm_cell_29/strided_slice_2/stack_1:output:05lstm_29/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_29/lstm_cell_29/MatMul_6MatMullstm_29/zeros:output:0-lstm_29/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @з
lstm_29/lstm_cell_29/add_4AddV2'lstm_29/lstm_cell_29/BiasAdd_2:output:0'lstm_29/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @s
lstm_29/lstm_cell_29/ReluRelulstm_29/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @д
lstm_29/lstm_cell_29/mul_3Mul&lstm_29/lstm_cell_29/clip_by_value:z:0'lstm_29/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @Х
lstm_29/lstm_cell_29/add_5AddV2lstm_29/lstm_cell_29/mul_2:z:0lstm_29/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @У
%lstm_29/lstm_cell_29/ReadVariableOp_3ReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_29/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_29/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_29/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_29/lstm_cell_29/strided_slice_3StridedSlice-lstm_29/lstm_cell_29/ReadVariableOp_3:value:03lstm_29/lstm_cell_29/strided_slice_3/stack:output:05lstm_29/lstm_cell_29/strided_slice_3/stack_1:output:05lstm_29/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_29/lstm_cell_29/MatMul_7MatMullstm_29/zeros:output:0-lstm_29/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @з
lstm_29/lstm_cell_29/add_6AddV2'lstm_29/lstm_cell_29/BiasAdd_3:output:0'lstm_29/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @a
lstm_29/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_29/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_29/lstm_cell_29/Mul_4Mullstm_29/lstm_cell_29/add_6:z:0%lstm_29/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ь
lstm_29/lstm_cell_29/Add_7AddV2lstm_29/lstm_cell_29/Mul_4:z:0%lstm_29/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @s
.lstm_29/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_29/lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_29/lstm_cell_29/Add_7:z:07lstm_29/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_29/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_29/lstm_cell_29/clip_by_value_2Maximum0lstm_29/lstm_cell_29/clip_by_value_2/Minimum:z:0/lstm_29/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @u
lstm_29/lstm_cell_29/Relu_1Relulstm_29/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @и
lstm_29/lstm_cell_29/mul_5Mul(lstm_29/lstm_cell_29/clip_by_value_2:z:0)lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @v
%lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╨
lstm_29/TensorArrayV2_1TensorListReserve.lstm_29/TensorArrayV2_1/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
lstm_29/whileWhile#lstm_29/while/loop_counter:output:0)lstm_29/while/maximum_iterations:output:0lstm_29/time:output:0 lstm_29/TensorArrayV2_1:handle:0lstm_29/zeros:output:0lstm_29/zeros_1:output:0 lstm_29/strided_slice_1:output:0?lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_29_lstm_cell_29_split_readvariableop_resource4lstm_29_lstm_cell_29_split_1_readvariableop_resource,lstm_29_lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_29_while_body_235805*%
condR
lstm_29_while_cond_235804*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Й
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┌
*lstm_29/TensorArrayV2Stack/TensorListStackTensorListStacklstm_29/while:output:3Alstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0p
lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
lstm_29/strided_slice_3StridedSlice3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_29/strided_slice_3/stack:output:0(lstm_29/strided_slice_3/stack_1:output:0(lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskm
lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
lstm_29/transpose_1	Transpose3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_29/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @Ж
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Х
dense_14/MatMulMatMul lstm_29/strided_slice_3:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp$^lstm_28/lstm_cell_28/ReadVariableOp&^lstm_28/lstm_cell_28/ReadVariableOp_1&^lstm_28/lstm_cell_28/ReadVariableOp_2&^lstm_28/lstm_cell_28/ReadVariableOp_3*^lstm_28/lstm_cell_28/split/ReadVariableOp,^lstm_28/lstm_cell_28/split_1/ReadVariableOp^lstm_28/while$^lstm_29/lstm_cell_29/ReadVariableOp&^lstm_29/lstm_cell_29/ReadVariableOp_1&^lstm_29/lstm_cell_29/ReadVariableOp_2&^lstm_29/lstm_cell_29/ReadVariableOp_3*^lstm_29/lstm_cell_29/split/ReadVariableOp,^lstm_29/lstm_cell_29/split_1/ReadVariableOp^lstm_29/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2N
%lstm_28/lstm_cell_28/ReadVariableOp_1%lstm_28/lstm_cell_28/ReadVariableOp_12N
%lstm_28/lstm_cell_28/ReadVariableOp_2%lstm_28/lstm_cell_28/ReadVariableOp_22N
%lstm_28/lstm_cell_28/ReadVariableOp_3%lstm_28/lstm_cell_28/ReadVariableOp_32J
#lstm_28/lstm_cell_28/ReadVariableOp#lstm_28/lstm_cell_28/ReadVariableOp2V
)lstm_28/lstm_cell_28/split/ReadVariableOp)lstm_28/lstm_cell_28/split/ReadVariableOp2Z
+lstm_28/lstm_cell_28/split_1/ReadVariableOp+lstm_28/lstm_cell_28/split_1/ReadVariableOp2
lstm_28/whilelstm_28/while2N
%lstm_29/lstm_cell_29/ReadVariableOp_1%lstm_29/lstm_cell_29/ReadVariableOp_12N
%lstm_29/lstm_cell_29/ReadVariableOp_2%lstm_29/lstm_cell_29/ReadVariableOp_22N
%lstm_29/lstm_cell_29/ReadVariableOp_3%lstm_29/lstm_cell_29/ReadVariableOp_32J
#lstm_29/lstm_cell_29/ReadVariableOp#lstm_29/lstm_cell_29/ReadVariableOp2V
)lstm_29/lstm_cell_29/split/ReadVariableOp)lstm_29/lstm_cell_29/split/ReadVariableOp2Z
+lstm_29/lstm_cell_29/split_1/ReadVariableOp+lstm_29/lstm_cell_29/split_1/ReadVariableOp2
lstm_29/whilelstm_29/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ы	
├
while_cond_236624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236624___redundant_placeholder04
0while_while_cond_236624___redundant_placeholder14
0while_while_cond_236624___redundant_placeholder24
0while_while_cond_236624___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ў
ў
-__inference_lstm_cell_28_layer_call_fn_238654

inputs
states_0
states_1
unknown:	А
	unknown_0:	А
	unknown_1:
АА
identity

identity_1

identity_2ИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233501p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Аr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         А:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:         А
"
_user_specified_name
states_1:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦#
х
while_body_234022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_29_234046_0:
АА*
while_lstm_cell_29_234048_0:	А.
while_lstm_cell_29_234050_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_29_234046:
АА(
while_lstm_cell_29_234048:	А,
while_lstm_cell_29_234050:	@АИв*while/lstm_cell_29/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0│
*while/lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_29_234046_0while_lstm_cell_29_234048_0while_lstm_cell_29_234050_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233963▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_29/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Р
while/Identity_4Identity3while/lstm_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @Р
while/Identity_5Identity3while/lstm_cell_29/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @y

while/NoOpNoOp+^while/lstm_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_29_234046while_lstm_cell_29_234046_0"8
while_lstm_cell_29_234048while_lstm_cell_29_234048_0"8
while_lstm_cell_29_234050while_lstm_cell_29_234050_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2X
*while/lstm_cell_29/StatefulPartitionedCall*while/lstm_cell_29/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ч	
├
while_cond_234483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234483___redundant_placeholder04
0while_while_cond_234483___redundant_placeholder14
0while_while_cond_234483___redundant_placeholder24
0while_while_cond_234483___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ы	
├
while_cond_233312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_233312___redundant_placeholder04
0while_while_cond_233312___redundant_placeholder14
0while_while_cond_233312___redundant_placeholder24
0while_while_cond_233312___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
б~
ж	
while_body_236625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_28_split_readvariableop_resource_0:	АC
4while_lstm_cell_28_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_28_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_28_split_readvariableop_resource:	АA
2while_lstm_cell_28_split_1_readvariableop_resource:	А>
*while_lstm_cell_28_readvariableop_resource:
ААИв!while/lstm_cell_28/ReadVariableOpв#while/lstm_cell_28/ReadVariableOp_1в#while/lstm_cell_28/ReadVariableOp_2в#while/lstm_cell_28/ReadVariableOp_3в'while/lstm_cell_28/split/ReadVariableOpв)while/lstm_cell_28/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_28/split/ReadVariableOpReadVariableOp2while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_28/split_1Split-while/lstm_cell_28/split_1/split_dim:output:01while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_28/BiasAddBiasAdd#while/lstm_cell_28/MatMul:product:0#while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_1BiasAdd%while/lstm_cell_28/MatMul_1:product:0#while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_2BiasAdd%while/lstm_cell_28/MatMul_2:product:0#while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_3BiasAdd%while/lstm_cell_28/MatMul_3:product:0#while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_28/ReadVariableOpReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_28/strided_sliceStridedSlice)while/lstm_cell_28/ReadVariableOp:value:0/while/lstm_cell_28/strided_slice/stack:output:01while/lstm_cell_28/strided_slice/stack_1:output:01while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_28/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_28/addAddV2#while/lstm_cell_28/BiasAdd:output:0%while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_28/MulMulwhile/lstm_cell_28/add:z:0!while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_28/Add_1AddV2while/lstm_cell_28/Mul:z:0#while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_28/clip_by_value/MinimumMinimumwhile/lstm_cell_28/Add_1:z:03while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_28/clip_by_valueMaximum,while/lstm_cell_28/clip_by_value/Minimum:z:0+while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_1ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_1StridedSlice+while/lstm_cell_28/ReadVariableOp_1:value:01while/lstm_cell_28/strided_slice_1/stack:output:03while/lstm_cell_28/strided_slice_1/stack_1:output:03while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_2AddV2%while/lstm_cell_28/BiasAdd_1:output:0%while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_1Mulwhile/lstm_cell_28/add_2:z:0#while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_3AddV2while/lstm_cell_28/Mul_1:z:0#while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_1/MinimumMinimumwhile/lstm_cell_28/Add_3:z:05while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_1Maximum.while/lstm_cell_28/clip_by_value_1/Minimum:z:0-while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_28/mul_2Mul&while/lstm_cell_28/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_2ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_2StridedSlice+while/lstm_cell_28/ReadVariableOp_2:value:01while/lstm_cell_28/strided_slice_2/stack:output:03while/lstm_cell_28/strided_slice_2/stack_1:output:03while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_4AddV2%while/lstm_cell_28/BiasAdd_2:output:0%while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_28/ReluReluwhile/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_28/mul_3Mul$while/lstm_cell_28/clip_by_value:z:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_28/add_5AddV2while/lstm_cell_28/mul_2:z:0while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_3ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_3StridedSlice+while/lstm_cell_28/ReadVariableOp_3:value:01while/lstm_cell_28/strided_slice_3/stack:output:03while/lstm_cell_28/strided_slice_3/stack_1:output:03while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_6AddV2%while/lstm_cell_28/BiasAdd_3:output:0%while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_4Mulwhile/lstm_cell_28/add_6:z:0#while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_7AddV2while/lstm_cell_28/Mul_4:z:0#while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_2/MinimumMinimumwhile/lstm_cell_28/Add_7:z:05while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_2Maximum.while/lstm_cell_28/clip_by_value_2/Minimum:z:0-while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_28/mul_5Mul&while/lstm_cell_28/clip_by_value_2:z:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_28/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_28/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_28/ReadVariableOp$^while/lstm_cell_28/ReadVariableOp_1$^while/lstm_cell_28/ReadVariableOp_2$^while/lstm_cell_28/ReadVariableOp_3(^while/lstm_cell_28/split/ReadVariableOp*^while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_28_readvariableop_resource,while_lstm_cell_28_readvariableop_resource_0"j
2while_lstm_cell_28_split_1_readvariableop_resource4while_lstm_cell_28_split_1_readvariableop_resource_0"f
0while_lstm_cell_28_split_readvariableop_resource2while_lstm_cell_28_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_28/ReadVariableOp_1#while/lstm_cell_28/ReadVariableOp_12J
#while/lstm_cell_28/ReadVariableOp_2#while/lstm_cell_28/ReadVariableOp_22J
#while/lstm_cell_28/ReadVariableOp_3#while/lstm_cell_28/ReadVariableOp_32F
!while/lstm_cell_28/ReadVariableOp!while/lstm_cell_28/ReadVariableOp2R
'while/lstm_cell_28/split/ReadVariableOp'while/lstm_cell_28/split/ReadVariableOp2V
)while/lstm_cell_28/split_1/ReadVariableOp)while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ч	
├
while_cond_238460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_238460___redundant_placeholder04
0while_while_cond_238460___redundant_placeholder14
0while_while_cond_238460___redundant_placeholder24
0while_while_cond_238460___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
·
╢
(__inference_lstm_29_layer_call_fn_237577

inputs
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234947o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
б~
ж	
while_body_235085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_28_split_readvariableop_resource_0:	АC
4while_lstm_cell_28_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_28_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_28_split_readvariableop_resource:	АA
2while_lstm_cell_28_split_1_readvariableop_resource:	А>
*while_lstm_cell_28_readvariableop_resource:
ААИв!while/lstm_cell_28/ReadVariableOpв#while/lstm_cell_28/ReadVariableOp_1в#while/lstm_cell_28/ReadVariableOp_2в#while/lstm_cell_28/ReadVariableOp_3в'while/lstm_cell_28/split/ReadVariableOpв)while/lstm_cell_28/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_28/split/ReadVariableOpReadVariableOp2while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_28/split_1Split-while/lstm_cell_28/split_1/split_dim:output:01while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_28/BiasAddBiasAdd#while/lstm_cell_28/MatMul:product:0#while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_1BiasAdd%while/lstm_cell_28/MatMul_1:product:0#while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_2BiasAdd%while/lstm_cell_28/MatMul_2:product:0#while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_3BiasAdd%while/lstm_cell_28/MatMul_3:product:0#while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_28/ReadVariableOpReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_28/strided_sliceStridedSlice)while/lstm_cell_28/ReadVariableOp:value:0/while/lstm_cell_28/strided_slice/stack:output:01while/lstm_cell_28/strided_slice/stack_1:output:01while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_28/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_28/addAddV2#while/lstm_cell_28/BiasAdd:output:0%while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_28/MulMulwhile/lstm_cell_28/add:z:0!while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_28/Add_1AddV2while/lstm_cell_28/Mul:z:0#while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_28/clip_by_value/MinimumMinimumwhile/lstm_cell_28/Add_1:z:03while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_28/clip_by_valueMaximum,while/lstm_cell_28/clip_by_value/Minimum:z:0+while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_1ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_1StridedSlice+while/lstm_cell_28/ReadVariableOp_1:value:01while/lstm_cell_28/strided_slice_1/stack:output:03while/lstm_cell_28/strided_slice_1/stack_1:output:03while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_2AddV2%while/lstm_cell_28/BiasAdd_1:output:0%while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_1Mulwhile/lstm_cell_28/add_2:z:0#while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_3AddV2while/lstm_cell_28/Mul_1:z:0#while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_1/MinimumMinimumwhile/lstm_cell_28/Add_3:z:05while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_1Maximum.while/lstm_cell_28/clip_by_value_1/Minimum:z:0-while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_28/mul_2Mul&while/lstm_cell_28/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_2ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_2StridedSlice+while/lstm_cell_28/ReadVariableOp_2:value:01while/lstm_cell_28/strided_slice_2/stack:output:03while/lstm_cell_28/strided_slice_2/stack_1:output:03while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_4AddV2%while/lstm_cell_28/BiasAdd_2:output:0%while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_28/ReluReluwhile/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_28/mul_3Mul$while/lstm_cell_28/clip_by_value:z:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_28/add_5AddV2while/lstm_cell_28/mul_2:z:0while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_3ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_3StridedSlice+while/lstm_cell_28/ReadVariableOp_3:value:01while/lstm_cell_28/strided_slice_3/stack:output:03while/lstm_cell_28/strided_slice_3/stack_1:output:03while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_6AddV2%while/lstm_cell_28/BiasAdd_3:output:0%while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_4Mulwhile/lstm_cell_28/add_6:z:0#while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_7AddV2while/lstm_cell_28/Mul_4:z:0#while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_2/MinimumMinimumwhile/lstm_cell_28/Add_7:z:05while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_2Maximum.while/lstm_cell_28/clip_by_value_2/Minimum:z:0-while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_28/mul_5Mul&while/lstm_cell_28/clip_by_value_2:z:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_28/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_28/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_28/ReadVariableOp$^while/lstm_cell_28/ReadVariableOp_1$^while/lstm_cell_28/ReadVariableOp_2$^while/lstm_cell_28/ReadVariableOp_3(^while/lstm_cell_28/split/ReadVariableOp*^while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_28_readvariableop_resource,while_lstm_cell_28_readvariableop_resource_0"j
2while_lstm_cell_28_split_1_readvariableop_resource4while_lstm_cell_28_split_1_readvariableop_resource_0"f
0while_lstm_cell_28_split_readvariableop_resource2while_lstm_cell_28_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_28/ReadVariableOp_1#while/lstm_cell_28/ReadVariableOp_12J
#while/lstm_cell_28/ReadVariableOp_2#while/lstm_cell_28/ReadVariableOp_22J
#while/lstm_cell_28/ReadVariableOp_3#while/lstm_cell_28/ReadVariableOp_32F
!while/lstm_cell_28/ReadVariableOp!while/lstm_cell_28/ReadVariableOp2R
'while/lstm_cell_28/split/ReadVariableOp'while/lstm_cell_28/split/ReadVariableOp2V
)while/lstm_cell_28/split_1/ReadVariableOp)while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┐	
╞
$__inference_signature_wrapper_235395
lstm_28_input
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:
АА
	unknown_3:	А
	unknown_4:	@А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCalllstm_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_233175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_28_input
Ч	
├
while_cond_234021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234021___redundant_placeholder04
0while_while_cond_234021___redundant_placeholder14
0while_while_cond_234021___redundant_placeholder24
0while_while_cond_234021___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Т
╕
(__inference_lstm_29_layer_call_fn_237544
inputs_0
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_233843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  А
"
_user_specified_name
inputs_0
тJ
м
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_239044

inputs
states_0
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:         @\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:         @\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:         @\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ы
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:         @[
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:         @\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskh
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:         @]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:         @e
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*'
_output_shapes
:         @V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskh
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:         @]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_5:z:0*
T0*'
_output_shapes
:         @i
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*'
_output_shapes
:         @X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:         @└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         @:         @: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б~
ж	
while_body_236881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_28_split_readvariableop_resource_0:	АC
4while_lstm_cell_28_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_28_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_28_split_readvariableop_resource:	АA
2while_lstm_cell_28_split_1_readvariableop_resource:	А>
*while_lstm_cell_28_readvariableop_resource:
ААИв!while/lstm_cell_28/ReadVariableOpв#while/lstm_cell_28/ReadVariableOp_1в#while/lstm_cell_28/ReadVariableOp_2в#while/lstm_cell_28/ReadVariableOp_3в'while/lstm_cell_28/split/ReadVariableOpв)while/lstm_cell_28/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_28/split/ReadVariableOpReadVariableOp2while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_28/split_1Split-while/lstm_cell_28/split_1/split_dim:output:01while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_28/BiasAddBiasAdd#while/lstm_cell_28/MatMul:product:0#while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_1BiasAdd%while/lstm_cell_28/MatMul_1:product:0#while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_2BiasAdd%while/lstm_cell_28/MatMul_2:product:0#while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_3BiasAdd%while/lstm_cell_28/MatMul_3:product:0#while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_28/ReadVariableOpReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_28/strided_sliceStridedSlice)while/lstm_cell_28/ReadVariableOp:value:0/while/lstm_cell_28/strided_slice/stack:output:01while/lstm_cell_28/strided_slice/stack_1:output:01while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_28/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_28/addAddV2#while/lstm_cell_28/BiasAdd:output:0%while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_28/MulMulwhile/lstm_cell_28/add:z:0!while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_28/Add_1AddV2while/lstm_cell_28/Mul:z:0#while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_28/clip_by_value/MinimumMinimumwhile/lstm_cell_28/Add_1:z:03while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_28/clip_by_valueMaximum,while/lstm_cell_28/clip_by_value/Minimum:z:0+while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_1ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_1StridedSlice+while/lstm_cell_28/ReadVariableOp_1:value:01while/lstm_cell_28/strided_slice_1/stack:output:03while/lstm_cell_28/strided_slice_1/stack_1:output:03while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_2AddV2%while/lstm_cell_28/BiasAdd_1:output:0%while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_1Mulwhile/lstm_cell_28/add_2:z:0#while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_3AddV2while/lstm_cell_28/Mul_1:z:0#while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_1/MinimumMinimumwhile/lstm_cell_28/Add_3:z:05while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_1Maximum.while/lstm_cell_28/clip_by_value_1/Minimum:z:0-while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_28/mul_2Mul&while/lstm_cell_28/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_2ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_2StridedSlice+while/lstm_cell_28/ReadVariableOp_2:value:01while/lstm_cell_28/strided_slice_2/stack:output:03while/lstm_cell_28/strided_slice_2/stack_1:output:03while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_4AddV2%while/lstm_cell_28/BiasAdd_2:output:0%while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_28/ReluReluwhile/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_28/mul_3Mul$while/lstm_cell_28/clip_by_value:z:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_28/add_5AddV2while/lstm_cell_28/mul_2:z:0while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_3ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_3StridedSlice+while/lstm_cell_28/ReadVariableOp_3:value:01while/lstm_cell_28/strided_slice_3/stack:output:03while/lstm_cell_28/strided_slice_3/stack_1:output:03while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_6AddV2%while/lstm_cell_28/BiasAdd_3:output:0%while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_4Mulwhile/lstm_cell_28/add_6:z:0#while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_7AddV2while/lstm_cell_28/Mul_4:z:0#while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_2/MinimumMinimumwhile/lstm_cell_28/Add_7:z:05while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_2Maximum.while/lstm_cell_28/clip_by_value_2/Minimum:z:0-while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_28/mul_5Mul&while/lstm_cell_28/clip_by_value_2:z:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_28/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_28/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_28/ReadVariableOp$^while/lstm_cell_28/ReadVariableOp_1$^while/lstm_cell_28/ReadVariableOp_2$^while/lstm_cell_28/ReadVariableOp_3(^while/lstm_cell_28/split/ReadVariableOp*^while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_28_readvariableop_resource,while_lstm_cell_28_readvariableop_resource_0"j
2while_lstm_cell_28_split_1_readvariableop_resource4while_lstm_cell_28_split_1_readvariableop_resource_0"f
0while_lstm_cell_28_split_readvariableop_resource2while_lstm_cell_28_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_28/ReadVariableOp_1#while/lstm_cell_28/ReadVariableOp_12J
#while/lstm_cell_28/ReadVariableOp_2#while/lstm_cell_28/ReadVariableOp_22J
#while/lstm_cell_28/ReadVariableOp_3#while/lstm_cell_28/ReadVariableOp_32F
!while/lstm_cell_28/ReadVariableOp!while/lstm_cell_28/ReadVariableOp2R
'while/lstm_cell_28/split/ReadVariableOp'while/lstm_cell_28/split/ReadVariableOp2V
)while/lstm_cell_28/split_1/ReadVariableOp)while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
░
√
'sequential_14_lstm_28_while_cond_232776H
Dsequential_14_lstm_28_while_sequential_14_lstm_28_while_loop_counterN
Jsequential_14_lstm_28_while_sequential_14_lstm_28_while_maximum_iterations+
'sequential_14_lstm_28_while_placeholder-
)sequential_14_lstm_28_while_placeholder_1-
)sequential_14_lstm_28_while_placeholder_2-
)sequential_14_lstm_28_while_placeholder_3J
Fsequential_14_lstm_28_while_less_sequential_14_lstm_28_strided_slice_1`
\sequential_14_lstm_28_while_sequential_14_lstm_28_while_cond_232776___redundant_placeholder0`
\sequential_14_lstm_28_while_sequential_14_lstm_28_while_cond_232776___redundant_placeholder1`
\sequential_14_lstm_28_while_sequential_14_lstm_28_while_cond_232776___redundant_placeholder2`
\sequential_14_lstm_28_while_sequential_14_lstm_28_while_cond_232776___redundant_placeholder3(
$sequential_14_lstm_28_while_identity
║
 sequential_14/lstm_28/while/LessLess'sequential_14_lstm_28_while_placeholderFsequential_14_lstm_28_while_less_sequential_14_lstm_28_strided_slice_1*
T0*
_output_shapes
: w
$sequential_14/lstm_28/while/IdentityIdentity$sequential_14/lstm_28/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_14_lstm_28_while_identity-sequential_14/lstm_28/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_14/lstm_28/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_14/lstm_28/while/loop_counter
Е
ф
I__inference_sequential_14_layer_call_and_return_conditional_losses_235343
lstm_28_input!
lstm_28_235323:	А
lstm_28_235325:	А"
lstm_28_235327:
АА"
lstm_29_235330:
АА
lstm_29_235332:	А!
lstm_29_235334:	@А!
dense_14_235337:@
dense_14_235339:
identityИв dense_14/StatefulPartitionedCallвlstm_28/StatefulPartitionedCallвlstm_29/StatefulPartitionedCallК
lstm_28/StatefulPartitionedCallStatefulPartitionedCalllstm_28_inputlstm_28_235323lstm_28_235325lstm_28_235327*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_234361а
lstm_29/StatefulPartitionedCallStatefulPartitionedCall(lstm_28/StatefulPartitionedCall:output:0lstm_29_235330lstm_29_235332lstm_29_235334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234624Т
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0dense_14_235337dense_14_235339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_234642x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_14/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_28_input
с7
Ж
C__inference_lstm_28_layer_call_and_return_conditional_losses_233381

inputs&
lstm_cell_28_233300:	А"
lstm_cell_28_233302:	А'
lstm_cell_28_233304:
АА
identityИв$lstm_cell_28/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask°
$lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_28_233300lstm_cell_28_233302lstm_cell_28_233304*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233299n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╗
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_28_233300lstm_cell_28_233302lstm_cell_28_233304*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_233313*
condR
while_cond_233312*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  Аl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  Аu
NoOpNoOp%^lstm_cell_28/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_28/StatefulPartitionedCall$lstm_cell_28/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╦#
х
while_body_233775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_29_233799_0:
АА*
while_lstm_cell_29_233801_0:	А.
while_lstm_cell_29_233803_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_29_233799:
АА(
while_lstm_cell_29_233801:	А,
while_lstm_cell_29_233803:	@АИв*while/lstm_cell_29/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0│
*while/lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_29_233799_0while_lstm_cell_29_233801_0while_lstm_cell_29_233803_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233761▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_29/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Р
while/Identity_4Identity3while/lstm_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @Р
while/Identity_5Identity3while/lstm_cell_29/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @y

while/NoOpNoOp+^while/lstm_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_29_233799while_lstm_cell_29_233799_0"8
while_lstm_cell_29_233801while_lstm_cell_29_233801_0"8
while_lstm_cell_29_233803while_lstm_cell_29_233803_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2X
*while/lstm_cell_29/StatefulPartitionedCall*while/lstm_cell_29/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╟	
ї
D__inference_dense_14_layer_call_and_return_conditional_losses_238620

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╟	
ї
D__inference_dense_14_layer_call_and_return_conditional_losses_234642

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ы}
ж	
while_body_234484
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_29_split_readvariableop_resource_0:
ААC
4while_lstm_cell_29_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_29_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_29_split_readvariableop_resource:
ААA
2while_lstm_cell_29_split_1_readvariableop_resource:	А=
*while_lstm_cell_29_readvariableop_resource:	@АИв!while/lstm_cell_29/ReadVariableOpв#while/lstm_cell_29/ReadVariableOp_1в#while/lstm_cell_29/ReadVariableOp_2в#while/lstm_cell_29/ReadVariableOp_3в'while/lstm_cell_29/split/ReadVariableOpв)while/lstm_cell_29/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_29/split/ReadVariableOpReadVariableOp2while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_29/split_1Split-while/lstm_cell_29/split_1/split_dim:output:01while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_29/BiasAddBiasAdd#while/lstm_cell_29/MatMul:product:0#while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_1BiasAdd%while/lstm_cell_29/MatMul_1:product:0#while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_2BiasAdd%while/lstm_cell_29/MatMul_2:product:0#while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_3BiasAdd%while/lstm_cell_29/MatMul_3:product:0#while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_29/ReadVariableOpReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_29/strided_sliceStridedSlice)while/lstm_cell_29/ReadVariableOp:value:0/while/lstm_cell_29/strided_slice/stack:output:01while/lstm_cell_29/strided_slice/stack_1:output:01while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_29/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_29/addAddV2#while/lstm_cell_29/BiasAdd:output:0%while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_29/MulMulwhile/lstm_cell_29/add:z:0!while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_29/Add_1AddV2while/lstm_cell_29/Mul:z:0#while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_29/clip_by_value/MinimumMinimumwhile/lstm_cell_29/Add_1:z:03while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_29/clip_by_valueMaximum,while/lstm_cell_29/clip_by_value/Minimum:z:0+while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_1ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_1StridedSlice+while/lstm_cell_29/ReadVariableOp_1:value:01while/lstm_cell_29/strided_slice_1/stack:output:03while/lstm_cell_29/strided_slice_1/stack_1:output:03while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_2AddV2%while/lstm_cell_29/BiasAdd_1:output:0%while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_1Mulwhile/lstm_cell_29/add_2:z:0#while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_3AddV2while/lstm_cell_29/Mul_1:z:0#while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_1/MinimumMinimumwhile/lstm_cell_29/Add_3:z:05while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_1Maximum.while/lstm_cell_29/clip_by_value_1/Minimum:z:0-while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_29/mul_2Mul&while/lstm_cell_29/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_2ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_2StridedSlice+while/lstm_cell_29/ReadVariableOp_2:value:01while/lstm_cell_29/strided_slice_2/stack:output:03while/lstm_cell_29/strided_slice_2/stack_1:output:03while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_4AddV2%while/lstm_cell_29/BiasAdd_2:output:0%while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_29/ReluReluwhile/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_29/mul_3Mul$while/lstm_cell_29/clip_by_value:z:0%while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_29/add_5AddV2while/lstm_cell_29/mul_2:z:0while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_3ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_3StridedSlice+while/lstm_cell_29/ReadVariableOp_3:value:01while/lstm_cell_29/strided_slice_3/stack:output:03while/lstm_cell_29/strided_slice_3/stack_1:output:03while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_6AddV2%while/lstm_cell_29/BiasAdd_3:output:0%while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_4Mulwhile/lstm_cell_29/add_6:z:0#while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_7AddV2while/lstm_cell_29/Mul_4:z:0#while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_2/MinimumMinimumwhile/lstm_cell_29/Add_7:z:05while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_2Maximum.while/lstm_cell_29/clip_by_value_2/Minimum:z:0-while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_29/mul_5Mul&while/lstm_cell_29/clip_by_value_2:z:0'while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_29/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_29/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_29/ReadVariableOp$^while/lstm_cell_29/ReadVariableOp_1$^while/lstm_cell_29/ReadVariableOp_2$^while/lstm_cell_29/ReadVariableOp_3(^while/lstm_cell_29/split/ReadVariableOp*^while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_29_readvariableop_resource,while_lstm_cell_29_readvariableop_resource_0"j
2while_lstm_cell_29_split_1_readvariableop_resource4while_lstm_cell_29_split_1_readvariableop_resource_0"f
0while_lstm_cell_29_split_readvariableop_resource2while_lstm_cell_29_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_29/ReadVariableOp_1#while/lstm_cell_29/ReadVariableOp_12J
#while/lstm_cell_29/ReadVariableOp_2#while/lstm_cell_29/ReadVariableOp_22J
#while/lstm_cell_29/ReadVariableOp_3#while/lstm_cell_29/ReadVariableOp_32F
!while/lstm_cell_29/ReadVariableOp!while/lstm_cell_29/ReadVariableOp2R
'while/lstm_cell_29/split/ReadVariableOp'while/lstm_cell_29/split/ReadVariableOp2V
)while/lstm_cell_29/split_1/ReadVariableOp)while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
м
╕
(__inference_lstm_28_layer_call_fn_236476
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:
АА
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_233381}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
иИ
ш
C__inference_lstm_29_layer_call_and_return_conditional_losses_238601

inputs>
*lstm_cell_29_split_readvariableop_resource:
АА;
,lstm_cell_29_split_1_readvariableop_resource:	А7
$lstm_cell_29_readvariableop_resource:	@А
identityИвlstm_cell_29/ReadVariableOpвlstm_cell_29/ReadVariableOp_1вlstm_cell_29/ReadVariableOp_2вlstm_cell_29/ReadVariableOp_3в!lstm_cell_29/split/ReadVariableOpв#lstm_cell_29/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask^
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_29/split/ReadVariableOpReadVariableOp*lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0)lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_29/split_1/ReadVariableOpReadVariableOp,lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_29/split_1Split'lstm_cell_29/split_1/split_dim:output:0+lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_29/BiasAddBiasAddlstm_cell_29/MatMul:product:0lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_1BiasAddlstm_cell_29/MatMul_1:product:0lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_2BiasAddlstm_cell_29/MatMul_2:product:0lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_3BiasAddlstm_cell_29/MatMul_3:product:0lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_29/ReadVariableOpReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_29/strided_sliceStridedSlice#lstm_cell_29/ReadVariableOp:value:0)lstm_cell_29/strided_slice/stack:output:0+lstm_cell_29/strided_slice/stack_1:output:0+lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_29/MatMul_4MatMulzeros:output:0#lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_29/addAddV2lstm_cell_29/BiasAdd:output:0lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_29/MulMullstm_cell_29/add:z:0lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_29/Add_1AddV2lstm_cell_29/Mul:z:0lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_29/clip_by_value/MinimumMinimumlstm_cell_29/Add_1:z:0-lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_29/clip_by_valueMaximum&lstm_cell_29/clip_by_value/Minimum:z:0%lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_1ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_1StridedSlice%lstm_cell_29/ReadVariableOp_1:value:0+lstm_cell_29/strided_slice_1/stack:output:0-lstm_cell_29/strided_slice_1/stack_1:output:0-lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_5MatMulzeros:output:0%lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_2AddV2lstm_cell_29/BiasAdd_1:output:0lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_1Mullstm_cell_29/add_2:z:0lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_3AddV2lstm_cell_29/Mul_1:z:0lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_cell_29/Add_3:z:0/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_1Maximum(lstm_cell_29/clip_by_value_1/Minimum:z:0'lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_29/mul_2Mul lstm_cell_29/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_2ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_2StridedSlice%lstm_cell_29/ReadVariableOp_2:value:0+lstm_cell_29/strided_slice_2/stack:output:0-lstm_cell_29/strided_slice_2/stack_1:output:0-lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_6MatMulzeros:output:0%lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_4AddV2lstm_cell_29/BiasAdd_2:output:0lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/ReluRelulstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_29/mul_3Mullstm_cell_29/clip_by_value:z:0lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_29/add_5AddV2lstm_cell_29/mul_2:z:0lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_3ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_3StridedSlice%lstm_cell_29/ReadVariableOp_3:value:0+lstm_cell_29/strided_slice_3/stack:output:0-lstm_cell_29/strided_slice_3/stack_1:output:0-lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_7MatMulzeros:output:0%lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_6AddV2lstm_cell_29/BiasAdd_3:output:0lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_4Mullstm_cell_29/add_6:z:0lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_7AddV2lstm_cell_29/Mul_4:z:0lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_cell_29/Add_7:z:0/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_2Maximum(lstm_cell_29/clip_by_value_2/Minimum:z:0'lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_29/Relu_1Relulstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_29/mul_5Mul lstm_cell_29/clip_by_value_2:z:0!lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_29_split_readvariableop_resource,lstm_cell_29_split_1_readvariableop_resource$lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_238461*
condR
while_cond_238460*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_29/ReadVariableOp^lstm_cell_29/ReadVariableOp_1^lstm_cell_29/ReadVariableOp_2^lstm_cell_29/ReadVariableOp_3"^lstm_cell_29/split/ReadVariableOp$^lstm_cell_29/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         А: : : 2>
lstm_cell_29/ReadVariableOp_1lstm_cell_29/ReadVariableOp_12>
lstm_cell_29/ReadVariableOp_2lstm_cell_29/ReadVariableOp_22>
lstm_cell_29/ReadVariableOp_3lstm_cell_29/ReadVariableOp_32:
lstm_cell_29/ReadVariableOplstm_cell_29/ReadVariableOp2F
!lstm_cell_29/split/ReadVariableOp!lstm_cell_29/split/ReadVariableOp2J
#lstm_cell_29/split_1/ReadVariableOp#lstm_cell_29/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
ы}
ж	
while_body_238461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_29_split_readvariableop_resource_0:
ААC
4while_lstm_cell_29_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_29_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_29_split_readvariableop_resource:
ААA
2while_lstm_cell_29_split_1_readvariableop_resource:	А=
*while_lstm_cell_29_readvariableop_resource:	@АИв!while/lstm_cell_29/ReadVariableOpв#while/lstm_cell_29/ReadVariableOp_1в#while/lstm_cell_29/ReadVariableOp_2в#while/lstm_cell_29/ReadVariableOp_3в'while/lstm_cell_29/split/ReadVariableOpв)while/lstm_cell_29/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_29/split/ReadVariableOpReadVariableOp2while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_29/split_1Split-while/lstm_cell_29/split_1/split_dim:output:01while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_29/BiasAddBiasAdd#while/lstm_cell_29/MatMul:product:0#while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_1BiasAdd%while/lstm_cell_29/MatMul_1:product:0#while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_2BiasAdd%while/lstm_cell_29/MatMul_2:product:0#while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_3BiasAdd%while/lstm_cell_29/MatMul_3:product:0#while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_29/ReadVariableOpReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_29/strided_sliceStridedSlice)while/lstm_cell_29/ReadVariableOp:value:0/while/lstm_cell_29/strided_slice/stack:output:01while/lstm_cell_29/strided_slice/stack_1:output:01while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_29/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_29/addAddV2#while/lstm_cell_29/BiasAdd:output:0%while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_29/MulMulwhile/lstm_cell_29/add:z:0!while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_29/Add_1AddV2while/lstm_cell_29/Mul:z:0#while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_29/clip_by_value/MinimumMinimumwhile/lstm_cell_29/Add_1:z:03while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_29/clip_by_valueMaximum,while/lstm_cell_29/clip_by_value/Minimum:z:0+while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_1ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_1StridedSlice+while/lstm_cell_29/ReadVariableOp_1:value:01while/lstm_cell_29/strided_slice_1/stack:output:03while/lstm_cell_29/strided_slice_1/stack_1:output:03while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_2AddV2%while/lstm_cell_29/BiasAdd_1:output:0%while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_1Mulwhile/lstm_cell_29/add_2:z:0#while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_3AddV2while/lstm_cell_29/Mul_1:z:0#while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_1/MinimumMinimumwhile/lstm_cell_29/Add_3:z:05while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_1Maximum.while/lstm_cell_29/clip_by_value_1/Minimum:z:0-while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_29/mul_2Mul&while/lstm_cell_29/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_2ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_2StridedSlice+while/lstm_cell_29/ReadVariableOp_2:value:01while/lstm_cell_29/strided_slice_2/stack:output:03while/lstm_cell_29/strided_slice_2/stack_1:output:03while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_4AddV2%while/lstm_cell_29/BiasAdd_2:output:0%while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_29/ReluReluwhile/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_29/mul_3Mul$while/lstm_cell_29/clip_by_value:z:0%while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_29/add_5AddV2while/lstm_cell_29/mul_2:z:0while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_3ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_3StridedSlice+while/lstm_cell_29/ReadVariableOp_3:value:01while/lstm_cell_29/strided_slice_3/stack:output:03while/lstm_cell_29/strided_slice_3/stack_1:output:03while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_6AddV2%while/lstm_cell_29/BiasAdd_3:output:0%while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_4Mulwhile/lstm_cell_29/add_6:z:0#while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_7AddV2while/lstm_cell_29/Mul_4:z:0#while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_2/MinimumMinimumwhile/lstm_cell_29/Add_7:z:05while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_2Maximum.while/lstm_cell_29/clip_by_value_2/Minimum:z:0-while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_29/mul_5Mul&while/lstm_cell_29/clip_by_value_2:z:0'while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_29/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_29/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_29/ReadVariableOp$^while/lstm_cell_29/ReadVariableOp_1$^while/lstm_cell_29/ReadVariableOp_2$^while/lstm_cell_29/ReadVariableOp_3(^while/lstm_cell_29/split/ReadVariableOp*^while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_29_readvariableop_resource,while_lstm_cell_29_readvariableop_resource_0"j
2while_lstm_cell_29_split_1_readvariableop_resource4while_lstm_cell_29_split_1_readvariableop_resource_0"f
0while_lstm_cell_29_split_readvariableop_resource2while_lstm_cell_29_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_29/ReadVariableOp_1#while/lstm_cell_29/ReadVariableOp_12J
#while/lstm_cell_29/ReadVariableOp_2#while/lstm_cell_29/ReadVariableOp_22J
#while/lstm_cell_29/ReadVariableOp_3#while/lstm_cell_29/ReadVariableOp_32F
!while/lstm_cell_29/ReadVariableOp!while/lstm_cell_29/ReadVariableOp2R
'while/lstm_cell_29/split/ReadVariableOp'while/lstm_cell_29/split/ReadVariableOp2V
)while/lstm_cell_29/split_1/ReadVariableOp)while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╘J
к
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233963

inputs

states
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:         @\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:         @\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:         @\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ы
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskd
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:         @[
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:         @\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:         @]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:         @e
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*'
_output_shapes
:         @V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:         @]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_5:z:0*
T0*'
_output_shapes
:         @i
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*'
_output_shapes
:         @X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:         @└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         @:         @: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
В
╢
(__inference_lstm_28_layer_call_fn_236509

inputs
unknown:	А
	unknown_0:	А
	unknown_1:
АА
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_235225t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ю
ў
-__inference_lstm_cell_29_layer_call_fn_238849

inputs
states_0
states_1
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identity

identity_1

identity_2ИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_233761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ы	
├
while_cond_236880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236880___redundant_placeholder04
0while_while_cond_236880___redundant_placeholder14
0while_while_cond_236880___redundant_placeholder24
0while_while_cond_236880___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ч	
├
while_cond_233774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_233774___redundant_placeholder04
0while_while_cond_233774___redundant_placeholder14
0while_while_cond_233774___redundant_placeholder24
0while_while_cond_233774___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
В
╢
(__inference_lstm_28_layer_call_fn_236498

inputs
unknown:	А
	unknown_0:	А
	unknown_1:
АА
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_234361t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ё	
╨
.__inference_sequential_14_layer_call_fn_234668
lstm_28_input
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:
АА
	unknown_3:	А
	unknown_4:	@А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCalllstm_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_14_layer_call_and_return_conditional_losses_234649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_28_input
ы}
ж	
while_body_238205
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_29_split_readvariableop_resource_0:
ААC
4while_lstm_cell_29_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_29_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_29_split_readvariableop_resource:
ААA
2while_lstm_cell_29_split_1_readvariableop_resource:	А=
*while_lstm_cell_29_readvariableop_resource:	@АИв!while/lstm_cell_29/ReadVariableOpв#while/lstm_cell_29/ReadVariableOp_1в#while/lstm_cell_29/ReadVariableOp_2в#while/lstm_cell_29/ReadVariableOp_3в'while/lstm_cell_29/split/ReadVariableOpв)while/lstm_cell_29/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_29/split/ReadVariableOpReadVariableOp2while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_29/split_1Split-while/lstm_cell_29/split_1/split_dim:output:01while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_29/BiasAddBiasAdd#while/lstm_cell_29/MatMul:product:0#while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_1BiasAdd%while/lstm_cell_29/MatMul_1:product:0#while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_2BiasAdd%while/lstm_cell_29/MatMul_2:product:0#while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_3BiasAdd%while/lstm_cell_29/MatMul_3:product:0#while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_29/ReadVariableOpReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_29/strided_sliceStridedSlice)while/lstm_cell_29/ReadVariableOp:value:0/while/lstm_cell_29/strided_slice/stack:output:01while/lstm_cell_29/strided_slice/stack_1:output:01while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_29/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_29/addAddV2#while/lstm_cell_29/BiasAdd:output:0%while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_29/MulMulwhile/lstm_cell_29/add:z:0!while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_29/Add_1AddV2while/lstm_cell_29/Mul:z:0#while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_29/clip_by_value/MinimumMinimumwhile/lstm_cell_29/Add_1:z:03while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_29/clip_by_valueMaximum,while/lstm_cell_29/clip_by_value/Minimum:z:0+while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_1ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_1StridedSlice+while/lstm_cell_29/ReadVariableOp_1:value:01while/lstm_cell_29/strided_slice_1/stack:output:03while/lstm_cell_29/strided_slice_1/stack_1:output:03while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_2AddV2%while/lstm_cell_29/BiasAdd_1:output:0%while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_1Mulwhile/lstm_cell_29/add_2:z:0#while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_3AddV2while/lstm_cell_29/Mul_1:z:0#while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_1/MinimumMinimumwhile/lstm_cell_29/Add_3:z:05while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_1Maximum.while/lstm_cell_29/clip_by_value_1/Minimum:z:0-while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_29/mul_2Mul&while/lstm_cell_29/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_2ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_2StridedSlice+while/lstm_cell_29/ReadVariableOp_2:value:01while/lstm_cell_29/strided_slice_2/stack:output:03while/lstm_cell_29/strided_slice_2/stack_1:output:03while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_4AddV2%while/lstm_cell_29/BiasAdd_2:output:0%while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_29/ReluReluwhile/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_29/mul_3Mul$while/lstm_cell_29/clip_by_value:z:0%while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_29/add_5AddV2while/lstm_cell_29/mul_2:z:0while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_3ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_3StridedSlice+while/lstm_cell_29/ReadVariableOp_3:value:01while/lstm_cell_29/strided_slice_3/stack:output:03while/lstm_cell_29/strided_slice_3/stack_1:output:03while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_6AddV2%while/lstm_cell_29/BiasAdd_3:output:0%while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_4Mulwhile/lstm_cell_29/add_6:z:0#while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_7AddV2while/lstm_cell_29/Mul_4:z:0#while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_2/MinimumMinimumwhile/lstm_cell_29/Add_7:z:05while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_2Maximum.while/lstm_cell_29/clip_by_value_2/Minimum:z:0-while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_29/mul_5Mul&while/lstm_cell_29/clip_by_value_2:z:0'while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_29/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_29/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_29/ReadVariableOp$^while/lstm_cell_29/ReadVariableOp_1$^while/lstm_cell_29/ReadVariableOp_2$^while/lstm_cell_29/ReadVariableOp_3(^while/lstm_cell_29/split/ReadVariableOp*^while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_29_readvariableop_resource,while_lstm_cell_29_readvariableop_resource_0"j
2while_lstm_cell_29_split_1_readvariableop_resource4while_lstm_cell_29_split_1_readvariableop_resource_0"f
0while_lstm_cell_29_split_readvariableop_resource2while_lstm_cell_29_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_29/ReadVariableOp_1#while/lstm_cell_29/ReadVariableOp_12J
#while/lstm_cell_29/ReadVariableOp_2#while/lstm_cell_29/ReadVariableOp_22J
#while/lstm_cell_29/ReadVariableOp_3#while/lstm_cell_29/ReadVariableOp_32F
!while/lstm_cell_29/ReadVariableOp!while/lstm_cell_29/ReadVariableOp2R
'while/lstm_cell_29/split/ReadVariableOp'while/lstm_cell_29/split/ReadVariableOp2V
)while/lstm_cell_29/split_1/ReadVariableOp)while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╙#
х
while_body_233313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_28_233337_0:	А*
while_lstm_cell_28_233339_0:	А/
while_lstm_cell_28_233341_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_28_233337:	А(
while_lstm_cell_28_233339:	А-
while_lstm_cell_28_233341:
ААИв*while/lstm_cell_28/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╢
*while/lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_28_233337_0while_lstm_cell_28_233339_0while_lstm_cell_28_233341_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         А:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_233299▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_28/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: С
while/Identity_4Identity3while/lstm_cell_28/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         АС
while/Identity_5Identity3while/lstm_cell_28/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Аy

while/NoOpNoOp+^while/lstm_cell_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_28_233337while_lstm_cell_28_233337_0"8
while_lstm_cell_28_233339while_lstm_cell_28_233339_0"8
while_lstm_cell_28_233341while_lstm_cell_28_233341_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2X
*while/lstm_cell_28/StatefulPartitionedCall*while/lstm_cell_28/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ё
▌
I__inference_sequential_14_layer_call_and_return_conditional_losses_235280

inputs!
lstm_28_235260:	А
lstm_28_235262:	А"
lstm_28_235264:
АА"
lstm_29_235267:
АА
lstm_29_235269:	А!
lstm_29_235271:	@А!
dense_14_235274:@
dense_14_235276:
identityИв dense_14/StatefulPartitionedCallвlstm_28/StatefulPartitionedCallвlstm_29/StatefulPartitionedCallГ
lstm_28/StatefulPartitionedCallStatefulPartitionedCallinputslstm_28_235260lstm_28_235262lstm_28_235264*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_235225а
lstm_29/StatefulPartitionedCallStatefulPartitionedCall(lstm_28/StatefulPartitionedCall:output:0lstm_29_235267lstm_29_235269lstm_29_235271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234947Т
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0dense_14_235274dense_14_235276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_234642x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_14/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
м
√
'sequential_14_lstm_29_while_cond_233028H
Dsequential_14_lstm_29_while_sequential_14_lstm_29_while_loop_counterN
Jsequential_14_lstm_29_while_sequential_14_lstm_29_while_maximum_iterations+
'sequential_14_lstm_29_while_placeholder-
)sequential_14_lstm_29_while_placeholder_1-
)sequential_14_lstm_29_while_placeholder_2-
)sequential_14_lstm_29_while_placeholder_3J
Fsequential_14_lstm_29_while_less_sequential_14_lstm_29_strided_slice_1`
\sequential_14_lstm_29_while_sequential_14_lstm_29_while_cond_233028___redundant_placeholder0`
\sequential_14_lstm_29_while_sequential_14_lstm_29_while_cond_233028___redundant_placeholder1`
\sequential_14_lstm_29_while_sequential_14_lstm_29_while_cond_233028___redundant_placeholder2`
\sequential_14_lstm_29_while_sequential_14_lstm_29_while_cond_233028___redundant_placeholder3(
$sequential_14_lstm_29_while_identity
║
 sequential_14/lstm_29/while/LessLess'sequential_14_lstm_29_while_placeholderFsequential_14_lstm_29_while_less_sequential_14_lstm_29_strided_slice_1*
T0*
_output_shapes
: w
$sequential_14/lstm_29/while/IdentityIdentity$sequential_14/lstm_29/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_14_lstm_29_while_identity-sequential_14/lstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_14/lstm_29/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_14/lstm_29/while/loop_counter
Ч	
├
while_cond_234806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234806___redundant_placeholder04
0while_while_cond_234806___redundant_placeholder14
0while_while_cond_234806___redundant_placeholder24
0while_while_cond_234806___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
И
у
lstm_29_while_cond_235804,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3.
*lstm_29_while_less_lstm_29_strided_slice_1D
@lstm_29_while_lstm_29_while_cond_235804___redundant_placeholder0D
@lstm_29_while_lstm_29_while_cond_235804___redundant_placeholder1D
@lstm_29_while_lstm_29_while_cond_235804___redundant_placeholder2D
@lstm_29_while_lstm_29_while_cond_235804___redundant_placeholder3
lstm_29_while_identity
В
lstm_29/while/LessLesslstm_29_while_placeholder*lstm_29_while_less_lstm_29_strided_slice_1*
T0*
_output_shapes
: [
lstm_29/while/IdentityIdentitylstm_29/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_29_while_identitylstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_29/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_29/while/loop_counter
Ы	
├
while_cond_234220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_234220___redundant_placeholder04
0while_while_cond_234220___redundant_placeholder14
0while_while_cond_234220___redundant_placeholder24
0while_while_cond_234220___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▌И
ъ
C__inference_lstm_29_layer_call_and_return_conditional_losses_237833
inputs_0>
*lstm_cell_29_split_readvariableop_resource:
АА;
,lstm_cell_29_split_1_readvariableop_resource:	А7
$lstm_cell_29_readvariableop_resource:	@А
identityИвlstm_cell_29/ReadVariableOpвlstm_cell_29/ReadVariableOp_1вlstm_cell_29/ReadVariableOp_2вlstm_cell_29/ReadVariableOp_3в!lstm_cell_29/split/ReadVariableOpв#lstm_cell_29/split_1/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  АR
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask^
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_29/split/ReadVariableOpReadVariableOp*lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0)lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_29/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_29/split_1/ReadVariableOpReadVariableOp,lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_29/split_1Split'lstm_cell_29/split_1/split_dim:output:0+lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_29/BiasAddBiasAddlstm_cell_29/MatMul:product:0lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_1BiasAddlstm_cell_29/MatMul_1:product:0lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_2BiasAddlstm_cell_29/MatMul_2:product:0lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_29/BiasAdd_3BiasAddlstm_cell_29/MatMul_3:product:0lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_29/ReadVariableOpReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_29/strided_sliceStridedSlice#lstm_cell_29/ReadVariableOp:value:0)lstm_cell_29/strided_slice/stack:output:0+lstm_cell_29/strided_slice/stack_1:output:0+lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_29/MatMul_4MatMulzeros:output:0#lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_29/addAddV2lstm_cell_29/BiasAdd:output:0lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_29/MulMullstm_cell_29/add:z:0lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_29/Add_1AddV2lstm_cell_29/Mul:z:0lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_29/clip_by_value/MinimumMinimumlstm_cell_29/Add_1:z:0-lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_29/clip_by_valueMaximum&lstm_cell_29/clip_by_value/Minimum:z:0%lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_1ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_1StridedSlice%lstm_cell_29/ReadVariableOp_1:value:0+lstm_cell_29/strided_slice_1/stack:output:0-lstm_cell_29/strided_slice_1/stack_1:output:0-lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_5MatMulzeros:output:0%lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_2AddV2lstm_cell_29/BiasAdd_1:output:0lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_1Mullstm_cell_29/add_2:z:0lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_3AddV2lstm_cell_29/Mul_1:z:0lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_cell_29/Add_3:z:0/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_1Maximum(lstm_cell_29/clip_by_value_1/Minimum:z:0'lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_29/mul_2Mul lstm_cell_29/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_2ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_2StridedSlice%lstm_cell_29/ReadVariableOp_2:value:0+lstm_cell_29/strided_slice_2/stack:output:0-lstm_cell_29/strided_slice_2/stack_1:output:0-lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_6MatMulzeros:output:0%lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_4AddV2lstm_cell_29/BiasAdd_2:output:0lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/ReluRelulstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_29/mul_3Mullstm_cell_29/clip_by_value:z:0lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_29/add_5AddV2lstm_cell_29/mul_2:z:0lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_29/ReadVariableOp_3ReadVariableOp$lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_29/strided_slice_3StridedSlice%lstm_cell_29/ReadVariableOp_3:value:0+lstm_cell_29/strided_slice_3/stack:output:0-lstm_cell_29/strided_slice_3/stack_1:output:0-lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_29/MatMul_7MatMulzeros:output:0%lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_29/add_6AddV2lstm_cell_29/BiasAdd_3:output:0lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_29/Mul_4Mullstm_cell_29/add_6:z:0lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_29/Add_7AddV2lstm_cell_29/Mul_4:z:0lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_cell_29/Add_7:z:0/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_29/clip_by_value_2Maximum(lstm_cell_29/clip_by_value_2/Minimum:z:0'lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_29/Relu_1Relulstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_29/mul_5Mul lstm_cell_29/clip_by_value_2:z:0!lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_29_split_readvariableop_resource,lstm_cell_29_split_1_readvariableop_resource$lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_237693*
condR
while_cond_237692*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_29/ReadVariableOp^lstm_cell_29/ReadVariableOp_1^lstm_cell_29/ReadVariableOp_2^lstm_cell_29/ReadVariableOp_3"^lstm_cell_29/split/ReadVariableOp$^lstm_cell_29/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2>
lstm_cell_29/ReadVariableOp_1lstm_cell_29/ReadVariableOp_12>
lstm_cell_29/ReadVariableOp_2lstm_cell_29/ReadVariableOp_22>
lstm_cell_29/ReadVariableOp_3lstm_cell_29/ReadVariableOp_32:
lstm_cell_29/ReadVariableOplstm_cell_29/ReadVariableOp2F
!lstm_cell_29/split/ReadVariableOp!lstm_cell_29/split/ReadVariableOp2J
#lstm_cell_29/split_1/ReadVariableOp#lstm_cell_29/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  А
"
_user_specified_name
inputs_0
б~
ж	
while_body_234221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_28_split_readvariableop_resource_0:	АC
4while_lstm_cell_28_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_28_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_28_split_readvariableop_resource:	АA
2while_lstm_cell_28_split_1_readvariableop_resource:	А>
*while_lstm_cell_28_readvariableop_resource:
ААИв!while/lstm_cell_28/ReadVariableOpв#while/lstm_cell_28/ReadVariableOp_1в#while/lstm_cell_28/ReadVariableOp_2в#while/lstm_cell_28/ReadVariableOp_3в'while/lstm_cell_28/split/ReadVariableOpв)while/lstm_cell_28/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_28/split/ReadVariableOpReadVariableOp2while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_28/split_1Split-while/lstm_cell_28/split_1/split_dim:output:01while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_28/BiasAddBiasAdd#while/lstm_cell_28/MatMul:product:0#while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_1BiasAdd%while/lstm_cell_28/MatMul_1:product:0#while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_2BiasAdd%while/lstm_cell_28/MatMul_2:product:0#while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_3BiasAdd%while/lstm_cell_28/MatMul_3:product:0#while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_28/ReadVariableOpReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_28/strided_sliceStridedSlice)while/lstm_cell_28/ReadVariableOp:value:0/while/lstm_cell_28/strided_slice/stack:output:01while/lstm_cell_28/strided_slice/stack_1:output:01while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_28/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_28/addAddV2#while/lstm_cell_28/BiasAdd:output:0%while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_28/MulMulwhile/lstm_cell_28/add:z:0!while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_28/Add_1AddV2while/lstm_cell_28/Mul:z:0#while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_28/clip_by_value/MinimumMinimumwhile/lstm_cell_28/Add_1:z:03while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_28/clip_by_valueMaximum,while/lstm_cell_28/clip_by_value/Minimum:z:0+while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_1ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_1StridedSlice+while/lstm_cell_28/ReadVariableOp_1:value:01while/lstm_cell_28/strided_slice_1/stack:output:03while/lstm_cell_28/strided_slice_1/stack_1:output:03while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_2AddV2%while/lstm_cell_28/BiasAdd_1:output:0%while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_1Mulwhile/lstm_cell_28/add_2:z:0#while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_3AddV2while/lstm_cell_28/Mul_1:z:0#while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_1/MinimumMinimumwhile/lstm_cell_28/Add_3:z:05while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_1Maximum.while/lstm_cell_28/clip_by_value_1/Minimum:z:0-while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_28/mul_2Mul&while/lstm_cell_28/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_2ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_2StridedSlice+while/lstm_cell_28/ReadVariableOp_2:value:01while/lstm_cell_28/strided_slice_2/stack:output:03while/lstm_cell_28/strided_slice_2/stack_1:output:03while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_4AddV2%while/lstm_cell_28/BiasAdd_2:output:0%while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_28/ReluReluwhile/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_28/mul_3Mul$while/lstm_cell_28/clip_by_value:z:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_28/add_5AddV2while/lstm_cell_28/mul_2:z:0while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_3ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_3StridedSlice+while/lstm_cell_28/ReadVariableOp_3:value:01while/lstm_cell_28/strided_slice_3/stack:output:03while/lstm_cell_28/strided_slice_3/stack_1:output:03while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_6AddV2%while/lstm_cell_28/BiasAdd_3:output:0%while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_4Mulwhile/lstm_cell_28/add_6:z:0#while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_7AddV2while/lstm_cell_28/Mul_4:z:0#while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_2/MinimumMinimumwhile/lstm_cell_28/Add_7:z:05while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_2Maximum.while/lstm_cell_28/clip_by_value_2/Minimum:z:0-while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_28/mul_5Mul&while/lstm_cell_28/clip_by_value_2:z:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_28/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_28/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_28/ReadVariableOp$^while/lstm_cell_28/ReadVariableOp_1$^while/lstm_cell_28/ReadVariableOp_2$^while/lstm_cell_28/ReadVariableOp_3(^while/lstm_cell_28/split/ReadVariableOp*^while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_28_readvariableop_resource,while_lstm_cell_28_readvariableop_resource_0"j
2while_lstm_cell_28_split_1_readvariableop_resource4while_lstm_cell_28_split_1_readvariableop_resource_0"f
0while_lstm_cell_28_split_readvariableop_resource2while_lstm_cell_28_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_28/ReadVariableOp_1#while/lstm_cell_28/ReadVariableOp_12J
#while/lstm_cell_28/ReadVariableOp_2#while/lstm_cell_28/ReadVariableOp_22J
#while/lstm_cell_28/ReadVariableOp_3#while/lstm_cell_28/ReadVariableOp_32F
!while/lstm_cell_28/ReadVariableOp!while/lstm_cell_28/ReadVariableOp2R
'while/lstm_cell_28/split/ReadVariableOp'while/lstm_cell_28/split/ReadVariableOp2V
)while/lstm_cell_28/split_1/ReadVariableOp)while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
▌И
ш
C__inference_lstm_28_layer_call_and_return_conditional_losses_235225

inputs=
*lstm_cell_28_split_readvariableop_resource:	А;
,lstm_cell_28_split_1_readvariableop_resource:	А8
$lstm_cell_28_readvariableop_resource:
АА
identityИвlstm_cell_28/ReadVariableOpвlstm_cell_28/ReadVariableOp_1вlstm_cell_28/ReadVariableOp_2вlstm_cell_28/ReadVariableOp_3в!lstm_cell_28/split/ReadVariableOpв#lstm_cell_28/split_1/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         АS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask^
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_28/split/ReadVariableOpReadVariableOp*lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0)lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_28/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_28/split_1/ReadVariableOpReadVariableOp,lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_28/split_1Split'lstm_cell_28/split_1/split_dim:output:0+lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_28/BiasAddBiasAddlstm_cell_28/MatMul:product:0lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_1BiasAddlstm_cell_28/MatMul_1:product:0lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_2BiasAddlstm_cell_28/MatMul_2:product:0lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_28/BiasAdd_3BiasAddlstm_cell_28/MatMul_3:product:0lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_28/ReadVariableOpReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_28/strided_sliceStridedSlice#lstm_cell_28/ReadVariableOp:value:0)lstm_cell_28/strided_slice/stack:output:0+lstm_cell_28/strided_slice/stack_1:output:0+lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_28/MatMul_4MatMulzeros:output:0#lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_28/addAddV2lstm_cell_28/BiasAdd:output:0lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_28/MulMullstm_cell_28/add:z:0lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_28/Add_1AddV2lstm_cell_28/Mul:z:0lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_28/clip_by_value/MinimumMinimumlstm_cell_28/Add_1:z:0-lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_28/clip_by_valueMaximum&lstm_cell_28/clip_by_value/Minimum:z:0%lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_1ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_1StridedSlice%lstm_cell_28/ReadVariableOp_1:value:0+lstm_cell_28/strided_slice_1/stack:output:0-lstm_cell_28/strided_slice_1/stack_1:output:0-lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_5MatMulzeros:output:0%lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_2AddV2lstm_cell_28/BiasAdd_1:output:0lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_1Mullstm_cell_28/add_2:z:0lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_3AddV2lstm_cell_28/Mul_1:z:0lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_cell_28/Add_3:z:0/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_1Maximum(lstm_cell_28/clip_by_value_1/Minimum:z:0'lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_28/mul_2Mul lstm_cell_28/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_2ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_2StridedSlice%lstm_cell_28/ReadVariableOp_2:value:0+lstm_cell_28/strided_slice_2/stack:output:0-lstm_cell_28/strided_slice_2/stack_1:output:0-lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_6MatMulzeros:output:0%lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_4AddV2lstm_cell_28/BiasAdd_2:output:0lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_28/ReluRelulstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_28/mul_3Mullstm_cell_28/clip_by_value:z:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_28/add_5AddV2lstm_cell_28/mul_2:z:0lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_28/ReadVariableOp_3ReadVariableOp$lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_28/strided_slice_3StridedSlice%lstm_cell_28/ReadVariableOp_3:value:0+lstm_cell_28/strided_slice_3/stack:output:0-lstm_cell_28/strided_slice_3/stack_1:output:0-lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_28/MatMul_7MatMulzeros:output:0%lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_28/add_6AddV2lstm_cell_28/BiasAdd_3:output:0lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_28/Mul_4Mullstm_cell_28/add_6:z:0lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_28/Add_7AddV2lstm_cell_28/Mul_4:z:0lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_cell_28/Add_7:z:0/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_28/clip_by_value_2Maximum(lstm_cell_28/clip_by_value_2/Minimum:z:0'lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_28/Relu_1Relulstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_28/mul_5Mul lstm_cell_28/clip_by_value_2:z:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : №
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_28_split_readvariableop_resource,lstm_cell_28_split_1_readvariableop_resource$lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_235085*
condR
while_cond_235084*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         АЦ
NoOpNoOp^lstm_cell_28/ReadVariableOp^lstm_cell_28/ReadVariableOp_1^lstm_cell_28/ReadVariableOp_2^lstm_cell_28/ReadVariableOp_3"^lstm_cell_28/split/ReadVariableOp$^lstm_cell_28/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2>
lstm_cell_28/ReadVariableOp_1lstm_cell_28/ReadVariableOp_12>
lstm_cell_28/ReadVariableOp_2lstm_cell_28/ReadVariableOp_22>
lstm_cell_28/ReadVariableOp_3lstm_cell_28/ReadVariableOp_32:
lstm_cell_28/ReadVariableOplstm_cell_28/ReadVariableOp2F
!lstm_cell_28/split/ReadVariableOp!lstm_cell_28/split/ReadVariableOp2J
#lstm_cell_28/split_1/ReadVariableOp#lstm_cell_28/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
█П
╛
lstm_29_while_body_235805,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3+
'lstm_29_while_lstm_29_strided_slice_1_0g
clstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0N
:lstm_29_while_lstm_cell_29_split_readvariableop_resource_0:
ААK
<lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0:	АG
4lstm_29_while_lstm_cell_29_readvariableop_resource_0:	@А
lstm_29_while_identity
lstm_29_while_identity_1
lstm_29_while_identity_2
lstm_29_while_identity_3
lstm_29_while_identity_4
lstm_29_while_identity_5)
%lstm_29_while_lstm_29_strided_slice_1e
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorL
8lstm_29_while_lstm_cell_29_split_readvariableop_resource:
ААI
:lstm_29_while_lstm_cell_29_split_1_readvariableop_resource:	АE
2lstm_29_while_lstm_cell_29_readvariableop_resource:	@АИв)lstm_29/while/lstm_cell_29/ReadVariableOpв+lstm_29/while/lstm_cell_29/ReadVariableOp_1в+lstm_29/while/lstm_cell_29/ReadVariableOp_2в+lstm_29/while/lstm_cell_29/ReadVariableOp_3в/lstm_29/while/lstm_cell_29/split/ReadVariableOpв1lstm_29/while/lstm_cell_29/split_1/ReadVariableOpР
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╧
1lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0lstm_29_while_placeholderHlstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0l
*lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :м
/lstm_29/while/lstm_cell_29/split/ReadVariableOpReadVariableOp:lstm_29_while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0є
 lstm_29/while/lstm_cell_29/splitSplit3lstm_29/while/lstm_cell_29/split/split_dim:output:07lstm_29/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split┬
!lstm_29/while/lstm_cell_29/MatMulMatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @─
#lstm_29/while/lstm_cell_29/MatMul_1MatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @─
#lstm_29/while/lstm_cell_29/MatMul_2MatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @─
#lstm_29/while/lstm_cell_29/MatMul_3MatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_29/while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @n
,lstm_29/while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_29/while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp<lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0х
"lstm_29/while/lstm_cell_29/split_1Split5lstm_29/while/lstm_cell_29/split_1/split_dim:output:09lstm_29/while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split╣
"lstm_29/while/lstm_cell_29/BiasAddBiasAdd+lstm_29/while/lstm_cell_29/MatMul:product:0+lstm_29/while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @╜
$lstm_29/while/lstm_cell_29/BiasAdd_1BiasAdd-lstm_29/while/lstm_cell_29/MatMul_1:product:0+lstm_29/while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @╜
$lstm_29/while/lstm_cell_29/BiasAdd_2BiasAdd-lstm_29/while/lstm_cell_29/MatMul_2:product:0+lstm_29/while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @╜
$lstm_29/while/lstm_cell_29/BiasAdd_3BiasAdd-lstm_29/while/lstm_cell_29/MatMul_3:product:0+lstm_29/while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @Я
)lstm_29/while/lstm_cell_29/ReadVariableOpReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0
.lstm_29/while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_29/while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Б
0lstm_29/while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Є
(lstm_29/while/lstm_cell_29/strided_sliceStridedSlice1lstm_29/while/lstm_cell_29/ReadVariableOp:value:07lstm_29/while/lstm_cell_29/strided_slice/stack:output:09lstm_29/while/lstm_cell_29/strided_slice/stack_1:output:09lstm_29/while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskп
#lstm_29/while/lstm_cell_29/MatMul_4MatMullstm_29_while_placeholder_21lstm_29/while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @╡
lstm_29/while/lstm_cell_29/addAddV2+lstm_29/while/lstm_cell_29/BiasAdd:output:0-lstm_29/while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @e
 lstm_29/while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_29/while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?ж
lstm_29/while/lstm_cell_29/MulMul"lstm_29/while/lstm_cell_29/add:z:0)lstm_29/while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @м
 lstm_29/while/lstm_cell_29/Add_1AddV2"lstm_29/while/lstm_cell_29/Mul:z:0+lstm_29/while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @w
2lstm_29/while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╨
0lstm_29/while/lstm_cell_29/clip_by_value/MinimumMinimum$lstm_29/while/lstm_cell_29/Add_1:z:0;lstm_29/while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @o
*lstm_29/while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╨
(lstm_29/while/lstm_cell_29/clip_by_valueMaximum4lstm_29/while/lstm_cell_29/clip_by_value/Minimum:z:03lstm_29/while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @б
+lstm_29/while/lstm_cell_29/ReadVariableOp_1ReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_29/while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   Г
2lstm_29/while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_29/while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_29/while/lstm_cell_29/strided_slice_1StridedSlice3lstm_29/while/lstm_cell_29/ReadVariableOp_1:value:09lstm_29/while/lstm_cell_29/strided_slice_1/stack:output:0;lstm_29/while/lstm_cell_29/strided_slice_1/stack_1:output:0;lstm_29/while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_29/while/lstm_cell_29/MatMul_5MatMullstm_29_while_placeholder_23lstm_29/while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @╣
 lstm_29/while/lstm_cell_29/add_2AddV2-lstm_29/while/lstm_cell_29/BiasAdd_1:output:0-lstm_29/while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @g
"lstm_29/while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_29/while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_29/while/lstm_cell_29/Mul_1Mul$lstm_29/while/lstm_cell_29/add_2:z:0+lstm_29/while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @о
 lstm_29/while/lstm_cell_29/Add_3AddV2$lstm_29/while/lstm_cell_29/Mul_1:z:0+lstm_29/while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @y
4lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_29/while/lstm_cell_29/clip_by_value_1/MinimumMinimum$lstm_29/while/lstm_cell_29/Add_3:z:0=lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_29/while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_29/while/lstm_cell_29/clip_by_value_1Maximum6lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum:z:05lstm_29/while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @ж
 lstm_29/while/lstm_cell_29/mul_2Mul.lstm_29/while/lstm_cell_29/clip_by_value_1:z:0lstm_29_while_placeholder_3*
T0*'
_output_shapes
:         @б
+lstm_29/while/lstm_cell_29/ReadVariableOp_2ReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_29/while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_29/while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_29/while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_29/while/lstm_cell_29/strided_slice_2StridedSlice3lstm_29/while/lstm_cell_29/ReadVariableOp_2:value:09lstm_29/while/lstm_cell_29/strided_slice_2/stack:output:0;lstm_29/while/lstm_cell_29/strided_slice_2/stack_1:output:0;lstm_29/while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_29/while/lstm_cell_29/MatMul_6MatMullstm_29_while_placeholder_23lstm_29/while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @╣
 lstm_29/while/lstm_cell_29/add_4AddV2-lstm_29/while/lstm_cell_29/BiasAdd_2:output:0-lstm_29/while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @
lstm_29/while/lstm_cell_29/ReluRelu$lstm_29/while/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @╢
 lstm_29/while/lstm_cell_29/mul_3Mul,lstm_29/while/lstm_cell_29/clip_by_value:z:0-lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @з
 lstm_29/while/lstm_cell_29/add_5AddV2$lstm_29/while/lstm_cell_29/mul_2:z:0$lstm_29/while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @б
+lstm_29/while/lstm_cell_29/ReadVariableOp_3ReadVariableOp4lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_29/while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_29/while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_29/while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_29/while/lstm_cell_29/strided_slice_3StridedSlice3lstm_29/while/lstm_cell_29/ReadVariableOp_3:value:09lstm_29/while/lstm_cell_29/strided_slice_3/stack:output:0;lstm_29/while/lstm_cell_29/strided_slice_3/stack_1:output:0;lstm_29/while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_29/while/lstm_cell_29/MatMul_7MatMullstm_29_while_placeholder_23lstm_29/while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @╣
 lstm_29/while/lstm_cell_29/add_6AddV2-lstm_29/while/lstm_cell_29/BiasAdd_3:output:0-lstm_29/while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @g
"lstm_29/while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_29/while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_29/while/lstm_cell_29/Mul_4Mul$lstm_29/while/lstm_cell_29/add_6:z:0+lstm_29/while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @о
 lstm_29/while/lstm_cell_29/Add_7AddV2$lstm_29/while/lstm_cell_29/Mul_4:z:0+lstm_29/while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @y
4lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_29/while/lstm_cell_29/clip_by_value_2/MinimumMinimum$lstm_29/while/lstm_cell_29/Add_7:z:0=lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_29/while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_29/while/lstm_cell_29/clip_by_value_2Maximum6lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum:z:05lstm_29/while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @Б
!lstm_29/while/lstm_cell_29/Relu_1Relu$lstm_29/while/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @║
 lstm_29/while/lstm_cell_29/mul_5Mul.lstm_29/while/lstm_cell_29/clip_by_value_2:z:0/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @х
2lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_29_while_placeholder_1lstm_29_while_placeholder$lstm_29/while/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_29/while/addAddV2lstm_29_while_placeholderlstm_29/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_29/while/add_1AddV2(lstm_29_while_lstm_29_while_loop_counterlstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_29/while/IdentityIdentitylstm_29/while/add_1:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: К
lstm_29/while/Identity_1Identity.lstm_29_while_lstm_29_while_maximum_iterations^lstm_29/while/NoOp*
T0*
_output_shapes
: q
lstm_29/while/Identity_2Identitylstm_29/while/add:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: Ю
lstm_29/while/Identity_3IdentityBlstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_29/while/NoOp*
T0*
_output_shapes
: С
lstm_29/while/Identity_4Identity$lstm_29/while/lstm_cell_29/mul_5:z:0^lstm_29/while/NoOp*
T0*'
_output_shapes
:         @С
lstm_29/while/Identity_5Identity$lstm_29/while/lstm_cell_29/add_5:z:0^lstm_29/while/NoOp*
T0*'
_output_shapes
:         @Ё
lstm_29/while/NoOpNoOp*^lstm_29/while/lstm_cell_29/ReadVariableOp,^lstm_29/while/lstm_cell_29/ReadVariableOp_1,^lstm_29/while/lstm_cell_29/ReadVariableOp_2,^lstm_29/while/lstm_cell_29/ReadVariableOp_30^lstm_29/while/lstm_cell_29/split/ReadVariableOp2^lstm_29/while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_29_while_identity_1!lstm_29/while/Identity_1:output:0"=
lstm_29_while_identity_2!lstm_29/while/Identity_2:output:0"=
lstm_29_while_identity_3!lstm_29/while/Identity_3:output:0"=
lstm_29_while_identity_4!lstm_29/while/Identity_4:output:0"=
lstm_29_while_identity_5!lstm_29/while/Identity_5:output:0"9
lstm_29_while_identitylstm_29/while/Identity:output:0"P
%lstm_29_while_lstm_29_strided_slice_1'lstm_29_while_lstm_29_strided_slice_1_0"j
2lstm_29_while_lstm_cell_29_readvariableop_resource4lstm_29_while_lstm_cell_29_readvariableop_resource_0"z
:lstm_29_while_lstm_cell_29_split_1_readvariableop_resource<lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0"v
8lstm_29_while_lstm_cell_29_split_readvariableop_resource:lstm_29_while_lstm_cell_29_split_readvariableop_resource_0"╚
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2Z
+lstm_29/while/lstm_cell_29/ReadVariableOp_1+lstm_29/while/lstm_cell_29/ReadVariableOp_12Z
+lstm_29/while/lstm_cell_29/ReadVariableOp_2+lstm_29/while/lstm_cell_29/ReadVariableOp_22Z
+lstm_29/while/lstm_cell_29/ReadVariableOp_3+lstm_29/while/lstm_cell_29/ReadVariableOp_32V
)lstm_29/while/lstm_cell_29/ReadVariableOp)lstm_29/while/lstm_cell_29/ReadVariableOp2b
/lstm_29/while/lstm_cell_29/split/ReadVariableOp/lstm_29/while/lstm_cell_29/split/ReadVariableOp2f
1lstm_29/while/lstm_cell_29/split_1/ReadVariableOp1lstm_29/while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_29/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_29/while/loop_counter
Е
ф
I__inference_sequential_14_layer_call_and_return_conditional_losses_235366
lstm_28_input!
lstm_28_235346:	А
lstm_28_235348:	А"
lstm_28_235350:
АА"
lstm_29_235353:
АА
lstm_29_235355:	А!
lstm_29_235357:	@А!
dense_14_235360:@
dense_14_235362:
identityИв dense_14/StatefulPartitionedCallвlstm_28/StatefulPartitionedCallвlstm_29/StatefulPartitionedCallК
lstm_28/StatefulPartitionedCallStatefulPartitionedCalllstm_28_inputlstm_28_235346lstm_28_235348lstm_28_235350*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_235225а
lstm_29/StatefulPartitionedCallStatefulPartitionedCall(lstm_28/StatefulPartitionedCall:output:0lstm_29_235353lstm_29_235355lstm_29_235357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234947Т
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0dense_14_235360dense_14_235362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_234642x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_14/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_28_input
№╕
╩	
I__inference_sequential_14_layer_call_and_return_conditional_losses_236465

inputsE
2lstm_28_lstm_cell_28_split_readvariableop_resource:	АC
4lstm_28_lstm_cell_28_split_1_readvariableop_resource:	А@
,lstm_28_lstm_cell_28_readvariableop_resource:
ААF
2lstm_29_lstm_cell_29_split_readvariableop_resource:
ААC
4lstm_29_lstm_cell_29_split_1_readvariableop_resource:	А?
,lstm_29_lstm_cell_29_readvariableop_resource:	@А9
'dense_14_matmul_readvariableop_resource:@6
(dense_14_biasadd_readvariableop_resource:
identityИвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpв#lstm_28/lstm_cell_28/ReadVariableOpв%lstm_28/lstm_cell_28/ReadVariableOp_1в%lstm_28/lstm_cell_28/ReadVariableOp_2в%lstm_28/lstm_cell_28/ReadVariableOp_3в)lstm_28/lstm_cell_28/split/ReadVariableOpв+lstm_28/lstm_cell_28/split_1/ReadVariableOpвlstm_28/whileв#lstm_29/lstm_cell_29/ReadVariableOpв%lstm_29/lstm_cell_29/ReadVariableOp_1в%lstm_29/lstm_cell_29/ReadVariableOp_2в%lstm_29/lstm_cell_29/ReadVariableOp_3в)lstm_29/lstm_cell_29/split/ReadVariableOpв+lstm_29/lstm_cell_29/split_1/ReadVariableOpвlstm_29/whileQ
lstm_28/ShapeShapeinputs*
T0*
_output_shapes
::э╧e
lstm_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_28/strided_sliceStridedSlicelstm_28/Shape:output:0$lstm_28/strided_slice/stack:output:0&lstm_28/strided_slice/stack_1:output:0&lstm_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АЛ
lstm_28/zeros/packedPacklstm_28/strided_slice:output:0lstm_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
lstm_28/zerosFilllstm_28/zeros/packed:output:0lstm_28/zeros/Const:output:0*
T0*(
_output_shapes
:         А[
lstm_28/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АП
lstm_28/zeros_1/packedPacklstm_28/strided_slice:output:0!lstm_28/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_28/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
lstm_28/zeros_1Filllstm_28/zeros_1/packed:output:0lstm_28/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аk
lstm_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_28/transpose	Transposeinputslstm_28/transpose/perm:output:0*
T0*+
_output_shapes
:         b
lstm_28/Shape_1Shapelstm_28/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_28/strided_slice_1StridedSlicelstm_28/Shape_1:output:0&lstm_28/strided_slice_1/stack:output:0(lstm_28/strided_slice_1/stack_1:output:0(lstm_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_28/TensorArrayV2TensorListReserve,lstm_28/TensorArrayV2/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       °
/lstm_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_28/transpose:y:0Flstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
lstm_28/strided_slice_2StridedSlicelstm_28/transpose:y:0&lstm_28/strided_slice_2/stack:output:0(lstm_28/strided_slice_2/stack_1:output:0(lstm_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskf
$lstm_28/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Э
)lstm_28/lstm_cell_28/split/ReadVariableOpReadVariableOp2lstm_28_lstm_cell_28_split_readvariableop_resource*
_output_shapes
:	А*
dtype0с
lstm_28/lstm_cell_28/splitSplit-lstm_28/lstm_cell_28/split/split_dim:output:01lstm_28/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЯ
lstm_28/lstm_cell_28/MatMulMatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Аб
lstm_28/lstm_cell_28/MatMul_1MatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Аб
lstm_28/lstm_cell_28/MatMul_2MatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Аб
lstm_28/lstm_cell_28/MatMul_3MatMul lstm_28/strided_slice_2:output:0#lstm_28/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аh
&lstm_28/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_28/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4lstm_28_lstm_cell_28_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╫
lstm_28/lstm_cell_28/split_1Split/lstm_28/lstm_cell_28/split_1/split_dim:output:03lstm_28/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitи
lstm_28/lstm_cell_28/BiasAddBiasAdd%lstm_28/lstm_cell_28/MatMul:product:0%lstm_28/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Ам
lstm_28/lstm_cell_28/BiasAdd_1BiasAdd'lstm_28/lstm_cell_28/MatMul_1:product:0%lstm_28/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Ам
lstm_28/lstm_cell_28/BiasAdd_2BiasAdd'lstm_28/lstm_cell_28/MatMul_2:product:0%lstm_28/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Ам
lstm_28/lstm_cell_28/BiasAdd_3BiasAdd'lstm_28/lstm_cell_28/MatMul_3:product:0%lstm_28/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АТ
#lstm_28/lstm_cell_28/ReadVariableOpReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0y
(lstm_28/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_28/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*lstm_28/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"lstm_28/lstm_cell_28/strided_sliceStridedSlice+lstm_28/lstm_cell_28/ReadVariableOp:value:01lstm_28/lstm_cell_28/strided_slice/stack:output:03lstm_28/lstm_cell_28/strided_slice/stack_1:output:03lstm_28/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЯ
lstm_28/lstm_cell_28/MatMul_4MatMullstm_28/zeros:output:0+lstm_28/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         Ад
lstm_28/lstm_cell_28/addAddV2%lstm_28/lstm_cell_28/BiasAdd:output:0'lstm_28/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А_
lstm_28/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_28/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
lstm_28/lstm_cell_28/MulMullstm_28/lstm_cell_28/add:z:0#lstm_28/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АЫ
lstm_28/lstm_cell_28/Add_1AddV2lstm_28/lstm_cell_28/Mul:z:0%lstm_28/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аq
,lstm_28/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┐
*lstm_28/lstm_cell_28/clip_by_value/MinimumMinimumlstm_28/lstm_cell_28/Add_1:z:05lstm_28/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$lstm_28/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"lstm_28/lstm_cell_28/clip_by_valueMaximum.lstm_28/lstm_cell_28/clip_by_value/Minimum:z:0-lstm_28/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АФ
%lstm_28/lstm_cell_28/ReadVariableOp_1ReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_28/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_28/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_28/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_28/lstm_cell_28/strided_slice_1StridedSlice-lstm_28/lstm_cell_28/ReadVariableOp_1:value:03lstm_28/lstm_cell_28/strided_slice_1/stack:output:05lstm_28/lstm_cell_28/strided_slice_1/stack_1:output:05lstm_28/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_28/lstm_cell_28/MatMul_5MatMullstm_28/zeros:output:0-lstm_28/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Аи
lstm_28/lstm_cell_28/add_2AddV2'lstm_28/lstm_cell_28/BiasAdd_1:output:0'lstm_28/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         Аa
lstm_28/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_28/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_28/lstm_cell_28/Mul_1Mullstm_28/lstm_cell_28/add_2:z:0%lstm_28/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЭ
lstm_28/lstm_cell_28/Add_3AddV2lstm_28/lstm_cell_28/Mul_1:z:0%lstm_28/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аs
.lstm_28/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_28/lstm_cell_28/clip_by_value_1/MinimumMinimumlstm_28/lstm_cell_28/Add_3:z:07lstm_28/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_28/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_28/lstm_cell_28/clip_by_value_1Maximum0lstm_28/lstm_cell_28/clip_by_value_1/Minimum:z:0/lstm_28/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АШ
lstm_28/lstm_cell_28/mul_2Mul(lstm_28/lstm_cell_28/clip_by_value_1:z:0lstm_28/zeros_1:output:0*
T0*(
_output_shapes
:         АФ
%lstm_28/lstm_cell_28/ReadVariableOp_2ReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_28/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_28/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_28/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_28/lstm_cell_28/strided_slice_2StridedSlice-lstm_28/lstm_cell_28/ReadVariableOp_2:value:03lstm_28/lstm_cell_28/strided_slice_2/stack:output:05lstm_28/lstm_cell_28/strided_slice_2/stack_1:output:05lstm_28/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_28/lstm_cell_28/MatMul_6MatMullstm_28/zeros:output:0-lstm_28/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Аи
lstm_28/lstm_cell_28/add_4AddV2'lstm_28/lstm_cell_28/BiasAdd_2:output:0'lstm_28/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аt
lstm_28/lstm_cell_28/ReluRelulstm_28/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         Ае
lstm_28/lstm_cell_28/mul_3Mul&lstm_28/lstm_cell_28/clip_by_value:z:0'lstm_28/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АЦ
lstm_28/lstm_cell_28/add_5AddV2lstm_28/lstm_cell_28/mul_2:z:0lstm_28/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АФ
%lstm_28/lstm_cell_28/ReadVariableOp_3ReadVariableOp,lstm_28_lstm_cell_28_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_28/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_28/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_28/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_28/lstm_cell_28/strided_slice_3StridedSlice-lstm_28/lstm_cell_28/ReadVariableOp_3:value:03lstm_28/lstm_cell_28/strided_slice_3/stack:output:05lstm_28/lstm_cell_28/strided_slice_3/stack_1:output:05lstm_28/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_28/lstm_cell_28/MatMul_7MatMullstm_28/zeros:output:0-lstm_28/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Аи
lstm_28/lstm_cell_28/add_6AddV2'lstm_28/lstm_cell_28/BiasAdd_3:output:0'lstm_28/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         Аa
lstm_28/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_28/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_28/lstm_cell_28/Mul_4Mullstm_28/lstm_cell_28/add_6:z:0%lstm_28/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЭ
lstm_28/lstm_cell_28/Add_7AddV2lstm_28/lstm_cell_28/Mul_4:z:0%lstm_28/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аs
.lstm_28/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_28/lstm_cell_28/clip_by_value_2/MinimumMinimumlstm_28/lstm_cell_28/Add_7:z:07lstm_28/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_28/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_28/lstm_cell_28/clip_by_value_2Maximum0lstm_28/lstm_cell_28/clip_by_value_2/Minimum:z:0/lstm_28/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аv
lstm_28/lstm_cell_28/Relu_1Relulstm_28/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Ай
lstm_28/lstm_cell_28/mul_5Mul(lstm_28/lstm_cell_28/clip_by_value_2:z:0)lstm_28/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         Аv
%lstm_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╨
lstm_28/TensorArrayV2_1TensorListReserve.lstm_28/TensorArrayV2_1/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_28/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ь
lstm_28/whileWhile#lstm_28/while/loop_counter:output:0)lstm_28/while/maximum_iterations:output:0lstm_28/time:output:0 lstm_28/TensorArrayV2_1:handle:0lstm_28/zeros:output:0lstm_28/zeros_1:output:0 lstm_28/strided_slice_1:output:0?lstm_28/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_28_lstm_cell_28_split_readvariableop_resource4lstm_28_lstm_cell_28_split_1_readvariableop_resource,lstm_28_lstm_cell_28_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         А:         А: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_28_while_body_236067*%
condR
lstm_28_while_cond_236066*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Й
8lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   █
*lstm_28/TensorArrayV2Stack/TensorListStackTensorListStacklstm_28/while:output:3Alstm_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         А*
element_dtype0p
lstm_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_28/strided_slice_3StridedSlice3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_28/strided_slice_3/stack:output:0(lstm_28/strided_slice_3/stack_1:output:0(lstm_28/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskm
lstm_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
lstm_28/transpose_1	Transpose3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_28/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Аb
lstm_29/ShapeShapelstm_28/transpose_1:y:0*
T0*
_output_shapes
::э╧e
lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_29/strided_sliceStridedSlicelstm_29/Shape:output:0$lstm_29/strided_slice/stack:output:0&lstm_29/strided_slice/stack_1:output:0&lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Л
lstm_29/zeros/packedPacklstm_29/strided_slice:output:0lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
lstm_29/zerosFilllstm_29/zeros/packed:output:0lstm_29/zeros/Const:output:0*
T0*'
_output_shapes
:         @Z
lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@П
lstm_29/zeros_1/packedPacklstm_29/strided_slice:output:0!lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
lstm_29/zeros_1Filllstm_29/zeros_1/packed:output:0lstm_29/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @k
lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          П
lstm_29/transpose	Transposelstm_28/transpose_1:y:0lstm_29/transpose/perm:output:0*
T0*,
_output_shapes
:         Аb
lstm_29/Shape_1Shapelstm_29/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_29/strided_slice_1StridedSlicelstm_29/Shape_1:output:0&lstm_29/strided_slice_1/stack:output:0(lstm_29/strided_slice_1/stack_1:output:0(lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_29/TensorArrayV2TensorListReserve,lstm_29/TensorArrayV2/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   °
/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_29/transpose:y:0Flstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
lstm_29/strided_slice_2StridedSlicelstm_29/transpose:y:0&lstm_29/strided_slice_2/stack:output:0(lstm_29/strided_slice_2/stack_1:output:0(lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskf
$lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
)lstm_29/lstm_cell_29/split/ReadVariableOpReadVariableOp2lstm_29_lstm_cell_29_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0с
lstm_29/lstm_cell_29/splitSplit-lstm_29/lstm_cell_29/split/split_dim:output:01lstm_29/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЮ
lstm_29/lstm_cell_29/MatMulMatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @а
lstm_29/lstm_cell_29/MatMul_1MatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @а
lstm_29/lstm_cell_29/MatMul_2MatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @а
lstm_29/lstm_cell_29/MatMul_3MatMul lstm_29/strided_slice_2:output:0#lstm_29/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @h
&lstm_29/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_29/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4lstm_29_lstm_cell_29_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
lstm_29/lstm_cell_29/split_1Split/lstm_29/lstm_cell_29/split_1/split_dim:output:03lstm_29/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitз
lstm_29/lstm_cell_29/BiasAddBiasAdd%lstm_29/lstm_cell_29/MatMul:product:0%lstm_29/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @л
lstm_29/lstm_cell_29/BiasAdd_1BiasAdd'lstm_29/lstm_cell_29/MatMul_1:product:0%lstm_29/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @л
lstm_29/lstm_cell_29/BiasAdd_2BiasAdd'lstm_29/lstm_cell_29/MatMul_2:product:0%lstm_29/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @л
lstm_29/lstm_cell_29/BiasAdd_3BiasAdd'lstm_29/lstm_cell_29/MatMul_3:product:0%lstm_29/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @С
#lstm_29/lstm_cell_29/ReadVariableOpReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0y
(lstm_29/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_29/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm_29/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"lstm_29/lstm_cell_29/strided_sliceStridedSlice+lstm_29/lstm_cell_29/ReadVariableOp:value:01lstm_29/lstm_cell_29/strided_slice/stack:output:03lstm_29/lstm_cell_29/strided_slice/stack_1:output:03lstm_29/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЮ
lstm_29/lstm_cell_29/MatMul_4MatMullstm_29/zeros:output:0+lstm_29/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @г
lstm_29/lstm_cell_29/addAddV2%lstm_29/lstm_cell_29/BiasAdd:output:0'lstm_29/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @_
lstm_29/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_29/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
lstm_29/lstm_cell_29/MulMullstm_29/lstm_cell_29/add:z:0#lstm_29/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ъ
lstm_29/lstm_cell_29/Add_1AddV2lstm_29/lstm_cell_29/Mul:z:0%lstm_29/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @q
,lstm_29/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╛
*lstm_29/lstm_cell_29/clip_by_value/MinimumMinimumlstm_29/lstm_cell_29/Add_1:z:05lstm_29/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$lstm_29/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"lstm_29/lstm_cell_29/clip_by_valueMaximum.lstm_29/lstm_cell_29/clip_by_value/Minimum:z:0-lstm_29/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @У
%lstm_29/lstm_cell_29/ReadVariableOp_1ReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_29/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   }
,lstm_29/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_29/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_29/lstm_cell_29/strided_slice_1StridedSlice-lstm_29/lstm_cell_29/ReadVariableOp_1:value:03lstm_29/lstm_cell_29/strided_slice_1/stack:output:05lstm_29/lstm_cell_29/strided_slice_1/stack_1:output:05lstm_29/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_29/lstm_cell_29/MatMul_5MatMullstm_29/zeros:output:0-lstm_29/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @з
lstm_29/lstm_cell_29/add_2AddV2'lstm_29/lstm_cell_29/BiasAdd_1:output:0'lstm_29/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @a
lstm_29/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_29/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_29/lstm_cell_29/Mul_1Mullstm_29/lstm_cell_29/add_2:z:0%lstm_29/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ь
lstm_29/lstm_cell_29/Add_3AddV2lstm_29/lstm_cell_29/Mul_1:z:0%lstm_29/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @s
.lstm_29/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_29/lstm_cell_29/clip_by_value_1/MinimumMinimumlstm_29/lstm_cell_29/Add_3:z:07lstm_29/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_29/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_29/lstm_cell_29/clip_by_value_1Maximum0lstm_29/lstm_cell_29/clip_by_value_1/Minimum:z:0/lstm_29/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @Ч
lstm_29/lstm_cell_29/mul_2Mul(lstm_29/lstm_cell_29/clip_by_value_1:z:0lstm_29/zeros_1:output:0*
T0*'
_output_shapes
:         @У
%lstm_29/lstm_cell_29/ReadVariableOp_2ReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_29/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_29/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_29/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_29/lstm_cell_29/strided_slice_2StridedSlice-lstm_29/lstm_cell_29/ReadVariableOp_2:value:03lstm_29/lstm_cell_29/strided_slice_2/stack:output:05lstm_29/lstm_cell_29/strided_slice_2/stack_1:output:05lstm_29/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_29/lstm_cell_29/MatMul_6MatMullstm_29/zeros:output:0-lstm_29/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @з
lstm_29/lstm_cell_29/add_4AddV2'lstm_29/lstm_cell_29/BiasAdd_2:output:0'lstm_29/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @s
lstm_29/lstm_cell_29/ReluRelulstm_29/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @д
lstm_29/lstm_cell_29/mul_3Mul&lstm_29/lstm_cell_29/clip_by_value:z:0'lstm_29/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @Х
lstm_29/lstm_cell_29/add_5AddV2lstm_29/lstm_cell_29/mul_2:z:0lstm_29/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @У
%lstm_29/lstm_cell_29/ReadVariableOp_3ReadVariableOp,lstm_29_lstm_cell_29_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_29/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_29/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_29/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_29/lstm_cell_29/strided_slice_3StridedSlice-lstm_29/lstm_cell_29/ReadVariableOp_3:value:03lstm_29/lstm_cell_29/strided_slice_3/stack:output:05lstm_29/lstm_cell_29/strided_slice_3/stack_1:output:05lstm_29/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_29/lstm_cell_29/MatMul_7MatMullstm_29/zeros:output:0-lstm_29/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @з
lstm_29/lstm_cell_29/add_6AddV2'lstm_29/lstm_cell_29/BiasAdd_3:output:0'lstm_29/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @a
lstm_29/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_29/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_29/lstm_cell_29/Mul_4Mullstm_29/lstm_cell_29/add_6:z:0%lstm_29/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ь
lstm_29/lstm_cell_29/Add_7AddV2lstm_29/lstm_cell_29/Mul_4:z:0%lstm_29/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @s
.lstm_29/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_29/lstm_cell_29/clip_by_value_2/MinimumMinimumlstm_29/lstm_cell_29/Add_7:z:07lstm_29/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_29/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_29/lstm_cell_29/clip_by_value_2Maximum0lstm_29/lstm_cell_29/clip_by_value_2/Minimum:z:0/lstm_29/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @u
lstm_29/lstm_cell_29/Relu_1Relulstm_29/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @и
lstm_29/lstm_cell_29/mul_5Mul(lstm_29/lstm_cell_29/clip_by_value_2:z:0)lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @v
%lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╨
lstm_29/TensorArrayV2_1TensorListReserve.lstm_29/TensorArrayV2_1/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
lstm_29/whileWhile#lstm_29/while/loop_counter:output:0)lstm_29/while/maximum_iterations:output:0lstm_29/time:output:0 lstm_29/TensorArrayV2_1:handle:0lstm_29/zeros:output:0lstm_29/zeros_1:output:0 lstm_29/strided_slice_1:output:0?lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_29_lstm_cell_29_split_readvariableop_resource4lstm_29_lstm_cell_29_split_1_readvariableop_resource,lstm_29_lstm_cell_29_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_29_while_body_236319*%
condR
lstm_29_while_cond_236318*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Й
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┌
*lstm_29/TensorArrayV2Stack/TensorListStackTensorListStacklstm_29/while:output:3Alstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0p
lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
lstm_29/strided_slice_3StridedSlice3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_29/strided_slice_3/stack:output:0(lstm_29/strided_slice_3/stack_1:output:0(lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskm
lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
lstm_29/transpose_1	Transpose3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_29/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @Ж
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Х
dense_14/MatMulMatMul lstm_29/strided_slice_3:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp$^lstm_28/lstm_cell_28/ReadVariableOp&^lstm_28/lstm_cell_28/ReadVariableOp_1&^lstm_28/lstm_cell_28/ReadVariableOp_2&^lstm_28/lstm_cell_28/ReadVariableOp_3*^lstm_28/lstm_cell_28/split/ReadVariableOp,^lstm_28/lstm_cell_28/split_1/ReadVariableOp^lstm_28/while$^lstm_29/lstm_cell_29/ReadVariableOp&^lstm_29/lstm_cell_29/ReadVariableOp_1&^lstm_29/lstm_cell_29/ReadVariableOp_2&^lstm_29/lstm_cell_29/ReadVariableOp_3*^lstm_29/lstm_cell_29/split/ReadVariableOp,^lstm_29/lstm_cell_29/split_1/ReadVariableOp^lstm_29/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2N
%lstm_28/lstm_cell_28/ReadVariableOp_1%lstm_28/lstm_cell_28/ReadVariableOp_12N
%lstm_28/lstm_cell_28/ReadVariableOp_2%lstm_28/lstm_cell_28/ReadVariableOp_22N
%lstm_28/lstm_cell_28/ReadVariableOp_3%lstm_28/lstm_cell_28/ReadVariableOp_32J
#lstm_28/lstm_cell_28/ReadVariableOp#lstm_28/lstm_cell_28/ReadVariableOp2V
)lstm_28/lstm_cell_28/split/ReadVariableOp)lstm_28/lstm_cell_28/split/ReadVariableOp2Z
+lstm_28/lstm_cell_28/split_1/ReadVariableOp+lstm_28/lstm_cell_28/split_1/ReadVariableOp2
lstm_28/whilelstm_28/while2N
%lstm_29/lstm_cell_29/ReadVariableOp_1%lstm_29/lstm_cell_29/ReadVariableOp_12N
%lstm_29/lstm_cell_29/ReadVariableOp_2%lstm_29/lstm_cell_29/ReadVariableOp_22N
%lstm_29/lstm_cell_29/ReadVariableOp_3%lstm_29/lstm_cell_29/ReadVariableOp_32J
#lstm_29/lstm_cell_29/ReadVariableOp#lstm_29/lstm_cell_29/ReadVariableOp2V
)lstm_29/lstm_cell_29/split/ReadVariableOp)lstm_29/lstm_cell_29/split/ReadVariableOp2Z
+lstm_29/lstm_cell_29/split_1/ReadVariableOp+lstm_29/lstm_cell_29/split_1/ReadVariableOp2
lstm_29/whilelstm_29/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ч	
├
while_cond_238204
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_238204___redundant_placeholder04
0while_while_cond_238204___redundant_placeholder14
0while_while_cond_238204___redundant_placeholder24
0while_while_cond_238204___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┬
Ц
)__inference_dense_14_layer_call_fn_238610

inputs
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_234642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
М
у
lstm_28_while_cond_235552,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3.
*lstm_28_while_less_lstm_28_strided_slice_1D
@lstm_28_while_lstm_28_while_cond_235552___redundant_placeholder0D
@lstm_28_while_lstm_28_while_cond_235552___redundant_placeholder1D
@lstm_28_while_lstm_28_while_cond_235552___redundant_placeholder2D
@lstm_28_while_lstm_28_while_cond_235552___redundant_placeholder3
lstm_28_while_identity
В
lstm_28/while/LessLesslstm_28_while_placeholder*lstm_28_while_less_lstm_28_strided_slice_1*
T0*
_output_shapes
: [
lstm_28/while/IdentityIdentitylstm_28/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_28_while_identitylstm_28/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :XT

_output_shapes
: 
:
_user_specified_name" lstm_28/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_28/while/loop_counter
Ч	
├
while_cond_237948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237948___redundant_placeholder04
0while_while_cond_237948___redundant_placeholder14
0while_while_cond_237948___redundant_placeholder24
0while_while_cond_237948___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: :::::

_output_shapes
::

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ы	
├
while_cond_237392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237392___redundant_placeholder04
0while_while_cond_237392___redundant_placeholder14
0while_while_cond_237392___redundant_placeholder24
0while_while_cond_237392___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         А:         А: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Єо
к
'sequential_14_lstm_29_while_body_233029H
Dsequential_14_lstm_29_while_sequential_14_lstm_29_while_loop_counterN
Jsequential_14_lstm_29_while_sequential_14_lstm_29_while_maximum_iterations+
'sequential_14_lstm_29_while_placeholder-
)sequential_14_lstm_29_while_placeholder_1-
)sequential_14_lstm_29_while_placeholder_2-
)sequential_14_lstm_29_while_placeholder_3G
Csequential_14_lstm_29_while_sequential_14_lstm_29_strided_slice_1_0Г
sequential_14_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_29_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_14_lstm_29_while_lstm_cell_29_split_readvariableop_resource_0:
ААY
Jsequential_14_lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0:	АU
Bsequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource_0:	@А(
$sequential_14_lstm_29_while_identity*
&sequential_14_lstm_29_while_identity_1*
&sequential_14_lstm_29_while_identity_2*
&sequential_14_lstm_29_while_identity_3*
&sequential_14_lstm_29_while_identity_4*
&sequential_14_lstm_29_while_identity_5E
Asequential_14_lstm_29_while_sequential_14_lstm_29_strided_slice_1Б
}sequential_14_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_29_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_14_lstm_29_while_lstm_cell_29_split_readvariableop_resource:
ААW
Hsequential_14_lstm_29_while_lstm_cell_29_split_1_readvariableop_resource:	АS
@sequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource:	@АИв7sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOpв9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_1в9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_2в9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_3в=sequential_14/lstm_29/while/lstm_cell_29/split/ReadVariableOpв?sequential_14/lstm_29/while/lstm_cell_29/split_1/ReadVariableOpЮ
Msequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   Х
?sequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_14_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_29_tensorarrayunstack_tensorlistfromtensor_0'sequential_14_lstm_29_while_placeholderVsequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0z
8sequential_14/lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
=sequential_14/lstm_29/while/lstm_cell_29/split/ReadVariableOpReadVariableOpHsequential_14_lstm_29_while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Э
.sequential_14/lstm_29/while/lstm_cell_29/splitSplitAsequential_14/lstm_29/while/lstm_cell_29/split/split_dim:output:0Esequential_14/lstm_29/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitь
/sequential_14/lstm_29/while/lstm_cell_29/MatMulMatMulFsequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_29/while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @ю
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_1MatMulFsequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_29/while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @ю
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_2MatMulFsequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_29/while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @ю
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_3MatMulFsequential_14/lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_14/lstm_29/while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @|
:sequential_14/lstm_29/while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╟
?sequential_14/lstm_29/while/lstm_cell_29/split_1/ReadVariableOpReadVariableOpJsequential_14_lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0П
0sequential_14/lstm_29/while/lstm_cell_29/split_1SplitCsequential_14/lstm_29/while/lstm_cell_29/split_1/split_dim:output:0Gsequential_14/lstm_29/while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitу
0sequential_14/lstm_29/while/lstm_cell_29/BiasAddBiasAdd9sequential_14/lstm_29/while/lstm_cell_29/MatMul:product:09sequential_14/lstm_29/while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @ч
2sequential_14/lstm_29/while/lstm_cell_29/BiasAdd_1BiasAdd;sequential_14/lstm_29/while/lstm_cell_29/MatMul_1:product:09sequential_14/lstm_29/while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @ч
2sequential_14/lstm_29/while/lstm_cell_29/BiasAdd_2BiasAdd;sequential_14/lstm_29/while/lstm_cell_29/MatMul_2:product:09sequential_14/lstm_29/while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @ч
2sequential_14/lstm_29/while/lstm_cell_29/BiasAdd_3BiasAdd;sequential_14/lstm_29/while/lstm_cell_29/MatMul_3:product:09sequential_14/lstm_29/while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @╗
7sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOpReadVariableOpBsequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Н
<sequential_14/lstm_29/while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        П
>sequential_14/lstm_29/while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   П
>sequential_14/lstm_29/while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
6sequential_14/lstm_29/while/lstm_cell_29/strided_sliceStridedSlice?sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp:value:0Esequential_14/lstm_29/while/lstm_cell_29/strided_slice/stack:output:0Gsequential_14/lstm_29/while/lstm_cell_29/strided_slice/stack_1:output:0Gsequential_14/lstm_29/while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask┘
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_4MatMul)sequential_14_lstm_29_while_placeholder_2?sequential_14/lstm_29/while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @▀
,sequential_14/lstm_29/while/lstm_cell_29/addAddV29sequential_14/lstm_29/while/lstm_cell_29/BiasAdd:output:0;sequential_14/lstm_29/while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @s
.sequential_14/lstm_29/while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_14/lstm_29/while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╨
,sequential_14/lstm_29/while/lstm_cell_29/MulMul0sequential_14/lstm_29/while/lstm_cell_29/add:z:07sequential_14/lstm_29/while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @╓
.sequential_14/lstm_29/while/lstm_cell_29/Add_1AddV20sequential_14/lstm_29/while/lstm_cell_29/Mul:z:09sequential_14/lstm_29/while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @Е
@sequential_14/lstm_29/while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?·
>sequential_14/lstm_29/while/lstm_cell_29/clip_by_value/MinimumMinimum2sequential_14/lstm_29/while/lstm_cell_29/Add_1:z:0Isequential_14/lstm_29/while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @}
8sequential_14/lstm_29/while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
6sequential_14/lstm_29/while/lstm_cell_29/clip_by_valueMaximumBsequential_14/lstm_29/while/lstm_cell_29/clip_by_value/Minimum:z:0Asequential_14/lstm_29/while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @╜
9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_1ReadVariableOpBsequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0П
>sequential_14/lstm_29/while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   С
@sequential_14/lstm_29/while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   С
@sequential_14/lstm_29/while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_14/lstm_29/while/lstm_cell_29/strided_slice_1StridedSliceAsequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_1:value:0Gsequential_14/lstm_29/while/lstm_cell_29/strided_slice_1/stack:output:0Isequential_14/lstm_29/while/lstm_cell_29/strided_slice_1/stack_1:output:0Isequential_14/lstm_29/while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask█
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_5MatMul)sequential_14_lstm_29_while_placeholder_2Asequential_14/lstm_29/while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @у
.sequential_14/lstm_29/while/lstm_cell_29/add_2AddV2;sequential_14/lstm_29/while/lstm_cell_29/BiasAdd_1:output:0;sequential_14/lstm_29/while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @u
0sequential_14/lstm_29/while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_14/lstm_29/while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╓
.sequential_14/lstm_29/while/lstm_cell_29/Mul_1Mul2sequential_14/lstm_29/while/lstm_cell_29/add_2:z:09sequential_14/lstm_29/while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @╪
.sequential_14/lstm_29/while/lstm_cell_29/Add_3AddV22sequential_14/lstm_29/while/lstm_cell_29/Mul_1:z:09sequential_14/lstm_29/while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @З
Bsequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?■
@sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1/MinimumMinimum2sequential_14/lstm_29/while/lstm_cell_29/Add_3:z:0Ksequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @
:sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
8sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1MaximumDsequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1/Minimum:z:0Csequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @╨
.sequential_14/lstm_29/while/lstm_cell_29/mul_2Mul<sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_1:z:0)sequential_14_lstm_29_while_placeholder_3*
T0*'
_output_shapes
:         @╜
9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_2ReadVariableOpBsequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0П
>sequential_14/lstm_29/while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   С
@sequential_14/lstm_29/while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   С
@sequential_14/lstm_29/while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_14/lstm_29/while/lstm_cell_29/strided_slice_2StridedSliceAsequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_2:value:0Gsequential_14/lstm_29/while/lstm_cell_29/strided_slice_2/stack:output:0Isequential_14/lstm_29/while/lstm_cell_29/strided_slice_2/stack_1:output:0Isequential_14/lstm_29/while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask█
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_6MatMul)sequential_14_lstm_29_while_placeholder_2Asequential_14/lstm_29/while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @у
.sequential_14/lstm_29/while/lstm_cell_29/add_4AddV2;sequential_14/lstm_29/while/lstm_cell_29/BiasAdd_2:output:0;sequential_14/lstm_29/while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @Ы
-sequential_14/lstm_29/while/lstm_cell_29/ReluRelu2sequential_14/lstm_29/while/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @р
.sequential_14/lstm_29/while/lstm_cell_29/mul_3Mul:sequential_14/lstm_29/while/lstm_cell_29/clip_by_value:z:0;sequential_14/lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @╤
.sequential_14/lstm_29/while/lstm_cell_29/add_5AddV22sequential_14/lstm_29/while/lstm_cell_29/mul_2:z:02sequential_14/lstm_29/while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @╜
9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_3ReadVariableOpBsequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0П
>sequential_14/lstm_29/while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   С
@sequential_14/lstm_29/while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        С
@sequential_14/lstm_29/while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_14/lstm_29/while/lstm_cell_29/strided_slice_3StridedSliceAsequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_3:value:0Gsequential_14/lstm_29/while/lstm_cell_29/strided_slice_3/stack:output:0Isequential_14/lstm_29/while/lstm_cell_29/strided_slice_3/stack_1:output:0Isequential_14/lstm_29/while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask█
1sequential_14/lstm_29/while/lstm_cell_29/MatMul_7MatMul)sequential_14_lstm_29_while_placeholder_2Asequential_14/lstm_29/while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @у
.sequential_14/lstm_29/while/lstm_cell_29/add_6AddV2;sequential_14/lstm_29/while/lstm_cell_29/BiasAdd_3:output:0;sequential_14/lstm_29/while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @u
0sequential_14/lstm_29/while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_14/lstm_29/while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╓
.sequential_14/lstm_29/while/lstm_cell_29/Mul_4Mul2sequential_14/lstm_29/while/lstm_cell_29/add_6:z:09sequential_14/lstm_29/while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @╪
.sequential_14/lstm_29/while/lstm_cell_29/Add_7AddV22sequential_14/lstm_29/while/lstm_cell_29/Mul_4:z:09sequential_14/lstm_29/while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @З
Bsequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?■
@sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2/MinimumMinimum2sequential_14/lstm_29/while/lstm_cell_29/Add_7:z:0Ksequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @
:sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
8sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2MaximumDsequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2/Minimum:z:0Csequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @Э
/sequential_14/lstm_29/while/lstm_cell_29/Relu_1Relu2sequential_14/lstm_29/while/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @ф
.sequential_14/lstm_29/while/lstm_cell_29/mul_5Mul<sequential_14/lstm_29/while/lstm_cell_29/clip_by_value_2:z:0=sequential_14/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @Э
@sequential_14/lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_14_lstm_29_while_placeholder_1'sequential_14_lstm_29_while_placeholder2sequential_14/lstm_29/while/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥c
!sequential_14/lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ю
sequential_14/lstm_29/while/addAddV2'sequential_14_lstm_29_while_placeholder*sequential_14/lstm_29/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_14/lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_14/lstm_29/while/add_1AddV2Dsequential_14_lstm_29_while_sequential_14_lstm_29_while_loop_counter,sequential_14/lstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: Ы
$sequential_14/lstm_29/while/IdentityIdentity%sequential_14/lstm_29/while/add_1:z:0!^sequential_14/lstm_29/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_14/lstm_29/while/Identity_1IdentityJsequential_14_lstm_29_while_sequential_14_lstm_29_while_maximum_iterations!^sequential_14/lstm_29/while/NoOp*
T0*
_output_shapes
: Ы
&sequential_14/lstm_29/while/Identity_2Identity#sequential_14/lstm_29/while/add:z:0!^sequential_14/lstm_29/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_14/lstm_29/while/Identity_3IdentityPsequential_14/lstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_14/lstm_29/while/NoOp*
T0*
_output_shapes
: ╗
&sequential_14/lstm_29/while/Identity_4Identity2sequential_14/lstm_29/while/lstm_cell_29/mul_5:z:0!^sequential_14/lstm_29/while/NoOp*
T0*'
_output_shapes
:         @╗
&sequential_14/lstm_29/while/Identity_5Identity2sequential_14/lstm_29/while/lstm_cell_29/add_5:z:0!^sequential_14/lstm_29/while/NoOp*
T0*'
_output_shapes
:         @╥
 sequential_14/lstm_29/while/NoOpNoOp8^sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp:^sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_1:^sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_2:^sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_3>^sequential_14/lstm_29/while/lstm_cell_29/split/ReadVariableOp@^sequential_14/lstm_29/while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Y
&sequential_14_lstm_29_while_identity_1/sequential_14/lstm_29/while/Identity_1:output:0"Y
&sequential_14_lstm_29_while_identity_2/sequential_14/lstm_29/while/Identity_2:output:0"Y
&sequential_14_lstm_29_while_identity_3/sequential_14/lstm_29/while/Identity_3:output:0"Y
&sequential_14_lstm_29_while_identity_4/sequential_14/lstm_29/while/Identity_4:output:0"Y
&sequential_14_lstm_29_while_identity_5/sequential_14/lstm_29/while/Identity_5:output:0"U
$sequential_14_lstm_29_while_identity-sequential_14/lstm_29/while/Identity:output:0"Ж
@sequential_14_lstm_29_while_lstm_cell_29_readvariableop_resourceBsequential_14_lstm_29_while_lstm_cell_29_readvariableop_resource_0"Ц
Hsequential_14_lstm_29_while_lstm_cell_29_split_1_readvariableop_resourceJsequential_14_lstm_29_while_lstm_cell_29_split_1_readvariableop_resource_0"Т
Fsequential_14_lstm_29_while_lstm_cell_29_split_readvariableop_resourceHsequential_14_lstm_29_while_lstm_cell_29_split_readvariableop_resource_0"И
Asequential_14_lstm_29_while_sequential_14_lstm_29_strided_slice_1Csequential_14_lstm_29_while_sequential_14_lstm_29_strided_slice_1_0"А
}sequential_14_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_29_tensorarrayunstack_tensorlistfromtensorsequential_14_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_14_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2v
9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_19sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_12v
9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_29sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_22v
9sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_39sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp_32r
7sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp7sequential_14/lstm_29/while/lstm_cell_29/ReadVariableOp2~
=sequential_14/lstm_29/while/lstm_cell_29/split/ReadVariableOp=sequential_14/lstm_29/while/lstm_cell_29/split/ReadVariableOp2В
?sequential_14/lstm_29/while/lstm_cell_29/split_1/ReadVariableOp?sequential_14/lstm_29/while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :fb

_output_shapes
: 
H
_user_specified_name0.sequential_14/lstm_29/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_14/lstm_29/while/loop_counter
б~
ж	
while_body_237393
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_28_split_readvariableop_resource_0:	АC
4while_lstm_cell_28_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_28_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_28_split_readvariableop_resource:	АA
2while_lstm_cell_28_split_1_readvariableop_resource:	А>
*while_lstm_cell_28_readvariableop_resource:
ААИв!while/lstm_cell_28/ReadVariableOpв#while/lstm_cell_28/ReadVariableOp_1в#while/lstm_cell_28/ReadVariableOp_2в#while/lstm_cell_28/ReadVariableOp_3в'while/lstm_cell_28/split/ReadVariableOpв)while/lstm_cell_28/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_28/split/ReadVariableOpReadVariableOp2while_lstm_cell_28_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0/while/lstm_cell_28/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_28/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_28/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_28_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_28/split_1Split-while/lstm_cell_28/split_1/split_dim:output:01while/lstm_cell_28/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_28/BiasAddBiasAdd#while/lstm_cell_28/MatMul:product:0#while/lstm_cell_28/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_1BiasAdd%while/lstm_cell_28/MatMul_1:product:0#while/lstm_cell_28/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_2BiasAdd%while/lstm_cell_28/MatMul_2:product:0#while/lstm_cell_28/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_28/BiasAdd_3BiasAdd%while/lstm_cell_28/MatMul_3:product:0#while/lstm_cell_28/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_28/ReadVariableOpReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_28/strided_sliceStridedSlice)while/lstm_cell_28/ReadVariableOp:value:0/while/lstm_cell_28/strided_slice/stack:output:01while/lstm_cell_28/strided_slice/stack_1:output:01while/lstm_cell_28/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_28/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_28/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_28/addAddV2#while/lstm_cell_28/BiasAdd:output:0%while/lstm_cell_28/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_28/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_28/MulMulwhile/lstm_cell_28/add:z:0!while/lstm_cell_28/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_28/Add_1AddV2while/lstm_cell_28/Mul:z:0#while/lstm_cell_28/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_28/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_28/clip_by_value/MinimumMinimumwhile/lstm_cell_28/Add_1:z:03while/lstm_cell_28/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_28/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_28/clip_by_valueMaximum,while/lstm_cell_28/clip_by_value/Minimum:z:0+while/lstm_cell_28/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_1ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_1StridedSlice+while/lstm_cell_28/ReadVariableOp_1:value:01while/lstm_cell_28/strided_slice_1/stack:output:03while/lstm_cell_28/strided_slice_1/stack_1:output:03while/lstm_cell_28/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_2AddV2%while/lstm_cell_28/BiasAdd_1:output:0%while/lstm_cell_28/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_1Mulwhile/lstm_cell_28/add_2:z:0#while/lstm_cell_28/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_3AddV2while/lstm_cell_28/Mul_1:z:0#while/lstm_cell_28/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_1/MinimumMinimumwhile/lstm_cell_28/Add_3:z:05while/lstm_cell_28/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_1Maximum.while/lstm_cell_28/clip_by_value_1/Minimum:z:0-while/lstm_cell_28/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_28/mul_2Mul&while/lstm_cell_28/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_2ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_2StridedSlice+while/lstm_cell_28/ReadVariableOp_2:value:01while/lstm_cell_28/strided_slice_2/stack:output:03while/lstm_cell_28/strided_slice_2/stack_1:output:03while/lstm_cell_28/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_4AddV2%while/lstm_cell_28/BiasAdd_2:output:0%while/lstm_cell_28/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_28/ReluReluwhile/lstm_cell_28/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_28/mul_3Mul$while/lstm_cell_28/clip_by_value:z:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_28/add_5AddV2while/lstm_cell_28/mul_2:z:0while/lstm_cell_28/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_28/ReadVariableOp_3ReadVariableOp,while_lstm_cell_28_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_28/strided_slice_3StridedSlice+while/lstm_cell_28/ReadVariableOp_3:value:01while/lstm_cell_28/strided_slice_3/stack:output:03while/lstm_cell_28/strided_slice_3/stack_1:output:03while/lstm_cell_28/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_28/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_28/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_28/add_6AddV2%while/lstm_cell_28/BiasAdd_3:output:0%while/lstm_cell_28/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_28/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_28/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_28/Mul_4Mulwhile/lstm_cell_28/add_6:z:0#while/lstm_cell_28/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_28/Add_7AddV2while/lstm_cell_28/Mul_4:z:0#while/lstm_cell_28/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_28/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_28/clip_by_value_2/MinimumMinimumwhile/lstm_cell_28/Add_7:z:05while/lstm_cell_28/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_28/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_28/clip_by_value_2Maximum.while/lstm_cell_28/clip_by_value_2/Minimum:z:0-while/lstm_cell_28/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_28/mul_5Mul&while/lstm_cell_28/clip_by_value_2:z:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_28/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_28/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_28/ReadVariableOp$^while/lstm_cell_28/ReadVariableOp_1$^while/lstm_cell_28/ReadVariableOp_2$^while/lstm_cell_28/ReadVariableOp_3(^while/lstm_cell_28/split/ReadVariableOp*^while/lstm_cell_28/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_28_readvariableop_resource,while_lstm_cell_28_readvariableop_resource_0"j
2while_lstm_cell_28_split_1_readvariableop_resource4while_lstm_cell_28_split_1_readvariableop_resource_0"f
0while_lstm_cell_28_split_readvariableop_resource2while_lstm_cell_28_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_28/ReadVariableOp_1#while/lstm_cell_28/ReadVariableOp_12J
#while/lstm_cell_28/ReadVariableOp_2#while/lstm_cell_28/ReadVariableOp_22J
#while/lstm_cell_28/ReadVariableOp_3#while/lstm_cell_28/ReadVariableOp_32F
!while/lstm_cell_28/ReadVariableOp!while/lstm_cell_28/ReadVariableOp2R
'while/lstm_cell_28/split/ReadVariableOp'while/lstm_cell_28/split/ReadVariableOp2V
)while/lstm_cell_28/split_1/ReadVariableOp)while/lstm_cell_28/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         А:.*
(
_output_shapes
:         А:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
тJ
м
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_238955

inputs
states_0
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:         @\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:         @\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:         @\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:         @S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         @l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         @l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         @l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         @g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ы
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskf
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         @J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:         @[
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:         @\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskh
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:         @h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         @L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:         @]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:         @i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:         @h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         @I
ReluRelu	add_4:z:0*
T0*'
_output_shapes
:         @e
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*'
_output_shapes
:         @V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:         @i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskh
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:         @h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         @L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:         @]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:         @^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_5:z:0*
T0*'
_output_shapes
:         @i
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*'
_output_shapes
:         @X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:         @Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:         @└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         А:         @:         @: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:QM
'
_output_shapes
:         @
"
_user_specified_name
states_1:QM
'
_output_shapes
:         @
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ЪK
м
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238832

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А+
readvariableop_resource:
АА
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:         А]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:         А]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:         А]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:         АS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:         Аm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:         Аm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:         Аh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      э
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:         Аe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:         АJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:         А\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:         А\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         АT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:         Аi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:         АL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:         А^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:         Аj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:         Аi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:         АJ
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:         Аf
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:         АW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:         Аi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:         АL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:         А^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АL
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:         Аj
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:         А└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         А:         А: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:RN
(
_output_shapes
:         А
"
_user_specified_name
states_1:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы}
ж	
while_body_234807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_29_split_readvariableop_resource_0:
ААC
4while_lstm_cell_29_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_29_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_29_split_readvariableop_resource:
ААA
2while_lstm_cell_29_split_1_readvariableop_resource:	А=
*while_lstm_cell_29_readvariableop_resource:	@АИв!while/lstm_cell_29/ReadVariableOpв#while/lstm_cell_29/ReadVariableOp_1в#while/lstm_cell_29/ReadVariableOp_2в#while/lstm_cell_29/ReadVariableOp_3в'while/lstm_cell_29/split/ReadVariableOpв)while/lstm_cell_29/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_29/split/ReadVariableOpReadVariableOp2while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_29/split_1Split-while/lstm_cell_29/split_1/split_dim:output:01while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_29/BiasAddBiasAdd#while/lstm_cell_29/MatMul:product:0#while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_1BiasAdd%while/lstm_cell_29/MatMul_1:product:0#while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_2BiasAdd%while/lstm_cell_29/MatMul_2:product:0#while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_3BiasAdd%while/lstm_cell_29/MatMul_3:product:0#while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_29/ReadVariableOpReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_29/strided_sliceStridedSlice)while/lstm_cell_29/ReadVariableOp:value:0/while/lstm_cell_29/strided_slice/stack:output:01while/lstm_cell_29/strided_slice/stack_1:output:01while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_29/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_29/addAddV2#while/lstm_cell_29/BiasAdd:output:0%while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_29/MulMulwhile/lstm_cell_29/add:z:0!while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_29/Add_1AddV2while/lstm_cell_29/Mul:z:0#while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_29/clip_by_value/MinimumMinimumwhile/lstm_cell_29/Add_1:z:03while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_29/clip_by_valueMaximum,while/lstm_cell_29/clip_by_value/Minimum:z:0+while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_1ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_1StridedSlice+while/lstm_cell_29/ReadVariableOp_1:value:01while/lstm_cell_29/strided_slice_1/stack:output:03while/lstm_cell_29/strided_slice_1/stack_1:output:03while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_2AddV2%while/lstm_cell_29/BiasAdd_1:output:0%while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_1Mulwhile/lstm_cell_29/add_2:z:0#while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_3AddV2while/lstm_cell_29/Mul_1:z:0#while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_1/MinimumMinimumwhile/lstm_cell_29/Add_3:z:05while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_1Maximum.while/lstm_cell_29/clip_by_value_1/Minimum:z:0-while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_29/mul_2Mul&while/lstm_cell_29/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_2ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_2StridedSlice+while/lstm_cell_29/ReadVariableOp_2:value:01while/lstm_cell_29/strided_slice_2/stack:output:03while/lstm_cell_29/strided_slice_2/stack_1:output:03while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_4AddV2%while/lstm_cell_29/BiasAdd_2:output:0%while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_29/ReluReluwhile/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_29/mul_3Mul$while/lstm_cell_29/clip_by_value:z:0%while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_29/add_5AddV2while/lstm_cell_29/mul_2:z:0while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_3ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_3StridedSlice+while/lstm_cell_29/ReadVariableOp_3:value:01while/lstm_cell_29/strided_slice_3/stack:output:03while/lstm_cell_29/strided_slice_3/stack_1:output:03while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_6AddV2%while/lstm_cell_29/BiasAdd_3:output:0%while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_4Mulwhile/lstm_cell_29/add_6:z:0#while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_7AddV2while/lstm_cell_29/Mul_4:z:0#while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_2/MinimumMinimumwhile/lstm_cell_29/Add_7:z:05while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_2Maximum.while/lstm_cell_29/clip_by_value_2/Minimum:z:0-while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_29/mul_5Mul&while/lstm_cell_29/clip_by_value_2:z:0'while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_29/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_29/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_29/ReadVariableOp$^while/lstm_cell_29/ReadVariableOp_1$^while/lstm_cell_29/ReadVariableOp_2$^while/lstm_cell_29/ReadVariableOp_3(^while/lstm_cell_29/split/ReadVariableOp*^while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_29_readvariableop_resource,while_lstm_cell_29_readvariableop_resource_0"j
2while_lstm_cell_29_split_1_readvariableop_resource4while_lstm_cell_29_split_1_readvariableop_resource_0"f
0while_lstm_cell_29_split_readvariableop_resource2while_lstm_cell_29_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_29/ReadVariableOp_1#while/lstm_cell_29/ReadVariableOp_12J
#while/lstm_cell_29/ReadVariableOp_2#while/lstm_cell_29/ReadVariableOp_22J
#while/lstm_cell_29/ReadVariableOp_3#while/lstm_cell_29/ReadVariableOp_32F
!while/lstm_cell_29/ReadVariableOp!while/lstm_cell_29/ReadVariableOp2R
'while/lstm_cell_29/split/ReadVariableOp'while/lstm_cell_29/split/ReadVariableOp2V
)while/lstm_cell_29/split_1/ReadVariableOp)while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ЪK
м
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238743

inputs
states_0
states_10
split_readvariableop_resource:	А.
split_1_readvariableop_resource:	А+
readvariableop_resource:
АА
identity

identity_1

identity_2ИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вsplit/ReadVariableOpвsplit_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	А*
dtype0в
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:         А]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:         А]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:         А]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:         АS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:         Аm
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:         Аm
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:         Аh
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      э
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:         Аe
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:         АJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:         А\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:         А\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         АT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:         Аi
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:         АL
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:         А^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:         Аj
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:         Аi
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:         АJ
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:         Аf
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:         АW
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:         Аj
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
АА*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:         Аi
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:         АL
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:         А^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:         А^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Д
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         АV
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АL
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:         Аj
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:         А└
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         А:         А: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:RN
(
_output_shapes
:         А
"
_user_specified_name
states_1:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
м
╕
(__inference_lstm_28_layer_call_fn_236487
inputs_0
unknown:	А
	unknown_0:	А
	unknown_1:
АА
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_233628}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Ё
▌
I__inference_sequential_14_layer_call_and_return_conditional_losses_234649

inputs!
lstm_28_234362:	А
lstm_28_234364:	А"
lstm_28_234366:
АА"
lstm_29_234625:
АА
lstm_29_234627:	А!
lstm_29_234629:	@А!
dense_14_234643:@
dense_14_234645:
identityИв dense_14/StatefulPartitionedCallвlstm_28/StatefulPartitionedCallвlstm_29/StatefulPartitionedCallГ
lstm_28/StatefulPartitionedCallStatefulPartitionedCallinputslstm_28_234362lstm_28_234364lstm_28_234366*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_28_layer_call_and_return_conditional_losses_234361а
lstm_29/StatefulPartitionedCallStatefulPartitionedCall(lstm_28/StatefulPartitionedCall:output:0lstm_29_234625lstm_29_234627lstm_29_234629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_29_layer_call_and_return_conditional_losses_234624Т
 dense_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0dense_14_234643dense_14_234645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_234642x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_14/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ы}
ж	
while_body_237693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_29_split_readvariableop_resource_0:
ААC
4while_lstm_cell_29_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_29_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_29_split_readvariableop_resource:
ААA
2while_lstm_cell_29_split_1_readvariableop_resource:	А=
*while_lstm_cell_29_readvariableop_resource:	@АИв!while/lstm_cell_29/ReadVariableOpв#while/lstm_cell_29/ReadVariableOp_1в#while/lstm_cell_29/ReadVariableOp_2в#while/lstm_cell_29/ReadVariableOp_3в'while/lstm_cell_29/split/ReadVariableOpв)while/lstm_cell_29/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_29/split/ReadVariableOpReadVariableOp2while_lstm_cell_29_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0/while/lstm_cell_29/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_29/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_29/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_29/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_29/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_29_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_29/split_1Split-while/lstm_cell_29/split_1/split_dim:output:01while/lstm_cell_29/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_29/BiasAddBiasAdd#while/lstm_cell_29/MatMul:product:0#while/lstm_cell_29/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_1BiasAdd%while/lstm_cell_29/MatMul_1:product:0#while/lstm_cell_29/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_2BiasAdd%while/lstm_cell_29/MatMul_2:product:0#while/lstm_cell_29/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_29/BiasAdd_3BiasAdd%while/lstm_cell_29/MatMul_3:product:0#while/lstm_cell_29/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_29/ReadVariableOpReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_29/strided_sliceStridedSlice)while/lstm_cell_29/ReadVariableOp:value:0/while/lstm_cell_29/strided_slice/stack:output:01while/lstm_cell_29/strided_slice/stack_1:output:01while/lstm_cell_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_29/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_29/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_29/addAddV2#while/lstm_cell_29/BiasAdd:output:0%while/lstm_cell_29/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_29/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_29/MulMulwhile/lstm_cell_29/add:z:0!while/lstm_cell_29/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_29/Add_1AddV2while/lstm_cell_29/Mul:z:0#while/lstm_cell_29/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_29/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_29/clip_by_value/MinimumMinimumwhile/lstm_cell_29/Add_1:z:03while/lstm_cell_29/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_29/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_29/clip_by_valueMaximum,while/lstm_cell_29/clip_by_value/Minimum:z:0+while/lstm_cell_29/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_1ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_1StridedSlice+while/lstm_cell_29/ReadVariableOp_1:value:01while/lstm_cell_29/strided_slice_1/stack:output:03while/lstm_cell_29/strided_slice_1/stack_1:output:03while/lstm_cell_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_2AddV2%while/lstm_cell_29/BiasAdd_1:output:0%while/lstm_cell_29/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_1Mulwhile/lstm_cell_29/add_2:z:0#while/lstm_cell_29/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_3AddV2while/lstm_cell_29/Mul_1:z:0#while/lstm_cell_29/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_1/MinimumMinimumwhile/lstm_cell_29/Add_3:z:05while/lstm_cell_29/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_1Maximum.while/lstm_cell_29/clip_by_value_1/Minimum:z:0-while/lstm_cell_29/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_29/mul_2Mul&while/lstm_cell_29/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_2ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_2StridedSlice+while/lstm_cell_29/ReadVariableOp_2:value:01while/lstm_cell_29/strided_slice_2/stack:output:03while/lstm_cell_29/strided_slice_2/stack_1:output:03while/lstm_cell_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_4AddV2%while/lstm_cell_29/BiasAdd_2:output:0%while/lstm_cell_29/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_29/ReluReluwhile/lstm_cell_29/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_29/mul_3Mul$while/lstm_cell_29/clip_by_value:z:0%while/lstm_cell_29/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_29/add_5AddV2while/lstm_cell_29/mul_2:z:0while/lstm_cell_29/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_29/ReadVariableOp_3ReadVariableOp,while_lstm_cell_29_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_29/strided_slice_3StridedSlice+while/lstm_cell_29/ReadVariableOp_3:value:01while/lstm_cell_29/strided_slice_3/stack:output:03while/lstm_cell_29/strided_slice_3/stack_1:output:03while/lstm_cell_29/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_29/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_29/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_29/add_6AddV2%while/lstm_cell_29/BiasAdd_3:output:0%while/lstm_cell_29/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_29/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_29/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_29/Mul_4Mulwhile/lstm_cell_29/add_6:z:0#while/lstm_cell_29/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_29/Add_7AddV2while/lstm_cell_29/Mul_4:z:0#while/lstm_cell_29/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_29/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_29/clip_by_value_2/MinimumMinimumwhile/lstm_cell_29/Add_7:z:05while/lstm_cell_29/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_29/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_29/clip_by_value_2Maximum.while/lstm_cell_29/clip_by_value_2/Minimum:z:0-while/lstm_cell_29/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_29/mul_5Mul&while/lstm_cell_29/clip_by_value_2:z:0'while/lstm_cell_29/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_29/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_29/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_29/ReadVariableOp$^while/lstm_cell_29/ReadVariableOp_1$^while/lstm_cell_29/ReadVariableOp_2$^while/lstm_cell_29/ReadVariableOp_3(^while/lstm_cell_29/split/ReadVariableOp*^while/lstm_cell_29/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_29_readvariableop_resource,while_lstm_cell_29_readvariableop_resource_0"j
2while_lstm_cell_29_split_1_readvariableop_resource4while_lstm_cell_29_split_1_readvariableop_resource_0"f
0while_lstm_cell_29_split_readvariableop_resource2while_lstm_cell_29_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_29/ReadVariableOp_1#while/lstm_cell_29/ReadVariableOp_12J
#while/lstm_cell_29/ReadVariableOp_2#while/lstm_cell_29/ReadVariableOp_22J
#while/lstm_cell_29/ReadVariableOp_3#while/lstm_cell_29/ReadVariableOp_32F
!while/lstm_cell_29/ReadVariableOp!while/lstm_cell_29/ReadVariableOp2R
'while/lstm_cell_29/split/ReadVariableOp'while/lstm_cell_29/split/ReadVariableOp2V
)while/lstm_cell_29/split_1/ReadVariableOp)while/lstm_cell_29/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultз
K
lstm_28_input:
serving_default_lstm_28_input:0         <
dense_140
StatefulPartitionedCall:0         tensorflow/serving/predict:╞ы
█
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
_default_save_signature
*	&call_and_return_all_conditional_losses

__call__
	optimizer

signatures"
_tf_keras_sequential
├
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__
cell

state_spec"
_tf_keras_rnn_layer
├
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__
cell

state_spec"
_tf_keras_rnn_layer
╗
	variables
trainable_variables
regularization_losses
 	keras_api
*!&call_and_return_all_conditional_losses
"__call__

#kernel
$bias"
_tf_keras_layer
X
%0
&1
'2
(3
)4
*5
#6
$7"
trackable_list_wrapper
X
%0
&1
'2
(3
)4
*5
#6
$7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
+layer_metrics

,layers
	variables
-non_trainable_variables
.layer_regularization_losses
trainable_variables
regularization_losses
/metrics

__call__
_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Й
0trace_02ь
!__inference__wrapped_model_233175╞
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *0в-
+К(
lstm_28_input         z0trace_0
╧
1trace_0
2trace_1
3trace_2
4trace_32ф
I__inference_sequential_14_layer_call_and_return_conditional_losses_235951
I__inference_sequential_14_layer_call_and_return_conditional_losses_236465
I__inference_sequential_14_layer_call_and_return_conditional_losses_235343
I__inference_sequential_14_layer_call_and_return_conditional_losses_235366╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z1trace_0z2trace_1z3trace_2z4trace_3
у
5trace_0
6trace_1
7trace_2
8trace_32°
.__inference_sequential_14_layer_call_fn_234668
.__inference_sequential_14_layer_call_fn_235416
.__inference_sequential_14_layer_call_fn_235437
.__inference_sequential_14_layer_call_fn_235320╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z5trace_0z6trace_1z7trace_2z8trace_3
Б

9beta_1

:beta_2
	;decay
<learning_rate
=iter#mЗ$mИ%mЙ&mК'mЛ(mМ)mН*mО#vП$vР%vС&vТ'vУ(vФ)vХ*vЦ"
tf_deprecated_optimizer
,
>serving_default"
signature_map
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣
?layer_metrics

@states

Alayers
	variables
Bnon_trainable_variables
Clayer_regularization_losses
trainable_variables
regularization_losses
Dmetrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╠
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32с
C__inference_lstm_28_layer_call_and_return_conditional_losses_236765
C__inference_lstm_28_layer_call_and_return_conditional_losses_237021
C__inference_lstm_28_layer_call_and_return_conditional_losses_237277
C__inference_lstm_28_layer_call_and_return_conditional_losses_237533╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
р
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32ї
(__inference_lstm_28_layer_call_fn_236476
(__inference_lstm_28_layer_call_fn_236487
(__inference_lstm_28_layer_call_fn_236498
(__inference_lstm_28_layer_call_fn_236509╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
с
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
*Q&call_and_return_all_conditional_losses
R__call__
S
state_size

%kernel
&recurrent_kernel
'bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣
Tlayer_metrics

Ustates

Vlayers
	variables
Wnon_trainable_variables
Xlayer_regularization_losses
trainable_variables
regularization_losses
Ymetrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╠
Ztrace_0
[trace_1
\trace_2
]trace_32с
C__inference_lstm_29_layer_call_and_return_conditional_losses_237833
C__inference_lstm_29_layer_call_and_return_conditional_losses_238089
C__inference_lstm_29_layer_call_and_return_conditional_losses_238345
C__inference_lstm_29_layer_call_and_return_conditional_losses_238601╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zZtrace_0z[trace_1z\trace_2z]trace_3
р
^trace_0
_trace_1
`trace_2
atrace_32ї
(__inference_lstm_29_layer_call_fn_237544
(__inference_lstm_29_layer_call_fn_237555
(__inference_lstm_29_layer_call_fn_237566
(__inference_lstm_29_layer_call_fn_237577╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z^trace_0z_trace_1z`trace_2zatrace_3
с
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
*f&call_and_return_all_conditional_losses
g__call__
h
state_size

(kernel
)recurrent_kernel
*bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
ilayer_metrics

jlayers
knon_trainable_variables
	variables
llayer_regularization_losses
trainable_variables
regularization_losses
mmetrics
"__call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
■
ntrace_02с
D__inference_dense_14_layer_call_and_return_conditional_losses_238620Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zntrace_0
у
otrace_02╞
)__inference_dense_14_layer_call_fn_238610Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zotrace_0
!:@2dense_14/kernel
:2dense_14/bias
.:,	А2lstm_28/lstm_cell_28/kernel
9:7
АА2%lstm_28/lstm_cell_28/recurrent_kernel
(:&А2lstm_28/lstm_cell_28/bias
/:-
АА2lstm_29/lstm_cell_29/kernel
8:6	@А2%lstm_29/lstm_cell_29/recurrent_kernel
(:&А2lstm_29/lstm_cell_29/bias
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
АB¤
!__inference__wrapped_model_233175lstm_28_input"╞
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *0в-
+К(
lstm_28_input         
РBН
I__inference_sequential_14_layer_call_and_return_conditional_losses_235951inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
I__inference_sequential_14_layer_call_and_return_conditional_losses_236465inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
I__inference_sequential_14_layer_call_and_return_conditional_losses_235343lstm_28_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
I__inference_sequential_14_layer_call_and_return_conditional_losses_235366lstm_28_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
.__inference_sequential_14_layer_call_fn_234668lstm_28_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
.__inference_sequential_14_layer_call_fn_235416inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
.__inference_sequential_14_layer_call_fn_235437inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
.__inference_sequential_14_layer_call_fn_235320lstm_28_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
╤B╬
$__inference_signature_wrapper_235395lstm_28_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
бBЮ
C__inference_lstm_28_layer_call_and_return_conditional_losses_236765inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
C__inference_lstm_28_layer_call_and_return_conditional_losses_237021inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
C__inference_lstm_28_layer_call_and_return_conditional_losses_237277inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
C__inference_lstm_28_layer_call_and_return_conditional_losses_237533inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
(__inference_lstm_28_layer_call_fn_236476inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
(__inference_lstm_28_layer_call_fn_236487inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
(__inference_lstm_28_layer_call_fn_236498inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
(__inference_lstm_28_layer_call_fn_236509inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
qlayer_metrics

rlayers
snon_trainable_variables
M	variables
tlayer_regularization_losses
Ntrainable_variables
Oregularization_losses
umetrics
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
Б
vtrace_0
wtrace_12╩
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238743
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238832│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zvtrace_0zwtrace_1
╦
xtrace_0
ytrace_12Ф
-__inference_lstm_cell_28_layer_call_fn_238637
-__inference_lstm_cell_28_layer_call_fn_238654│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zxtrace_0zytrace_1
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
бBЮ
C__inference_lstm_29_layer_call_and_return_conditional_losses_237833inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
C__inference_lstm_29_layer_call_and_return_conditional_losses_238089inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
C__inference_lstm_29_layer_call_and_return_conditional_losses_238345inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
C__inference_lstm_29_layer_call_and_return_conditional_losses_238601inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
(__inference_lstm_29_layer_call_fn_237544inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
(__inference_lstm_29_layer_call_fn_237555inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
(__inference_lstm_29_layer_call_fn_237566inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
(__inference_lstm_29_layer_call_fn_237577inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
zlayer_metrics

{layers
|non_trainable_variables
b	variables
}layer_regularization_losses
ctrainable_variables
dregularization_losses
~metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Г
trace_0
Аtrace_12╩
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_238955
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_239044│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0zАtrace_1
╧
Бtrace_0
Вtrace_12Ф
-__inference_lstm_cell_29_layer_call_fn_238849
-__inference_lstm_cell_29_layer_call_fn_238866│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zБtrace_0zВtrace_1
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
юBы
D__inference_dense_14_layer_call_and_return_conditional_losses_238620inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙B╨
)__inference_dense_14_layer_call_fn_238610inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
Г	variables
Д	keras_api

Еtotal

Жcount"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
бBЮ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238743inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238832inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
-__inference_lstm_cell_28_layer_call_fn_238637inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
-__inference_lstm_cell_28_layer_call_fn_238654inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
бBЮ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_238955inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_239044inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
-__inference_lstm_cell_29_layer_call_fn_238849inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
-__inference_lstm_cell_29_layer_call_fn_238866inputsstates_0states_1"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Е0
Ж1"
trackable_list_wrapper
.
Г	variables"
_generic_user_object
:  (2total
:  (2count
&:$@2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
3:1	А2"Adam/lstm_28/lstm_cell_28/kernel/m
>:<
АА2,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m
-:+А2 Adam/lstm_28/lstm_cell_28/bias/m
4:2
АА2"Adam/lstm_29/lstm_cell_29/kernel/m
=:;	@А2,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m
-:+А2 Adam/lstm_29/lstm_cell_29/bias/m
&:$@2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
3:1	А2"Adam/lstm_28/lstm_cell_28/kernel/v
>:<
АА2,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v
-:+А2 Adam/lstm_28/lstm_cell_28/bias/v
4:2
АА2"Adam/lstm_29/lstm_cell_29/kernel/v
=:;	@А2,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v
-:+А2 Adam/lstm_29/lstm_cell_29/bias/vа
!__inference__wrapped_model_233175{%'&(*)#$:в7
0в-
+К(
lstm_28_input         
к "3к0
.
dense_14"К
dense_14         л
D__inference_dense_14_layer_call_and_return_conditional_losses_238620c#$/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         
Ъ Е
)__inference_dense_14_layer_call_fn_238610X#$/в,
%в"
 К
inputs         @
к "!К
unknown         ┌
C__inference_lstm_28_layer_call_and_return_conditional_losses_236765Т%'&OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p 

 
к ":в7
0К-
tensor_0                  А
Ъ ┌
C__inference_lstm_28_layer_call_and_return_conditional_losses_237021Т%'&OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p

 
к ":в7
0К-
tensor_0                  А
Ъ └
C__inference_lstm_28_layer_call_and_return_conditional_losses_237277y%'&?в<
5в2
$К!
inputs         

 
p 

 
к "1в.
'К$
tensor_0         А
Ъ └
C__inference_lstm_28_layer_call_and_return_conditional_losses_237533y%'&?в<
5в2
$К!
inputs         

 
p

 
к "1в.
'К$
tensor_0         А
Ъ ┤
(__inference_lstm_28_layer_call_fn_236476З%'&OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p 

 
к "/К,
unknown                  А┤
(__inference_lstm_28_layer_call_fn_236487З%'&OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p

 
к "/К,
unknown                  АЪ
(__inference_lstm_28_layer_call_fn_236498n%'&?в<
5в2
$К!
inputs         

 
p 

 
к "&К#
unknown         АЪ
(__inference_lstm_28_layer_call_fn_236509n%'&?в<
5в2
$К!
inputs         

 
p

 
к "&К#
unknown         А═
C__inference_lstm_29_layer_call_and_return_conditional_losses_237833Е(*)PвM
FвC
5Ъ2
0К-
inputs_0                  А

 
p 

 
к ",в)
"К
tensor_0         @
Ъ ═
C__inference_lstm_29_layer_call_and_return_conditional_losses_238089Е(*)PвM
FвC
5Ъ2
0К-
inputs_0                  А

 
p

 
к ",в)
"К
tensor_0         @
Ъ ╝
C__inference_lstm_29_layer_call_and_return_conditional_losses_238345u(*)@в=
6в3
%К"
inputs         А

 
p 

 
к ",в)
"К
tensor_0         @
Ъ ╝
C__inference_lstm_29_layer_call_and_return_conditional_losses_238601u(*)@в=
6в3
%К"
inputs         А

 
p

 
к ",в)
"К
tensor_0         @
Ъ ж
(__inference_lstm_29_layer_call_fn_237544z(*)PвM
FвC
5Ъ2
0К-
inputs_0                  А

 
p 

 
к "!К
unknown         @ж
(__inference_lstm_29_layer_call_fn_237555z(*)PвM
FвC
5Ъ2
0К-
inputs_0                  А

 
p

 
к "!К
unknown         @Ц
(__inference_lstm_29_layer_call_fn_237566j(*)@в=
6в3
%К"
inputs         А

 
p 

 
к "!К
unknown         @Ц
(__inference_lstm_29_layer_call_fn_237577j(*)@в=
6в3
%К"
inputs         А

 
p

 
к "!К
unknown         @ч
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238743Ъ%'&Вв
xвu
 К
inputs         
MвJ
#К 
states_0         А
#К 
states_1         А
p 
к "НвЙ
Бв~
%К"

tensor_0_0         А
UЪR
'К$
tensor_0_1_0         А
'К$
tensor_0_1_1         А
Ъ ч
H__inference_lstm_cell_28_layer_call_and_return_conditional_losses_238832Ъ%'&Вв
xвu
 К
inputs         
MвJ
#К 
states_0         А
#К 
states_1         А
p
к "НвЙ
Бв~
%К"

tensor_0_0         А
UЪR
'К$
tensor_0_1_0         А
'К$
tensor_0_1_1         А
Ъ ╣
-__inference_lstm_cell_28_layer_call_fn_238637З%'&Вв
xвu
 К
inputs         
MвJ
#К 
states_0         А
#К 
states_1         А
p 
к "{вx
#К 
tensor_0         А
QЪN
%К"

tensor_1_0         А
%К"

tensor_1_1         А╣
-__inference_lstm_cell_28_layer_call_fn_238654З%'&Вв
xвu
 К
inputs         
MвJ
#К 
states_0         А
#К 
states_1         А
p
к "{вx
#К 
tensor_0         А
QЪN
%К"

tensor_1_0         А
%К"

tensor_1_1         Ат
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_238955Х(*)Бв~
wвt
!К
inputs         А
KвH
"К
states_0         @
"К
states_1         @
p 
к "ЙвЕ
~в{
$К!

tensor_0_0         @
SЪP
&К#
tensor_0_1_0         @
&К#
tensor_0_1_1         @
Ъ т
H__inference_lstm_cell_29_layer_call_and_return_conditional_losses_239044Х(*)Бв~
wвt
!К
inputs         А
KвH
"К
states_0         @
"К
states_1         @
p
к "ЙвЕ
~в{
$К!

tensor_0_0         @
SЪP
&К#
tensor_0_1_0         @
&К#
tensor_0_1_1         @
Ъ ╡
-__inference_lstm_cell_29_layer_call_fn_238849Г(*)Бв~
wвt
!К
inputs         А
KвH
"К
states_0         @
"К
states_1         @
p 
к "xвu
"К
tensor_0         @
OЪL
$К!

tensor_1_0         @
$К!

tensor_1_1         @╡
-__inference_lstm_cell_29_layer_call_fn_238866Г(*)Бв~
wвt
!К
inputs         А
KвH
"К
states_0         @
"К
states_1         @
p
к "xвu
"К
tensor_0         @
OЪL
$К!

tensor_1_0         @
$К!

tensor_1_1         @╔
I__inference_sequential_14_layer_call_and_return_conditional_losses_235343|%'&(*)#$Bв?
8в5
+К(
lstm_28_input         
p 

 
к ",в)
"К
tensor_0         
Ъ ╔
I__inference_sequential_14_layer_call_and_return_conditional_losses_235366|%'&(*)#$Bв?
8в5
+К(
lstm_28_input         
p

 
к ",в)
"К
tensor_0         
Ъ ┬
I__inference_sequential_14_layer_call_and_return_conditional_losses_235951u%'&(*)#$;в8
1в.
$К!
inputs         
p 

 
к ",в)
"К
tensor_0         
Ъ ┬
I__inference_sequential_14_layer_call_and_return_conditional_losses_236465u%'&(*)#$;в8
1в.
$К!
inputs         
p

 
к ",в)
"К
tensor_0         
Ъ г
.__inference_sequential_14_layer_call_fn_234668q%'&(*)#$Bв?
8в5
+К(
lstm_28_input         
p 

 
к "!К
unknown         г
.__inference_sequential_14_layer_call_fn_235320q%'&(*)#$Bв?
8в5
+К(
lstm_28_input         
p

 
к "!К
unknown         Ь
.__inference_sequential_14_layer_call_fn_235416j%'&(*)#$;в8
1в.
$К!
inputs         
p 

 
к "!К
unknown         Ь
.__inference_sequential_14_layer_call_fn_235437j%'&(*)#$;в8
1в.
$К!
inputs         
p

 
к "!К
unknown         ╡
$__inference_signature_wrapper_235395М%'&(*)#$KвH
в 
Aк>
<
lstm_28_input+К(
lstm_28_input         "3к0
.
dense_14"К
dense_14         