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
 Adam/lstm_47/lstm_cell_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_47/lstm_cell_47/bias/v
Т
4Adam/lstm_47/lstm_cell_47/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_47/lstm_cell_47/bias/v*
_output_shapes	
:А*
dtype0
╡
,Adam/lstm_47/lstm_cell_47/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*=
shared_name.,Adam/lstm_47/lstm_cell_47/recurrent_kernel/v
о
@Adam/lstm_47/lstm_cell_47/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_47/lstm_cell_47/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
в
"Adam/lstm_47/lstm_cell_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*3
shared_name$"Adam/lstm_47/lstm_cell_47/kernel/v
Ы
6Adam/lstm_47/lstm_cell_47/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_47/lstm_cell_47/kernel/v* 
_output_shapes
:
АА*
dtype0
Щ
 Adam/lstm_46/lstm_cell_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_46/lstm_cell_46/bias/v
Т
4Adam/lstm_46/lstm_cell_46/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_46/lstm_cell_46/bias/v*
_output_shapes	
:А*
dtype0
╢
,Adam/lstm_46/lstm_cell_46/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*=
shared_name.,Adam/lstm_46/lstm_cell_46/recurrent_kernel/v
п
@Adam/lstm_46/lstm_cell_46/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_46/lstm_cell_46/recurrent_kernel/v* 
_output_shapes
:
АА*
dtype0
б
"Adam/lstm_46/lstm_cell_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_46/lstm_cell_46/kernel/v
Ъ
6Adam/lstm_46/lstm_cell_46/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_46/lstm_cell_46/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/v
Б
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:@*
dtype0
Щ
 Adam/lstm_47/lstm_cell_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_47/lstm_cell_47/bias/m
Т
4Adam/lstm_47/lstm_cell_47/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_47/lstm_cell_47/bias/m*
_output_shapes	
:А*
dtype0
╡
,Adam/lstm_47/lstm_cell_47/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*=
shared_name.,Adam/lstm_47/lstm_cell_47/recurrent_kernel/m
о
@Adam/lstm_47/lstm_cell_47/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_47/lstm_cell_47/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
в
"Adam/lstm_47/lstm_cell_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*3
shared_name$"Adam/lstm_47/lstm_cell_47/kernel/m
Ы
6Adam/lstm_47/lstm_cell_47/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_47/lstm_cell_47/kernel/m* 
_output_shapes
:
АА*
dtype0
Щ
 Adam/lstm_46/lstm_cell_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_46/lstm_cell_46/bias/m
Т
4Adam/lstm_46/lstm_cell_46/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_46/lstm_cell_46/bias/m*
_output_shapes	
:А*
dtype0
╢
,Adam/lstm_46/lstm_cell_46/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*=
shared_name.,Adam/lstm_46/lstm_cell_46/recurrent_kernel/m
п
@Adam/lstm_46/lstm_cell_46/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_46/lstm_cell_46/recurrent_kernel/m* 
_output_shapes
:
АА*
dtype0
б
"Adam/lstm_46/lstm_cell_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/lstm_46/lstm_cell_46/kernel/m
Ъ
6Adam/lstm_46/lstm_cell_46/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_46/lstm_cell_46/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/m
Б
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
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
lstm_47/lstm_cell_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_47/lstm_cell_47/bias
Д
-lstm_47/lstm_cell_47/bias/Read/ReadVariableOpReadVariableOplstm_47/lstm_cell_47/bias*
_output_shapes	
:А*
dtype0
з
%lstm_47/lstm_cell_47/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*6
shared_name'%lstm_47/lstm_cell_47/recurrent_kernel
а
9lstm_47/lstm_cell_47/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_47/lstm_cell_47/recurrent_kernel*
_output_shapes
:	@А*
dtype0
Ф
lstm_47/lstm_cell_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*,
shared_namelstm_47/lstm_cell_47/kernel
Н
/lstm_47/lstm_cell_47/kernel/Read/ReadVariableOpReadVariableOplstm_47/lstm_cell_47/kernel* 
_output_shapes
:
АА*
dtype0
Л
lstm_46/lstm_cell_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_46/lstm_cell_46/bias
Д
-lstm_46/lstm_cell_46/bias/Read/ReadVariableOpReadVariableOplstm_46/lstm_cell_46/bias*
_output_shapes	
:А*
dtype0
и
%lstm_46/lstm_cell_46/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*6
shared_name'%lstm_46/lstm_cell_46/recurrent_kernel
б
9lstm_46/lstm_cell_46/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_46/lstm_cell_46/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
У
lstm_46/lstm_cell_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_namelstm_46/lstm_cell_46/kernel
М
/lstm_46/lstm_cell_46/kernel/Read/ReadVariableOpReadVariableOplstm_46/lstm_cell_46/kernel*
_output_shapes
:	А*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:@*
dtype0
И
serving_default_lstm_46_inputPlaceholder*+
_output_shapes
:         
*
dtype0* 
shape:         

ж
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_46_inputlstm_46/lstm_cell_46/kernellstm_46/lstm_cell_46/bias%lstm_46/lstm_cell_46/recurrent_kernellstm_47/lstm_cell_47/kernellstm_47/lstm_cell_47/bias%lstm_47/lstm_cell_47/recurrent_kerneldense_23/kerneldense_23/bias*
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
$__inference_signature_wrapper_379523

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
VARIABLE_VALUEdense_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_46/lstm_cell_46/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_46/lstm_cell_46/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_46/lstm_cell_46/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_47/lstm_cell_47/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_47/lstm_cell_47/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_47/lstm_cell_47/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_46/lstm_cell_46/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_46/lstm_cell_46/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_46/lstm_cell_46/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_47/lstm_cell_47/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_47/lstm_cell_47/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_47/lstm_cell_47/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_46/lstm_cell_46/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_46/lstm_cell_46/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_46/lstm_cell_46/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_47/lstm_cell_47/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/lstm_47/lstm_cell_47/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_47/lstm_cell_47/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Г	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/biaslstm_46/lstm_cell_46/kernel%lstm_46/lstm_cell_46/recurrent_kernellstm_46/lstm_cell_46/biaslstm_47/lstm_cell_47/kernel%lstm_47/lstm_cell_47/recurrent_kernellstm_47/lstm_cell_47/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense_23/kernel/mAdam/dense_23/bias/m"Adam/lstm_46/lstm_cell_46/kernel/m,Adam/lstm_46/lstm_cell_46/recurrent_kernel/m Adam/lstm_46/lstm_cell_46/bias/m"Adam/lstm_47/lstm_cell_47/kernel/m,Adam/lstm_47/lstm_cell_47/recurrent_kernel/m Adam/lstm_47/lstm_cell_47/bias/mAdam/dense_23/kernel/vAdam/dense_23/bias/v"Adam/lstm_46/lstm_cell_46/kernel/v,Adam/lstm_46/lstm_cell_46/recurrent_kernel/v Adam/lstm_46/lstm_cell_46/bias/v"Adam/lstm_47/lstm_cell_47/kernel/v,Adam/lstm_47/lstm_cell_47/recurrent_kernel/v Adam/lstm_47/lstm_cell_47/bias/vConst*,
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
__inference__traced_save_383381
■
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/biaslstm_46/lstm_cell_46/kernel%lstm_46/lstm_cell_46/recurrent_kernellstm_46/lstm_cell_46/biaslstm_47/lstm_cell_47/kernel%lstm_47/lstm_cell_47/recurrent_kernellstm_47/lstm_cell_47/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense_23/kernel/mAdam/dense_23/bias/m"Adam/lstm_46/lstm_cell_46/kernel/m,Adam/lstm_46/lstm_cell_46/recurrent_kernel/m Adam/lstm_46/lstm_cell_46/bias/m"Adam/lstm_47/lstm_cell_47/kernel/m,Adam/lstm_47/lstm_cell_47/recurrent_kernel/m Adam/lstm_47/lstm_cell_47/bias/mAdam/dense_23/kernel/vAdam/dense_23/bias/v"Adam/lstm_46/lstm_cell_46/kernel/v,Adam/lstm_46/lstm_cell_46/recurrent_kernel/v Adam/lstm_46/lstm_cell_46/bias/v"Adam/lstm_47/lstm_cell_47/kernel/v,Adam/lstm_47/lstm_cell_47/recurrent_kernel/v Adam/lstm_47/lstm_cell_47/bias/v*+
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
"__inference__traced_restore_383484■№5
иИ
ш
C__inference_lstm_47_layer_call_and_return_conditional_losses_382729

inputs>
*lstm_cell_47_split_readvariableop_resource:
АА;
,lstm_cell_47_split_1_readvariableop_resource:	А7
$lstm_cell_47_readvariableop_resource:	@А
identityИвlstm_cell_47/ReadVariableOpвlstm_cell_47/ReadVariableOp_1вlstm_cell_47/ReadVariableOp_2вlstm_cell_47/ReadVariableOp_3в!lstm_cell_47/split/ReadVariableOpв#lstm_cell_47/split_1/ReadVariableOpвwhileI
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
:
         АR
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
lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_47/split/ReadVariableOpReadVariableOp*lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_47/splitSplit%lstm_cell_47/split/split_dim:output:0)lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_47/MatMulMatMulstrided_slice_2:output:0lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_47/split_1/ReadVariableOpReadVariableOp,lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_47/split_1Split'lstm_cell_47/split_1/split_dim:output:0+lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_47/BiasAddBiasAddlstm_cell_47/MatMul:product:0lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_1BiasAddlstm_cell_47/MatMul_1:product:0lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_2BiasAddlstm_cell_47/MatMul_2:product:0lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_3BiasAddlstm_cell_47/MatMul_3:product:0lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_47/ReadVariableOpReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_47/strided_sliceStridedSlice#lstm_cell_47/ReadVariableOp:value:0)lstm_cell_47/strided_slice/stack:output:0+lstm_cell_47/strided_slice/stack_1:output:0+lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_47/MatMul_4MatMulzeros:output:0#lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_47/addAddV2lstm_cell_47/BiasAdd:output:0lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_47/MulMullstm_cell_47/add:z:0lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_47/Add_1AddV2lstm_cell_47/Mul:z:0lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_47/clip_by_value/MinimumMinimumlstm_cell_47/Add_1:z:0-lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_47/clip_by_valueMaximum&lstm_cell_47/clip_by_value/Minimum:z:0%lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_1ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_1StridedSlice%lstm_cell_47/ReadVariableOp_1:value:0+lstm_cell_47/strided_slice_1/stack:output:0-lstm_cell_47/strided_slice_1/stack_1:output:0-lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_5MatMulzeros:output:0%lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_2AddV2lstm_cell_47/BiasAdd_1:output:0lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_1Mullstm_cell_47/add_2:z:0lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_3AddV2lstm_cell_47/Mul_1:z:0lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_cell_47/Add_3:z:0/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_1Maximum(lstm_cell_47/clip_by_value_1/Minimum:z:0'lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_47/mul_2Mul lstm_cell_47/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_2ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_2StridedSlice%lstm_cell_47/ReadVariableOp_2:value:0+lstm_cell_47/strided_slice_2/stack:output:0-lstm_cell_47/strided_slice_2/stack_1:output:0-lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_6MatMulzeros:output:0%lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_4AddV2lstm_cell_47/BiasAdd_2:output:0lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/ReluRelulstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_47/mul_3Mullstm_cell_47/clip_by_value:z:0lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_47/add_5AddV2lstm_cell_47/mul_2:z:0lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_3ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_3StridedSlice%lstm_cell_47/ReadVariableOp_3:value:0+lstm_cell_47/strided_slice_3/stack:output:0-lstm_cell_47/strided_slice_3/stack_1:output:0-lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_7MatMulzeros:output:0%lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_6AddV2lstm_cell_47/BiasAdd_3:output:0lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_4Mullstm_cell_47/add_6:z:0lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_7AddV2lstm_cell_47/Mul_4:z:0lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_cell_47/Add_7:z:0/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_2Maximum(lstm_cell_47/clip_by_value_2/Minimum:z:0'lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_47/Relu_1Relulstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_47/mul_5Mul lstm_cell_47/clip_by_value_2:z:0!lstm_cell_47/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_47_split_readvariableop_resource,lstm_cell_47_split_1_readvariableop_resource$lstm_cell_47_readvariableop_resource*
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
while_body_382589*
condR
while_cond_382588*K
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
:
         @*
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
:         
@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_47/ReadVariableOp^lstm_cell_47/ReadVariableOp_1^lstm_cell_47/ReadVariableOp_2^lstm_cell_47/ReadVariableOp_3"^lstm_cell_47/split/ReadVariableOp$^lstm_cell_47/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         
А: : : 2>
lstm_cell_47/ReadVariableOp_1lstm_cell_47/ReadVariableOp_12>
lstm_cell_47/ReadVariableOp_2lstm_cell_47/ReadVariableOp_22>
lstm_cell_47/ReadVariableOp_3lstm_cell_47/ReadVariableOp_32:
lstm_cell_47/ReadVariableOplstm_cell_47/ReadVariableOp2F
!lstm_cell_47/split/ReadVariableOp!lstm_cell_47/split/ReadVariableOp2J
#lstm_cell_47/split_1/ReadVariableOp#lstm_cell_47/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         
А
 
_user_specified_nameinputs
Ч	
├
while_cond_377902
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_377902___redundant_placeholder04
0while_while_cond_377902___redundant_placeholder14
0while_while_cond_377902___redundant_placeholder24
0while_while_cond_377902___redundant_placeholder3
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
Е
ф
I__inference_sequential_23_layer_call_and_return_conditional_losses_379471
lstm_46_input!
lstm_46_379451:	А
lstm_46_379453:	А"
lstm_46_379455:
АА"
lstm_47_379458:
АА
lstm_47_379460:	А!
lstm_47_379462:	@А!
dense_23_379465:@
dense_23_379467:
identityИв dense_23/StatefulPartitionedCallвlstm_46/StatefulPartitionedCallвlstm_47/StatefulPartitionedCallК
lstm_46/StatefulPartitionedCallStatefulPartitionedCalllstm_46_inputlstm_46_379451lstm_46_379453lstm_46_379455*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         
А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_46_layer_call_and_return_conditional_losses_378489а
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_379458lstm_47_379460lstm_47_379462*
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_378752Т
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_23_379465dense_23_379467*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_378770x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_23/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_46_input
И
у
lstm_47_while_cond_380446,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3.
*lstm_47_while_less_lstm_47_strided_slice_1D
@lstm_47_while_lstm_47_while_cond_380446___redundant_placeholder0D
@lstm_47_while_lstm_47_while_cond_380446___redundant_placeholder1D
@lstm_47_while_lstm_47_while_cond_380446___redundant_placeholder2D
@lstm_47_while_lstm_47_while_cond_380446___redundant_placeholder3
lstm_47_while_identity
В
lstm_47/while/LessLesslstm_47_while_placeholder*lstm_47_while_less_lstm_47_strided_slice_1*
T0*
_output_shapes
: [
lstm_47/while/IdentityIdentitylstm_47/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_47_while_identitylstm_47/while/Identity:output:0*(
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
_user_specified_name" lstm_47/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_47/while/loop_counter
б~
ж	
while_body_380753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_46_split_readvariableop_resource_0:	АC
4while_lstm_cell_46_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_46_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_46_split_readvariableop_resource:	АA
2while_lstm_cell_46_split_1_readvariableop_resource:	А>
*while_lstm_cell_46_readvariableop_resource:
ААИв!while/lstm_cell_46/ReadVariableOpв#while/lstm_cell_46/ReadVariableOp_1в#while/lstm_cell_46/ReadVariableOp_2в#while/lstm_cell_46/ReadVariableOp_3в'while/lstm_cell_46/split/ReadVariableOpв)while/lstm_cell_46/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_46/split/ReadVariableOpReadVariableOp2while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_46/splitSplit+while/lstm_cell_46/split/split_dim:output:0/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_46/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_46/split_1Split-while/lstm_cell_46/split_1/split_dim:output:01while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_46/BiasAddBiasAdd#while/lstm_cell_46/MatMul:product:0#while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_1BiasAdd%while/lstm_cell_46/MatMul_1:product:0#while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_2BiasAdd%while/lstm_cell_46/MatMul_2:product:0#while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_3BiasAdd%while/lstm_cell_46/MatMul_3:product:0#while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_46/ReadVariableOpReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_46/strided_sliceStridedSlice)while/lstm_cell_46/ReadVariableOp:value:0/while/lstm_cell_46/strided_slice/stack:output:01while/lstm_cell_46/strided_slice/stack_1:output:01while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_46/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_46/addAddV2#while/lstm_cell_46/BiasAdd:output:0%while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_46/MulMulwhile/lstm_cell_46/add:z:0!while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_46/Add_1AddV2while/lstm_cell_46/Mul:z:0#while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_46/clip_by_value/MinimumMinimumwhile/lstm_cell_46/Add_1:z:03while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_46/clip_by_valueMaximum,while/lstm_cell_46/clip_by_value/Minimum:z:0+while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_1ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_1StridedSlice+while/lstm_cell_46/ReadVariableOp_1:value:01while/lstm_cell_46/strided_slice_1/stack:output:03while/lstm_cell_46/strided_slice_1/stack_1:output:03while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_2AddV2%while/lstm_cell_46/BiasAdd_1:output:0%while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_1Mulwhile/lstm_cell_46/add_2:z:0#while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_3AddV2while/lstm_cell_46/Mul_1:z:0#while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_1/MinimumMinimumwhile/lstm_cell_46/Add_3:z:05while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_1Maximum.while/lstm_cell_46/clip_by_value_1/Minimum:z:0-while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_46/mul_2Mul&while/lstm_cell_46/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_2ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_2StridedSlice+while/lstm_cell_46/ReadVariableOp_2:value:01while/lstm_cell_46/strided_slice_2/stack:output:03while/lstm_cell_46/strided_slice_2/stack_1:output:03while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_4AddV2%while/lstm_cell_46/BiasAdd_2:output:0%while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_46/ReluReluwhile/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_46/mul_3Mul$while/lstm_cell_46/clip_by_value:z:0%while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_46/add_5AddV2while/lstm_cell_46/mul_2:z:0while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_3ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_3StridedSlice+while/lstm_cell_46/ReadVariableOp_3:value:01while/lstm_cell_46/strided_slice_3/stack:output:03while/lstm_cell_46/strided_slice_3/stack_1:output:03while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_6AddV2%while/lstm_cell_46/BiasAdd_3:output:0%while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_4Mulwhile/lstm_cell_46/add_6:z:0#while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_7AddV2while/lstm_cell_46/Mul_4:z:0#while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_2/MinimumMinimumwhile/lstm_cell_46/Add_7:z:05while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_2Maximum.while/lstm_cell_46/clip_by_value_2/Minimum:z:0-while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_46/Relu_1Reluwhile/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_46/mul_5Mul&while/lstm_cell_46/clip_by_value_2:z:0'while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_46/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_46/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_46/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_46/ReadVariableOp$^while/lstm_cell_46/ReadVariableOp_1$^while/lstm_cell_46/ReadVariableOp_2$^while/lstm_cell_46/ReadVariableOp_3(^while/lstm_cell_46/split/ReadVariableOp*^while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_46_readvariableop_resource,while_lstm_cell_46_readvariableop_resource_0"j
2while_lstm_cell_46_split_1_readvariableop_resource4while_lstm_cell_46_split_1_readvariableop_resource_0"f
0while_lstm_cell_46_split_readvariableop_resource2while_lstm_cell_46_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_46/ReadVariableOp_1#while/lstm_cell_46/ReadVariableOp_12J
#while/lstm_cell_46/ReadVariableOp_2#while/lstm_cell_46/ReadVariableOp_22J
#while/lstm_cell_46/ReadVariableOp_3#while/lstm_cell_46/ReadVariableOp_32F
!while/lstm_cell_46/ReadVariableOp!while/lstm_cell_46/ReadVariableOp2R
'while/lstm_cell_46/split/ReadVariableOp'while/lstm_cell_46/split/ReadVariableOp2V
)while/lstm_cell_46/split_1/ReadVariableOp)while/lstm_cell_46/split_1/ReadVariableOp:
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
╟	
ї
D__inference_dense_23_layer_call_and_return_conditional_losses_382748

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
█П
╛
lstm_47_while_body_379933,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3+
'lstm_47_while_lstm_47_strided_slice_1_0g
clstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0N
:lstm_47_while_lstm_cell_47_split_readvariableop_resource_0:
ААK
<lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0:	АG
4lstm_47_while_lstm_cell_47_readvariableop_resource_0:	@А
lstm_47_while_identity
lstm_47_while_identity_1
lstm_47_while_identity_2
lstm_47_while_identity_3
lstm_47_while_identity_4
lstm_47_while_identity_5)
%lstm_47_while_lstm_47_strided_slice_1e
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorL
8lstm_47_while_lstm_cell_47_split_readvariableop_resource:
ААI
:lstm_47_while_lstm_cell_47_split_1_readvariableop_resource:	АE
2lstm_47_while_lstm_cell_47_readvariableop_resource:	@АИв)lstm_47/while/lstm_cell_47/ReadVariableOpв+lstm_47/while/lstm_cell_47/ReadVariableOp_1в+lstm_47/while/lstm_cell_47/ReadVariableOp_2в+lstm_47/while/lstm_cell_47/ReadVariableOp_3в/lstm_47/while/lstm_cell_47/split/ReadVariableOpв1lstm_47/while/lstm_cell_47/split_1/ReadVariableOpР
?lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╧
1lstm_47/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0lstm_47_while_placeholderHlstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0l
*lstm_47/while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :м
/lstm_47/while/lstm_cell_47/split/ReadVariableOpReadVariableOp:lstm_47_while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0є
 lstm_47/while/lstm_cell_47/splitSplit3lstm_47/while/lstm_cell_47/split/split_dim:output:07lstm_47/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split┬
!lstm_47/while/lstm_cell_47/MatMulMatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @─
#lstm_47/while/lstm_cell_47/MatMul_1MatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @─
#lstm_47/while/lstm_cell_47/MatMul_2MatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @─
#lstm_47/while/lstm_cell_47/MatMul_3MatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @n
,lstm_47/while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_47/while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp<lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0х
"lstm_47/while/lstm_cell_47/split_1Split5lstm_47/while/lstm_cell_47/split_1/split_dim:output:09lstm_47/while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split╣
"lstm_47/while/lstm_cell_47/BiasAddBiasAdd+lstm_47/while/lstm_cell_47/MatMul:product:0+lstm_47/while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @╜
$lstm_47/while/lstm_cell_47/BiasAdd_1BiasAdd-lstm_47/while/lstm_cell_47/MatMul_1:product:0+lstm_47/while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @╜
$lstm_47/while/lstm_cell_47/BiasAdd_2BiasAdd-lstm_47/while/lstm_cell_47/MatMul_2:product:0+lstm_47/while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @╜
$lstm_47/while/lstm_cell_47/BiasAdd_3BiasAdd-lstm_47/while/lstm_cell_47/MatMul_3:product:0+lstm_47/while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Я
)lstm_47/while/lstm_cell_47/ReadVariableOpReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0
.lstm_47/while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_47/while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Б
0lstm_47/while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Є
(lstm_47/while/lstm_cell_47/strided_sliceStridedSlice1lstm_47/while/lstm_cell_47/ReadVariableOp:value:07lstm_47/while/lstm_cell_47/strided_slice/stack:output:09lstm_47/while/lstm_cell_47/strided_slice/stack_1:output:09lstm_47/while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskп
#lstm_47/while/lstm_cell_47/MatMul_4MatMullstm_47_while_placeholder_21lstm_47/while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @╡
lstm_47/while/lstm_cell_47/addAddV2+lstm_47/while/lstm_cell_47/BiasAdd:output:0-lstm_47/while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @e
 lstm_47/while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_47/while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?ж
lstm_47/while/lstm_cell_47/MulMul"lstm_47/while/lstm_cell_47/add:z:0)lstm_47/while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @м
 lstm_47/while/lstm_cell_47/Add_1AddV2"lstm_47/while/lstm_cell_47/Mul:z:0+lstm_47/while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @w
2lstm_47/while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╨
0lstm_47/while/lstm_cell_47/clip_by_value/MinimumMinimum$lstm_47/while/lstm_cell_47/Add_1:z:0;lstm_47/while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @o
*lstm_47/while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╨
(lstm_47/while/lstm_cell_47/clip_by_valueMaximum4lstm_47/while/lstm_cell_47/clip_by_value/Minimum:z:03lstm_47/while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @б
+lstm_47/while/lstm_cell_47/ReadVariableOp_1ReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_47/while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   Г
2lstm_47/while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_47/while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_47/while/lstm_cell_47/strided_slice_1StridedSlice3lstm_47/while/lstm_cell_47/ReadVariableOp_1:value:09lstm_47/while/lstm_cell_47/strided_slice_1/stack:output:0;lstm_47/while/lstm_cell_47/strided_slice_1/stack_1:output:0;lstm_47/while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_47/while/lstm_cell_47/MatMul_5MatMullstm_47_while_placeholder_23lstm_47/while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @╣
 lstm_47/while/lstm_cell_47/add_2AddV2-lstm_47/while/lstm_cell_47/BiasAdd_1:output:0-lstm_47/while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @g
"lstm_47/while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_47/while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_47/while/lstm_cell_47/Mul_1Mul$lstm_47/while/lstm_cell_47/add_2:z:0+lstm_47/while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @о
 lstm_47/while/lstm_cell_47/Add_3AddV2$lstm_47/while/lstm_cell_47/Mul_1:z:0+lstm_47/while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @y
4lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_47/while/lstm_cell_47/clip_by_value_1/MinimumMinimum$lstm_47/while/lstm_cell_47/Add_3:z:0=lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_47/while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_47/while/lstm_cell_47/clip_by_value_1Maximum6lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum:z:05lstm_47/while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @ж
 lstm_47/while/lstm_cell_47/mul_2Mul.lstm_47/while/lstm_cell_47/clip_by_value_1:z:0lstm_47_while_placeholder_3*
T0*'
_output_shapes
:         @б
+lstm_47/while/lstm_cell_47/ReadVariableOp_2ReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_47/while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_47/while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_47/while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_47/while/lstm_cell_47/strided_slice_2StridedSlice3lstm_47/while/lstm_cell_47/ReadVariableOp_2:value:09lstm_47/while/lstm_cell_47/strided_slice_2/stack:output:0;lstm_47/while/lstm_cell_47/strided_slice_2/stack_1:output:0;lstm_47/while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_47/while/lstm_cell_47/MatMul_6MatMullstm_47_while_placeholder_23lstm_47/while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @╣
 lstm_47/while/lstm_cell_47/add_4AddV2-lstm_47/while/lstm_cell_47/BiasAdd_2:output:0-lstm_47/while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @
lstm_47/while/lstm_cell_47/ReluRelu$lstm_47/while/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @╢
 lstm_47/while/lstm_cell_47/mul_3Mul,lstm_47/while/lstm_cell_47/clip_by_value:z:0-lstm_47/while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @з
 lstm_47/while/lstm_cell_47/add_5AddV2$lstm_47/while/lstm_cell_47/mul_2:z:0$lstm_47/while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @б
+lstm_47/while/lstm_cell_47/ReadVariableOp_3ReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_47/while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_47/while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_47/while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_47/while/lstm_cell_47/strided_slice_3StridedSlice3lstm_47/while/lstm_cell_47/ReadVariableOp_3:value:09lstm_47/while/lstm_cell_47/strided_slice_3/stack:output:0;lstm_47/while/lstm_cell_47/strided_slice_3/stack_1:output:0;lstm_47/while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_47/while/lstm_cell_47/MatMul_7MatMullstm_47_while_placeholder_23lstm_47/while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @╣
 lstm_47/while/lstm_cell_47/add_6AddV2-lstm_47/while/lstm_cell_47/BiasAdd_3:output:0-lstm_47/while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @g
"lstm_47/while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_47/while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_47/while/lstm_cell_47/Mul_4Mul$lstm_47/while/lstm_cell_47/add_6:z:0+lstm_47/while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @о
 lstm_47/while/lstm_cell_47/Add_7AddV2$lstm_47/while/lstm_cell_47/Mul_4:z:0+lstm_47/while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @y
4lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_47/while/lstm_cell_47/clip_by_value_2/MinimumMinimum$lstm_47/while/lstm_cell_47/Add_7:z:0=lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_47/while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_47/while/lstm_cell_47/clip_by_value_2Maximum6lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum:z:05lstm_47/while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @Б
!lstm_47/while/lstm_cell_47/Relu_1Relu$lstm_47/while/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @║
 lstm_47/while/lstm_cell_47/mul_5Mul.lstm_47/while/lstm_cell_47/clip_by_value_2:z:0/lstm_47/while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @х
2lstm_47/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_47_while_placeholder_1lstm_47_while_placeholder$lstm_47/while/lstm_cell_47/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_47/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_47/while/addAddV2lstm_47_while_placeholderlstm_47/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_47/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_47/while/add_1AddV2(lstm_47_while_lstm_47_while_loop_counterlstm_47/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_47/while/IdentityIdentitylstm_47/while/add_1:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: К
lstm_47/while/Identity_1Identity.lstm_47_while_lstm_47_while_maximum_iterations^lstm_47/while/NoOp*
T0*
_output_shapes
: q
lstm_47/while/Identity_2Identitylstm_47/while/add:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: Ю
lstm_47/while/Identity_3IdentityBlstm_47/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_47/while/NoOp*
T0*
_output_shapes
: С
lstm_47/while/Identity_4Identity$lstm_47/while/lstm_cell_47/mul_5:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:         @С
lstm_47/while/Identity_5Identity$lstm_47/while/lstm_cell_47/add_5:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:         @Ё
lstm_47/while/NoOpNoOp*^lstm_47/while/lstm_cell_47/ReadVariableOp,^lstm_47/while/lstm_cell_47/ReadVariableOp_1,^lstm_47/while/lstm_cell_47/ReadVariableOp_2,^lstm_47/while/lstm_cell_47/ReadVariableOp_30^lstm_47/while/lstm_cell_47/split/ReadVariableOp2^lstm_47/while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_47_while_identity_1!lstm_47/while/Identity_1:output:0"=
lstm_47_while_identity_2!lstm_47/while/Identity_2:output:0"=
lstm_47_while_identity_3!lstm_47/while/Identity_3:output:0"=
lstm_47_while_identity_4!lstm_47/while/Identity_4:output:0"=
lstm_47_while_identity_5!lstm_47/while/Identity_5:output:0"9
lstm_47_while_identitylstm_47/while/Identity:output:0"P
%lstm_47_while_lstm_47_strided_slice_1'lstm_47_while_lstm_47_strided_slice_1_0"j
2lstm_47_while_lstm_cell_47_readvariableop_resource4lstm_47_while_lstm_cell_47_readvariableop_resource_0"z
:lstm_47_while_lstm_cell_47_split_1_readvariableop_resource<lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0"v
8lstm_47_while_lstm_cell_47_split_readvariableop_resource:lstm_47_while_lstm_cell_47_split_readvariableop_resource_0"╚
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2Z
+lstm_47/while/lstm_cell_47/ReadVariableOp_1+lstm_47/while/lstm_cell_47/ReadVariableOp_12Z
+lstm_47/while/lstm_cell_47/ReadVariableOp_2+lstm_47/while/lstm_cell_47/ReadVariableOp_22Z
+lstm_47/while/lstm_cell_47/ReadVariableOp_3+lstm_47/while/lstm_cell_47/ReadVariableOp_32V
)lstm_47/while/lstm_cell_47/ReadVariableOp)lstm_47/while/lstm_cell_47/ReadVariableOp2b
/lstm_47/while/lstm_cell_47/split/ReadVariableOp/lstm_47/while/lstm_cell_47/split/ReadVariableOp2f
1lstm_47/while/lstm_cell_47/split_1/ReadVariableOp1lstm_47/while/lstm_cell_47/split_1/ReadVariableOp:
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
_user_specified_name" lstm_47/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_47/while/loop_counter
ю
ў
-__inference_lstm_cell_47_layer_call_fn_382994

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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_378091o
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
м
√
'sequential_23_lstm_47_while_cond_377156H
Dsequential_23_lstm_47_while_sequential_23_lstm_47_while_loop_counterN
Jsequential_23_lstm_47_while_sequential_23_lstm_47_while_maximum_iterations+
'sequential_23_lstm_47_while_placeholder-
)sequential_23_lstm_47_while_placeholder_1-
)sequential_23_lstm_47_while_placeholder_2-
)sequential_23_lstm_47_while_placeholder_3J
Fsequential_23_lstm_47_while_less_sequential_23_lstm_47_strided_slice_1`
\sequential_23_lstm_47_while_sequential_23_lstm_47_while_cond_377156___redundant_placeholder0`
\sequential_23_lstm_47_while_sequential_23_lstm_47_while_cond_377156___redundant_placeholder1`
\sequential_23_lstm_47_while_sequential_23_lstm_47_while_cond_377156___redundant_placeholder2`
\sequential_23_lstm_47_while_sequential_23_lstm_47_while_cond_377156___redundant_placeholder3(
$sequential_23_lstm_47_while_identity
║
 sequential_23/lstm_47/while/LessLess'sequential_23_lstm_47_while_placeholderFsequential_23_lstm_47_while_less_sequential_23_lstm_47_strided_slice_1*
T0*
_output_shapes
: w
$sequential_23/lstm_47/while/IdentityIdentity$sequential_23/lstm_47/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_23_lstm_47_while_identity-sequential_23/lstm_47/while/Identity:output:0*(
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
_user_specified_name0.sequential_23/lstm_47/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_23/lstm_47/while/loop_counter
М
у
lstm_46_while_cond_380194,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3.
*lstm_46_while_less_lstm_46_strided_slice_1D
@lstm_46_while_lstm_46_while_cond_380194___redundant_placeholder0D
@lstm_46_while_lstm_46_while_cond_380194___redundant_placeholder1D
@lstm_46_while_lstm_46_while_cond_380194___redundant_placeholder2D
@lstm_46_while_lstm_46_while_cond_380194___redundant_placeholder3
lstm_46_while_identity
В
lstm_46/while/LessLesslstm_46_while_placeholder*lstm_46_while_less_lstm_46_strided_slice_1*
T0*
_output_shapes
: [
lstm_46/while/IdentityIdentitylstm_46/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_46_while_identitylstm_46/while/Identity:output:0*(
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
_user_specified_name" lstm_46/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_46/while/loop_counter
Ў
ў
-__inference_lstm_cell_46_layer_call_fn_382765

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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377427p
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
▌И
ъ
C__inference_lstm_47_layer_call_and_return_conditional_losses_381961
inputs_0>
*lstm_cell_47_split_readvariableop_resource:
АА;
,lstm_cell_47_split_1_readvariableop_resource:	А7
$lstm_cell_47_readvariableop_resource:	@А
identityИвlstm_cell_47/ReadVariableOpвlstm_cell_47/ReadVariableOp_1вlstm_cell_47/ReadVariableOp_2вlstm_cell_47/ReadVariableOp_3в!lstm_cell_47/split/ReadVariableOpв#lstm_cell_47/split_1/ReadVariableOpвwhileK
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
lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_47/split/ReadVariableOpReadVariableOp*lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_47/splitSplit%lstm_cell_47/split/split_dim:output:0)lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_47/MatMulMatMulstrided_slice_2:output:0lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_47/split_1/ReadVariableOpReadVariableOp,lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_47/split_1Split'lstm_cell_47/split_1/split_dim:output:0+lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_47/BiasAddBiasAddlstm_cell_47/MatMul:product:0lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_1BiasAddlstm_cell_47/MatMul_1:product:0lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_2BiasAddlstm_cell_47/MatMul_2:product:0lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_3BiasAddlstm_cell_47/MatMul_3:product:0lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_47/ReadVariableOpReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_47/strided_sliceStridedSlice#lstm_cell_47/ReadVariableOp:value:0)lstm_cell_47/strided_slice/stack:output:0+lstm_cell_47/strided_slice/stack_1:output:0+lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_47/MatMul_4MatMulzeros:output:0#lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_47/addAddV2lstm_cell_47/BiasAdd:output:0lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_47/MulMullstm_cell_47/add:z:0lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_47/Add_1AddV2lstm_cell_47/Mul:z:0lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_47/clip_by_value/MinimumMinimumlstm_cell_47/Add_1:z:0-lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_47/clip_by_valueMaximum&lstm_cell_47/clip_by_value/Minimum:z:0%lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_1ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_1StridedSlice%lstm_cell_47/ReadVariableOp_1:value:0+lstm_cell_47/strided_slice_1/stack:output:0-lstm_cell_47/strided_slice_1/stack_1:output:0-lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_5MatMulzeros:output:0%lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_2AddV2lstm_cell_47/BiasAdd_1:output:0lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_1Mullstm_cell_47/add_2:z:0lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_3AddV2lstm_cell_47/Mul_1:z:0lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_cell_47/Add_3:z:0/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_1Maximum(lstm_cell_47/clip_by_value_1/Minimum:z:0'lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_47/mul_2Mul lstm_cell_47/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_2ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_2StridedSlice%lstm_cell_47/ReadVariableOp_2:value:0+lstm_cell_47/strided_slice_2/stack:output:0-lstm_cell_47/strided_slice_2/stack_1:output:0-lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_6MatMulzeros:output:0%lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_4AddV2lstm_cell_47/BiasAdd_2:output:0lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/ReluRelulstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_47/mul_3Mullstm_cell_47/clip_by_value:z:0lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_47/add_5AddV2lstm_cell_47/mul_2:z:0lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_3ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_3StridedSlice%lstm_cell_47/ReadVariableOp_3:value:0+lstm_cell_47/strided_slice_3/stack:output:0-lstm_cell_47/strided_slice_3/stack_1:output:0-lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_7MatMulzeros:output:0%lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_6AddV2lstm_cell_47/BiasAdd_3:output:0lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_4Mullstm_cell_47/add_6:z:0lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_7AddV2lstm_cell_47/Mul_4:z:0lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_cell_47/Add_7:z:0/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_2Maximum(lstm_cell_47/clip_by_value_2/Minimum:z:0'lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_47/Relu_1Relulstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_47/mul_5Mul lstm_cell_47/clip_by_value_2:z:0!lstm_cell_47/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_47_split_readvariableop_resource,lstm_cell_47_split_1_readvariableop_resource$lstm_cell_47_readvariableop_resource*
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
while_body_381821*
condR
while_cond_381820*K
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
NoOpNoOp^lstm_cell_47/ReadVariableOp^lstm_cell_47/ReadVariableOp_1^lstm_cell_47/ReadVariableOp_2^lstm_cell_47/ReadVariableOp_3"^lstm_cell_47/split/ReadVariableOp$^lstm_cell_47/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2>
lstm_cell_47/ReadVariableOp_1lstm_cell_47/ReadVariableOp_12>
lstm_cell_47/ReadVariableOp_2lstm_cell_47/ReadVariableOp_22>
lstm_cell_47/ReadVariableOp_3lstm_cell_47/ReadVariableOp_32:
lstm_cell_47/ReadVariableOplstm_cell_47/ReadVariableOp2F
!lstm_cell_47/split/ReadVariableOp!lstm_cell_47/split/ReadVariableOp2J
#lstm_cell_47/split_1/ReadVariableOp#lstm_cell_47/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  А
"
_user_specified_name
inputs_0
ЫЙ
ъ
C__inference_lstm_46_layer_call_and_return_conditional_losses_380893
inputs_0=
*lstm_cell_46_split_readvariableop_resource:	А;
,lstm_cell_46_split_1_readvariableop_resource:	А8
$lstm_cell_46_readvariableop_resource:
АА
identityИвlstm_cell_46/ReadVariableOpвlstm_cell_46/ReadVariableOp_1вlstm_cell_46/ReadVariableOp_2вlstm_cell_46/ReadVariableOp_3в!lstm_cell_46/split/ReadVariableOpв#lstm_cell_46/split_1/ReadVariableOpвwhileK
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
lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_46/split/ReadVariableOpReadVariableOp*lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_46/splitSplit%lstm_cell_46/split/split_dim:output:0)lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_46/MatMulMatMulstrided_slice_2:output:0lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_46/split_1/ReadVariableOpReadVariableOp,lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_46/split_1Split'lstm_cell_46/split_1/split_dim:output:0+lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_46/BiasAddBiasAddlstm_cell_46/MatMul:product:0lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_1BiasAddlstm_cell_46/MatMul_1:product:0lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_2BiasAddlstm_cell_46/MatMul_2:product:0lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_3BiasAddlstm_cell_46/MatMul_3:product:0lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_46/ReadVariableOpReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_46/strided_sliceStridedSlice#lstm_cell_46/ReadVariableOp:value:0)lstm_cell_46/strided_slice/stack:output:0+lstm_cell_46/strided_slice/stack_1:output:0+lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_46/MatMul_4MatMulzeros:output:0#lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_46/addAddV2lstm_cell_46/BiasAdd:output:0lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_46/MulMullstm_cell_46/add:z:0lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_46/Add_1AddV2lstm_cell_46/Mul:z:0lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_46/clip_by_value/MinimumMinimumlstm_cell_46/Add_1:z:0-lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_46/clip_by_valueMaximum&lstm_cell_46/clip_by_value/Minimum:z:0%lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_1ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_1StridedSlice%lstm_cell_46/ReadVariableOp_1:value:0+lstm_cell_46/strided_slice_1/stack:output:0-lstm_cell_46/strided_slice_1/stack_1:output:0-lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_5MatMulzeros:output:0%lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_2AddV2lstm_cell_46/BiasAdd_1:output:0lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_1Mullstm_cell_46/add_2:z:0lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_3AddV2lstm_cell_46/Mul_1:z:0lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_cell_46/Add_3:z:0/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_1Maximum(lstm_cell_46/clip_by_value_1/Minimum:z:0'lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_46/mul_2Mul lstm_cell_46/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_2ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_2StridedSlice%lstm_cell_46/ReadVariableOp_2:value:0+lstm_cell_46/strided_slice_2/stack:output:0-lstm_cell_46/strided_slice_2/stack_1:output:0-lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_6MatMulzeros:output:0%lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_4AddV2lstm_cell_46/BiasAdd_2:output:0lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_46/ReluRelulstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_46/mul_3Mullstm_cell_46/clip_by_value:z:0lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_46/add_5AddV2lstm_cell_46/mul_2:z:0lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_3ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_3StridedSlice%lstm_cell_46/ReadVariableOp_3:value:0+lstm_cell_46/strided_slice_3/stack:output:0-lstm_cell_46/strided_slice_3/stack_1:output:0-lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_7MatMulzeros:output:0%lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_6AddV2lstm_cell_46/BiasAdd_3:output:0lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_4Mullstm_cell_46/add_6:z:0lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_7AddV2lstm_cell_46/Mul_4:z:0lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_cell_46/Add_7:z:0/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_2Maximum(lstm_cell_46/clip_by_value_2/Minimum:z:0'lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_46/Relu_1Relulstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_46/mul_5Mul lstm_cell_46/clip_by_value_2:z:0!lstm_cell_46/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_46_split_readvariableop_resource,lstm_cell_46_split_1_readvariableop_resource$lstm_cell_46_readvariableop_resource*
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
while_body_380753*
condR
while_cond_380752*M
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
NoOpNoOp^lstm_cell_46/ReadVariableOp^lstm_cell_46/ReadVariableOp_1^lstm_cell_46/ReadVariableOp_2^lstm_cell_46/ReadVariableOp_3"^lstm_cell_46/split/ReadVariableOp$^lstm_cell_46/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2>
lstm_cell_46/ReadVariableOp_1lstm_cell_46/ReadVariableOp_12>
lstm_cell_46/ReadVariableOp_2lstm_cell_46/ReadVariableOp_22>
lstm_cell_46/ReadVariableOp_3lstm_cell_46/ReadVariableOp_32:
lstm_cell_46/ReadVariableOplstm_cell_46/ReadVariableOp2F
!lstm_cell_46/split/ReadVariableOp!lstm_cell_46/split/ReadVariableOp2J
#lstm_cell_46/split_1/ReadVariableOp#lstm_cell_46/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Пъ
Н
__inference__traced_save_383381
file_prefix8
&read_disablecopyonread_dense_23_kernel:@4
&read_1_disablecopyonread_dense_23_bias:G
4read_2_disablecopyonread_lstm_46_lstm_cell_46_kernel:	АR
>read_3_disablecopyonread_lstm_46_lstm_cell_46_recurrent_kernel:
ААA
2read_4_disablecopyonread_lstm_46_lstm_cell_46_bias:	АH
4read_5_disablecopyonread_lstm_47_lstm_cell_47_kernel:
ААQ
>read_6_disablecopyonread_lstm_47_lstm_cell_47_recurrent_kernel:	@АA
2read_7_disablecopyonread_lstm_47_lstm_cell_47_bias:	А)
read_8_disablecopyonread_beta_1: )
read_9_disablecopyonread_beta_2: )
read_10_disablecopyonread_decay: 1
'read_11_disablecopyonread_learning_rate: -
#read_12_disablecopyonread_adam_iter:	 )
read_13_disablecopyonread_total: )
read_14_disablecopyonread_count: B
0read_15_disablecopyonread_adam_dense_23_kernel_m:@<
.read_16_disablecopyonread_adam_dense_23_bias_m:O
<read_17_disablecopyonread_adam_lstm_46_lstm_cell_46_kernel_m:	АZ
Fread_18_disablecopyonread_adam_lstm_46_lstm_cell_46_recurrent_kernel_m:
ААI
:read_19_disablecopyonread_adam_lstm_46_lstm_cell_46_bias_m:	АP
<read_20_disablecopyonread_adam_lstm_47_lstm_cell_47_kernel_m:
ААY
Fread_21_disablecopyonread_adam_lstm_47_lstm_cell_47_recurrent_kernel_m:	@АI
:read_22_disablecopyonread_adam_lstm_47_lstm_cell_47_bias_m:	АB
0read_23_disablecopyonread_adam_dense_23_kernel_v:@<
.read_24_disablecopyonread_adam_dense_23_bias_v:O
<read_25_disablecopyonread_adam_lstm_46_lstm_cell_46_kernel_v:	АZ
Fread_26_disablecopyonread_adam_lstm_46_lstm_cell_46_recurrent_kernel_v:
ААI
:read_27_disablecopyonread_adam_lstm_46_lstm_cell_46_bias_v:	АP
<read_28_disablecopyonread_adam_lstm_47_lstm_cell_47_kernel_v:
ААY
Fread_29_disablecopyonread_adam_lstm_47_lstm_cell_47_recurrent_kernel_v:	@АI
:read_30_disablecopyonread_adam_lstm_47_lstm_cell_47_bias_v:	А
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
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 в
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_23_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 в
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_23_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_lstm_46_lstm_cell_46_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_lstm_46_lstm_cell_46_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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
Read_3/DisableCopyOnReadDisableCopyOnRead>read_3_disablecopyonread_lstm_46_lstm_cell_46_recurrent_kernel"/device:CPU:0*
_output_shapes
 └
Read_3/ReadVariableOpReadVariableOp>read_3_disablecopyonread_lstm_46_lstm_cell_46_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
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
Read_4/DisableCopyOnReadDisableCopyOnRead2read_4_disablecopyonread_lstm_46_lstm_cell_46_bias"/device:CPU:0*
_output_shapes
 п
Read_4/ReadVariableOpReadVariableOp2read_4_disablecopyonread_lstm_46_lstm_cell_46_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_lstm_47_lstm_cell_47_kernel"/device:CPU:0*
_output_shapes
 ╢
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_lstm_47_lstm_cell_47_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
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
Read_6/DisableCopyOnReadDisableCopyOnRead>read_6_disablecopyonread_lstm_47_lstm_cell_47_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_6/ReadVariableOpReadVariableOp>read_6_disablecopyonread_lstm_47_lstm_cell_47_recurrent_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
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
Read_7/DisableCopyOnReadDisableCopyOnRead2read_7_disablecopyonread_lstm_47_lstm_cell_47_bias"/device:CPU:0*
_output_shapes
 п
Read_7/ReadVariableOpReadVariableOp2read_7_disablecopyonread_lstm_47_lstm_cell_47_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_dense_23_kernel_m"/device:CPU:0*
_output_shapes
 ▓
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_dense_23_kernel_m^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_dense_23_bias_m"/device:CPU:0*
_output_shapes
 м
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_dense_23_bias_m^Read_16/DisableCopyOnRead"/device:CPU:0*
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
Read_17/DisableCopyOnReadDisableCopyOnRead<read_17_disablecopyonread_adam_lstm_46_lstm_cell_46_kernel_m"/device:CPU:0*
_output_shapes
 ┐
Read_17/ReadVariableOpReadVariableOp<read_17_disablecopyonread_adam_lstm_46_lstm_cell_46_kernel_m^Read_17/DisableCopyOnRead"/device:CPU:0*
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
Read_18/DisableCopyOnReadDisableCopyOnReadFread_18_disablecopyonread_adam_lstm_46_lstm_cell_46_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 ╩
Read_18/ReadVariableOpReadVariableOpFread_18_disablecopyonread_adam_lstm_46_lstm_cell_46_recurrent_kernel_m^Read_18/DisableCopyOnRead"/device:CPU:0* 
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
Read_19/DisableCopyOnReadDisableCopyOnRead:read_19_disablecopyonread_adam_lstm_46_lstm_cell_46_bias_m"/device:CPU:0*
_output_shapes
 ╣
Read_19/ReadVariableOpReadVariableOp:read_19_disablecopyonread_adam_lstm_46_lstm_cell_46_bias_m^Read_19/DisableCopyOnRead"/device:CPU:0*
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
Read_20/DisableCopyOnReadDisableCopyOnRead<read_20_disablecopyonread_adam_lstm_47_lstm_cell_47_kernel_m"/device:CPU:0*
_output_shapes
 └
Read_20/ReadVariableOpReadVariableOp<read_20_disablecopyonread_adam_lstm_47_lstm_cell_47_kernel_m^Read_20/DisableCopyOnRead"/device:CPU:0* 
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
Read_21/DisableCopyOnReadDisableCopyOnReadFread_21_disablecopyonread_adam_lstm_47_lstm_cell_47_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 ╔
Read_21/ReadVariableOpReadVariableOpFread_21_disablecopyonread_adam_lstm_47_lstm_cell_47_recurrent_kernel_m^Read_21/DisableCopyOnRead"/device:CPU:0*
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
Read_22/DisableCopyOnReadDisableCopyOnRead:read_22_disablecopyonread_adam_lstm_47_lstm_cell_47_bias_m"/device:CPU:0*
_output_shapes
 ╣
Read_22/ReadVariableOpReadVariableOp:read_22_disablecopyonread_adam_lstm_47_lstm_cell_47_bias_m^Read_22/DisableCopyOnRead"/device:CPU:0*
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
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_dense_23_kernel_v"/device:CPU:0*
_output_shapes
 ▓
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_dense_23_kernel_v^Read_23/DisableCopyOnRead"/device:CPU:0*
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
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_dense_23_bias_v"/device:CPU:0*
_output_shapes
 м
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_dense_23_bias_v^Read_24/DisableCopyOnRead"/device:CPU:0*
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
Read_25/DisableCopyOnReadDisableCopyOnRead<read_25_disablecopyonread_adam_lstm_46_lstm_cell_46_kernel_v"/device:CPU:0*
_output_shapes
 ┐
Read_25/ReadVariableOpReadVariableOp<read_25_disablecopyonread_adam_lstm_46_lstm_cell_46_kernel_v^Read_25/DisableCopyOnRead"/device:CPU:0*
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
Read_26/DisableCopyOnReadDisableCopyOnReadFread_26_disablecopyonread_adam_lstm_46_lstm_cell_46_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 ╩
Read_26/ReadVariableOpReadVariableOpFread_26_disablecopyonread_adam_lstm_46_lstm_cell_46_recurrent_kernel_v^Read_26/DisableCopyOnRead"/device:CPU:0* 
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
Read_27/DisableCopyOnReadDisableCopyOnRead:read_27_disablecopyonread_adam_lstm_46_lstm_cell_46_bias_v"/device:CPU:0*
_output_shapes
 ╣
Read_27/ReadVariableOpReadVariableOp:read_27_disablecopyonread_adam_lstm_46_lstm_cell_46_bias_v^Read_27/DisableCopyOnRead"/device:CPU:0*
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
Read_28/DisableCopyOnReadDisableCopyOnRead<read_28_disablecopyonread_adam_lstm_47_lstm_cell_47_kernel_v"/device:CPU:0*
_output_shapes
 └
Read_28/ReadVariableOpReadVariableOp<read_28_disablecopyonread_adam_lstm_47_lstm_cell_47_kernel_v^Read_28/DisableCopyOnRead"/device:CPU:0* 
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
Read_29/DisableCopyOnReadDisableCopyOnReadFread_29_disablecopyonread_adam_lstm_47_lstm_cell_47_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 ╔
Read_29/ReadVariableOpReadVariableOpFread_29_disablecopyonread_adam_lstm_47_lstm_cell_47_recurrent_kernel_v^Read_29/DisableCopyOnRead"/device:CPU:0*
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
Read_30/DisableCopyOnReadDisableCopyOnRead:read_30_disablecopyonread_adam_lstm_47_lstm_cell_47_bias_v"/device:CPU:0*
_output_shapes
 ╣
Read_30/ReadVariableOpReadVariableOp:read_30_disablecopyonread_adam_lstm_47_lstm_cell_47_bias_v^Read_30/DisableCopyOnRead"/device:CPU:0*
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
Ў
ў
-__inference_lstm_cell_46_layer_call_fn_382782

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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377629p
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
while_body_378150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_47_378174_0:
АА*
while_lstm_cell_47_378176_0:	А.
while_lstm_cell_47_378178_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_47_378174:
АА(
while_lstm_cell_47_378176:	А,
while_lstm_cell_47_378178:	@АИв*while/lstm_cell_47/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0│
*while/lstm_cell_47/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_47_378174_0while_lstm_cell_47_378176_0while_lstm_cell_47_378178_0*
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_378091▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_47/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_47/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @Р
while/Identity_5Identity3while/lstm_cell_47/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @y

while/NoOpNoOp+^while/lstm_cell_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_47_378174while_lstm_cell_47_378174_0"8
while_lstm_cell_47_378176while_lstm_cell_47_378176_0"8
while_lstm_cell_47_378178while_lstm_cell_47_378178_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2X
*while/lstm_cell_47/StatefulPartitionedCall*while/lstm_cell_47/StatefulPartitionedCall:
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
Ы	
├
while_cond_377440
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_377440___redundant_placeholder04
0while_while_cond_377440___redundant_placeholder14
0while_while_cond_377440___redundant_placeholder24
0while_while_cond_377440___redundant_placeholder3
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
Ё
▌
I__inference_sequential_23_layer_call_and_return_conditional_losses_378777

inputs!
lstm_46_378490:	А
lstm_46_378492:	А"
lstm_46_378494:
АА"
lstm_47_378753:
АА
lstm_47_378755:	А!
lstm_47_378757:	@А!
dense_23_378771:@
dense_23_378773:
identityИв dense_23/StatefulPartitionedCallвlstm_46/StatefulPartitionedCallвlstm_47/StatefulPartitionedCallГ
lstm_46/StatefulPartitionedCallStatefulPartitionedCallinputslstm_46_378490lstm_46_378492lstm_46_378494*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         
А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_46_layer_call_and_return_conditional_losses_378489а
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_378753lstm_47_378755lstm_47_378757*
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_378752Т
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_23_378771dense_23_378773*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_378770x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_23/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
б~
ж	
while_body_378349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_46_split_readvariableop_resource_0:	АC
4while_lstm_cell_46_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_46_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_46_split_readvariableop_resource:	АA
2while_lstm_cell_46_split_1_readvariableop_resource:	А>
*while_lstm_cell_46_readvariableop_resource:
ААИв!while/lstm_cell_46/ReadVariableOpв#while/lstm_cell_46/ReadVariableOp_1в#while/lstm_cell_46/ReadVariableOp_2в#while/lstm_cell_46/ReadVariableOp_3в'while/lstm_cell_46/split/ReadVariableOpв)while/lstm_cell_46/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_46/split/ReadVariableOpReadVariableOp2while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_46/splitSplit+while/lstm_cell_46/split/split_dim:output:0/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_46/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_46/split_1Split-while/lstm_cell_46/split_1/split_dim:output:01while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_46/BiasAddBiasAdd#while/lstm_cell_46/MatMul:product:0#while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_1BiasAdd%while/lstm_cell_46/MatMul_1:product:0#while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_2BiasAdd%while/lstm_cell_46/MatMul_2:product:0#while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_3BiasAdd%while/lstm_cell_46/MatMul_3:product:0#while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_46/ReadVariableOpReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_46/strided_sliceStridedSlice)while/lstm_cell_46/ReadVariableOp:value:0/while/lstm_cell_46/strided_slice/stack:output:01while/lstm_cell_46/strided_slice/stack_1:output:01while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_46/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_46/addAddV2#while/lstm_cell_46/BiasAdd:output:0%while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_46/MulMulwhile/lstm_cell_46/add:z:0!while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_46/Add_1AddV2while/lstm_cell_46/Mul:z:0#while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_46/clip_by_value/MinimumMinimumwhile/lstm_cell_46/Add_1:z:03while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_46/clip_by_valueMaximum,while/lstm_cell_46/clip_by_value/Minimum:z:0+while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_1ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_1StridedSlice+while/lstm_cell_46/ReadVariableOp_1:value:01while/lstm_cell_46/strided_slice_1/stack:output:03while/lstm_cell_46/strided_slice_1/stack_1:output:03while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_2AddV2%while/lstm_cell_46/BiasAdd_1:output:0%while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_1Mulwhile/lstm_cell_46/add_2:z:0#while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_3AddV2while/lstm_cell_46/Mul_1:z:0#while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_1/MinimumMinimumwhile/lstm_cell_46/Add_3:z:05while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_1Maximum.while/lstm_cell_46/clip_by_value_1/Minimum:z:0-while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_46/mul_2Mul&while/lstm_cell_46/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_2ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_2StridedSlice+while/lstm_cell_46/ReadVariableOp_2:value:01while/lstm_cell_46/strided_slice_2/stack:output:03while/lstm_cell_46/strided_slice_2/stack_1:output:03while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_4AddV2%while/lstm_cell_46/BiasAdd_2:output:0%while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_46/ReluReluwhile/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_46/mul_3Mul$while/lstm_cell_46/clip_by_value:z:0%while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_46/add_5AddV2while/lstm_cell_46/mul_2:z:0while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_3ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_3StridedSlice+while/lstm_cell_46/ReadVariableOp_3:value:01while/lstm_cell_46/strided_slice_3/stack:output:03while/lstm_cell_46/strided_slice_3/stack_1:output:03while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_6AddV2%while/lstm_cell_46/BiasAdd_3:output:0%while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_4Mulwhile/lstm_cell_46/add_6:z:0#while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_7AddV2while/lstm_cell_46/Mul_4:z:0#while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_2/MinimumMinimumwhile/lstm_cell_46/Add_7:z:05while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_2Maximum.while/lstm_cell_46/clip_by_value_2/Minimum:z:0-while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_46/Relu_1Reluwhile/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_46/mul_5Mul&while/lstm_cell_46/clip_by_value_2:z:0'while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_46/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_46/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_46/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_46/ReadVariableOp$^while/lstm_cell_46/ReadVariableOp_1$^while/lstm_cell_46/ReadVariableOp_2$^while/lstm_cell_46/ReadVariableOp_3(^while/lstm_cell_46/split/ReadVariableOp*^while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_46_readvariableop_resource,while_lstm_cell_46_readvariableop_resource_0"j
2while_lstm_cell_46_split_1_readvariableop_resource4while_lstm_cell_46_split_1_readvariableop_resource_0"f
0while_lstm_cell_46_split_readvariableop_resource2while_lstm_cell_46_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_46/ReadVariableOp_1#while/lstm_cell_46/ReadVariableOp_12J
#while/lstm_cell_46/ReadVariableOp_2#while/lstm_cell_46/ReadVariableOp_22J
#while/lstm_cell_46/ReadVariableOp_3#while/lstm_cell_46/ReadVariableOp_32F
!while/lstm_cell_46/ReadVariableOp!while/lstm_cell_46/ReadVariableOp2R
'while/lstm_cell_46/split/ReadVariableOp'while/lstm_cell_46/split/ReadVariableOp2V
)while/lstm_cell_46/split_1/ReadVariableOp)while/lstm_cell_46/split_1/ReadVariableOp:
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
В
╢
(__inference_lstm_46_layer_call_fn_380637

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
:         
А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_46_layer_call_and_return_conditional_losses_379353t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         
А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
В
╢
(__inference_lstm_46_layer_call_fn_380626

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
:         
А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_46_layer_call_and_return_conditional_losses_378489t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         
А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ТР
╛
lstm_46_while_body_380195,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3+
'lstm_46_while_lstm_46_strided_slice_1_0g
clstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_46_while_lstm_cell_46_split_readvariableop_resource_0:	АK
<lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0:	АH
4lstm_46_while_lstm_cell_46_readvariableop_resource_0:
АА
lstm_46_while_identity
lstm_46_while_identity_1
lstm_46_while_identity_2
lstm_46_while_identity_3
lstm_46_while_identity_4
lstm_46_while_identity_5)
%lstm_46_while_lstm_46_strided_slice_1e
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorK
8lstm_46_while_lstm_cell_46_split_readvariableop_resource:	АI
:lstm_46_while_lstm_cell_46_split_1_readvariableop_resource:	АF
2lstm_46_while_lstm_cell_46_readvariableop_resource:
ААИв)lstm_46/while/lstm_cell_46/ReadVariableOpв+lstm_46/while/lstm_cell_46/ReadVariableOp_1в+lstm_46/while/lstm_cell_46/ReadVariableOp_2в+lstm_46/while/lstm_cell_46/ReadVariableOp_3в/lstm_46/while/lstm_cell_46/split/ReadVariableOpв1lstm_46/while/lstm_cell_46/split_1/ReadVariableOpР
?lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_46/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0lstm_46_while_placeholderHlstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0l
*lstm_46/while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
/lstm_46/while/lstm_cell_46/split/ReadVariableOpReadVariableOp:lstm_46_while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0є
 lstm_46/while/lstm_cell_46/splitSplit3lstm_46/while/lstm_cell_46/split/split_dim:output:07lstm_46/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split├
!lstm_46/while/lstm_cell_46/MatMulMatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         А┼
#lstm_46/while/lstm_cell_46/MatMul_1MatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         А┼
#lstm_46/while/lstm_cell_46/MatMul_2MatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         А┼
#lstm_46/while/lstm_cell_46/MatMul_3MatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аn
,lstm_46/while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_46/while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp<lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0щ
"lstm_46/while/lstm_cell_46/split_1Split5lstm_46/while/lstm_cell_46/split_1/split_dim:output:09lstm_46/while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_split║
"lstm_46/while/lstm_cell_46/BiasAddBiasAdd+lstm_46/while/lstm_cell_46/MatMul:product:0+lstm_46/while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         А╛
$lstm_46/while/lstm_cell_46/BiasAdd_1BiasAdd-lstm_46/while/lstm_cell_46/MatMul_1:product:0+lstm_46/while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         А╛
$lstm_46/while/lstm_cell_46/BiasAdd_2BiasAdd-lstm_46/while/lstm_cell_46/MatMul_2:product:0+lstm_46/while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         А╛
$lstm_46/while/lstm_cell_46/BiasAdd_3BiasAdd-lstm_46/while/lstm_cell_46/MatMul_3:product:0+lstm_46/while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         Аа
)lstm_46/while/lstm_cell_46/ReadVariableOpReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0
.lstm_46/while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_46/while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Б
0lstm_46/while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ї
(lstm_46/while/lstm_cell_46/strided_sliceStridedSlice1lstm_46/while/lstm_cell_46/ReadVariableOp:value:07lstm_46/while/lstm_cell_46/strided_slice/stack:output:09lstm_46/while/lstm_cell_46/strided_slice/stack_1:output:09lstm_46/while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask░
#lstm_46/while/lstm_cell_46/MatMul_4MatMullstm_46_while_placeholder_21lstm_46/while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         А╢
lstm_46/while/lstm_cell_46/addAddV2+lstm_46/while/lstm_cell_46/BiasAdd:output:0-lstm_46/while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         Аe
 lstm_46/while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_46/while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?з
lstm_46/while/lstm_cell_46/MulMul"lstm_46/while/lstm_cell_46/add:z:0)lstm_46/while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         Ан
 lstm_46/while/lstm_cell_46/Add_1AddV2"lstm_46/while/lstm_cell_46/Mul:z:0+lstm_46/while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аw
2lstm_46/while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
0lstm_46/while/lstm_cell_46/clip_by_value/MinimumMinimum$lstm_46/while/lstm_cell_46/Add_1:z:0;lstm_46/while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аo
*lstm_46/while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(lstm_46/while/lstm_cell_46/clip_by_valueMaximum4lstm_46/while/lstm_cell_46/clip_by_value/Minimum:z:03lstm_46/while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         Ав
+lstm_46/while/lstm_cell_46/ReadVariableOp_1ReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_46/while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_46/while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_46/while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_46/while/lstm_cell_46/strided_slice_1StridedSlice3lstm_46/while/lstm_cell_46/ReadVariableOp_1:value:09lstm_46/while/lstm_cell_46/strided_slice_1/stack:output:0;lstm_46/while/lstm_cell_46/strided_slice_1/stack_1:output:0;lstm_46/while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_46/while/lstm_cell_46/MatMul_5MatMullstm_46_while_placeholder_23lstm_46/while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         А║
 lstm_46/while/lstm_cell_46/add_2AddV2-lstm_46/while/lstm_cell_46/BiasAdd_1:output:0-lstm_46/while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         Аg
"lstm_46/while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_46/while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_46/while/lstm_cell_46/Mul_1Mul$lstm_46/while/lstm_cell_46/add_2:z:0+lstm_46/while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         Ап
 lstm_46/while/lstm_cell_46/Add_3AddV2$lstm_46/while/lstm_cell_46/Mul_1:z:0+lstm_46/while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аy
4lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_46/while/lstm_cell_46/clip_by_value_1/MinimumMinimum$lstm_46/while/lstm_cell_46/Add_3:z:0=lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_46/while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_46/while/lstm_cell_46/clip_by_value_1Maximum6lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum:z:05lstm_46/while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         Аз
 lstm_46/while/lstm_cell_46/mul_2Mul.lstm_46/while/lstm_cell_46/clip_by_value_1:z:0lstm_46_while_placeholder_3*
T0*(
_output_shapes
:         Ав
+lstm_46/while/lstm_cell_46/ReadVariableOp_2ReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_46/while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_46/while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_46/while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_46/while/lstm_cell_46/strided_slice_2StridedSlice3lstm_46/while/lstm_cell_46/ReadVariableOp_2:value:09lstm_46/while/lstm_cell_46/strided_slice_2/stack:output:0;lstm_46/while/lstm_cell_46/strided_slice_2/stack_1:output:0;lstm_46/while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_46/while/lstm_cell_46/MatMul_6MatMullstm_46_while_placeholder_23lstm_46/while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         А║
 lstm_46/while/lstm_cell_46/add_4AddV2-lstm_46/while/lstm_cell_46/BiasAdd_2:output:0-lstm_46/while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         АА
lstm_46/while/lstm_cell_46/ReluRelu$lstm_46/while/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         А╖
 lstm_46/while/lstm_cell_46/mul_3Mul,lstm_46/while/lstm_cell_46/clip_by_value:z:0-lstm_46/while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         Аи
 lstm_46/while/lstm_cell_46/add_5AddV2$lstm_46/while/lstm_cell_46/mul_2:z:0$lstm_46/while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         Ав
+lstm_46/while/lstm_cell_46/ReadVariableOp_3ReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_46/while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_46/while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_46/while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_46/while/lstm_cell_46/strided_slice_3StridedSlice3lstm_46/while/lstm_cell_46/ReadVariableOp_3:value:09lstm_46/while/lstm_cell_46/strided_slice_3/stack:output:0;lstm_46/while/lstm_cell_46/strided_slice_3/stack_1:output:0;lstm_46/while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_46/while/lstm_cell_46/MatMul_7MatMullstm_46_while_placeholder_23lstm_46/while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         А║
 lstm_46/while/lstm_cell_46/add_6AddV2-lstm_46/while/lstm_cell_46/BiasAdd_3:output:0-lstm_46/while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         Аg
"lstm_46/while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_46/while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_46/while/lstm_cell_46/Mul_4Mul$lstm_46/while/lstm_cell_46/add_6:z:0+lstm_46/while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         Ап
 lstm_46/while/lstm_cell_46/Add_7AddV2$lstm_46/while/lstm_cell_46/Mul_4:z:0+lstm_46/while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аy
4lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_46/while/lstm_cell_46/clip_by_value_2/MinimumMinimum$lstm_46/while/lstm_cell_46/Add_7:z:0=lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_46/while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_46/while/lstm_cell_46/clip_by_value_2Maximum6lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum:z:05lstm_46/while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АВ
!lstm_46/while/lstm_cell_46/Relu_1Relu$lstm_46/while/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         А╗
 lstm_46/while/lstm_cell_46/mul_5Mul.lstm_46/while/lstm_cell_46/clip_by_value_2:z:0/lstm_46/while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         Ах
2lstm_46/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_46_while_placeholder_1lstm_46_while_placeholder$lstm_46/while/lstm_cell_46/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_46/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_46/while/addAddV2lstm_46_while_placeholderlstm_46/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_46/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_46/while/add_1AddV2(lstm_46_while_lstm_46_while_loop_counterlstm_46/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_46/while/IdentityIdentitylstm_46/while/add_1:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: К
lstm_46/while/Identity_1Identity.lstm_46_while_lstm_46_while_maximum_iterations^lstm_46/while/NoOp*
T0*
_output_shapes
: q
lstm_46/while/Identity_2Identitylstm_46/while/add:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: Ю
lstm_46/while/Identity_3IdentityBlstm_46/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_46/while/NoOp*
T0*
_output_shapes
: Т
lstm_46/while/Identity_4Identity$lstm_46/while/lstm_cell_46/mul_5:z:0^lstm_46/while/NoOp*
T0*(
_output_shapes
:         АТ
lstm_46/while/Identity_5Identity$lstm_46/while/lstm_cell_46/add_5:z:0^lstm_46/while/NoOp*
T0*(
_output_shapes
:         АЁ
lstm_46/while/NoOpNoOp*^lstm_46/while/lstm_cell_46/ReadVariableOp,^lstm_46/while/lstm_cell_46/ReadVariableOp_1,^lstm_46/while/lstm_cell_46/ReadVariableOp_2,^lstm_46/while/lstm_cell_46/ReadVariableOp_30^lstm_46/while/lstm_cell_46/split/ReadVariableOp2^lstm_46/while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_46_while_identity_1!lstm_46/while/Identity_1:output:0"=
lstm_46_while_identity_2!lstm_46/while/Identity_2:output:0"=
lstm_46_while_identity_3!lstm_46/while/Identity_3:output:0"=
lstm_46_while_identity_4!lstm_46/while/Identity_4:output:0"=
lstm_46_while_identity_5!lstm_46/while/Identity_5:output:0"9
lstm_46_while_identitylstm_46/while/Identity:output:0"P
%lstm_46_while_lstm_46_strided_slice_1'lstm_46_while_lstm_46_strided_slice_1_0"j
2lstm_46_while_lstm_cell_46_readvariableop_resource4lstm_46_while_lstm_cell_46_readvariableop_resource_0"z
:lstm_46_while_lstm_cell_46_split_1_readvariableop_resource<lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0"v
8lstm_46_while_lstm_cell_46_split_readvariableop_resource:lstm_46_while_lstm_cell_46_split_readvariableop_resource_0"╚
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2Z
+lstm_46/while/lstm_cell_46/ReadVariableOp_1+lstm_46/while/lstm_cell_46/ReadVariableOp_12Z
+lstm_46/while/lstm_cell_46/ReadVariableOp_2+lstm_46/while/lstm_cell_46/ReadVariableOp_22Z
+lstm_46/while/lstm_cell_46/ReadVariableOp_3+lstm_46/while/lstm_cell_46/ReadVariableOp_32V
)lstm_46/while/lstm_cell_46/ReadVariableOp)lstm_46/while/lstm_cell_46/ReadVariableOp2b
/lstm_46/while/lstm_cell_46/split/ReadVariableOp/lstm_46/while/lstm_cell_46/split/ReadVariableOp2f
1lstm_46/while/lstm_cell_46/split_1/ReadVariableOp1lstm_46/while/lstm_cell_46/split_1/ReadVariableOp:
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
_user_specified_name" lstm_46/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_46/while/loop_counter
ип
к
'sequential_23_lstm_46_while_body_376905H
Dsequential_23_lstm_46_while_sequential_23_lstm_46_while_loop_counterN
Jsequential_23_lstm_46_while_sequential_23_lstm_46_while_maximum_iterations+
'sequential_23_lstm_46_while_placeholder-
)sequential_23_lstm_46_while_placeholder_1-
)sequential_23_lstm_46_while_placeholder_2-
)sequential_23_lstm_46_while_placeholder_3G
Csequential_23_lstm_46_while_sequential_23_lstm_46_strided_slice_1_0Г
sequential_23_lstm_46_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_46_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_23_lstm_46_while_lstm_cell_46_split_readvariableop_resource_0:	АY
Jsequential_23_lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0:	АV
Bsequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource_0:
АА(
$sequential_23_lstm_46_while_identity*
&sequential_23_lstm_46_while_identity_1*
&sequential_23_lstm_46_while_identity_2*
&sequential_23_lstm_46_while_identity_3*
&sequential_23_lstm_46_while_identity_4*
&sequential_23_lstm_46_while_identity_5E
Asequential_23_lstm_46_while_sequential_23_lstm_46_strided_slice_1Б
}sequential_23_lstm_46_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_46_tensorarrayunstack_tensorlistfromtensorY
Fsequential_23_lstm_46_while_lstm_cell_46_split_readvariableop_resource:	АW
Hsequential_23_lstm_46_while_lstm_cell_46_split_1_readvariableop_resource:	АT
@sequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource:
ААИв7sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOpв9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_1в9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_2в9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_3в=sequential_23/lstm_46/while/lstm_cell_46/split/ReadVariableOpв?sequential_23/lstm_46/while/lstm_cell_46/split_1/ReadVariableOpЮ
Msequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ф
?sequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_23_lstm_46_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_46_tensorarrayunstack_tensorlistfromtensor_0'sequential_23_lstm_46_while_placeholderVsequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0z
8sequential_23/lstm_46/while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╟
=sequential_23/lstm_46/while/lstm_cell_46/split/ReadVariableOpReadVariableOpHsequential_23_lstm_46_while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0Э
.sequential_23/lstm_46/while/lstm_cell_46/splitSplitAsequential_23/lstm_46/while/lstm_cell_46/split/split_dim:output:0Esequential_23/lstm_46/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitэ
/sequential_23/lstm_46/while/lstm_cell_46/MatMulMatMulFsequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_46/while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ая
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_1MatMulFsequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_46/while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ая
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_2MatMulFsequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_46/while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ая
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_3MatMulFsequential_23/lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_46/while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А|
:sequential_23/lstm_46/while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╟
?sequential_23/lstm_46/while/lstm_cell_46/split_1/ReadVariableOpReadVariableOpJsequential_23_lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0У
0sequential_23/lstm_46/while/lstm_cell_46/split_1SplitCsequential_23/lstm_46/while/lstm_cell_46/split_1/split_dim:output:0Gsequential_23/lstm_46/while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitф
0sequential_23/lstm_46/while/lstm_cell_46/BiasAddBiasAdd9sequential_23/lstm_46/while/lstm_cell_46/MatMul:product:09sequential_23/lstm_46/while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аш
2sequential_23/lstm_46/while/lstm_cell_46/BiasAdd_1BiasAdd;sequential_23/lstm_46/while/lstm_cell_46/MatMul_1:product:09sequential_23/lstm_46/while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аш
2sequential_23/lstm_46/while/lstm_cell_46/BiasAdd_2BiasAdd;sequential_23/lstm_46/while/lstm_cell_46/MatMul_2:product:09sequential_23/lstm_46/while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аш
2sequential_23/lstm_46/while/lstm_cell_46/BiasAdd_3BiasAdd;sequential_23/lstm_46/while/lstm_cell_46/MatMul_3:product:09sequential_23/lstm_46/while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         А╝
7sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOpReadVariableOpBsequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Н
<sequential_23/lstm_46/while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        П
>sequential_23/lstm_46/while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   П
>sequential_23/lstm_46/while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ║
6sequential_23/lstm_46/while/lstm_cell_46/strided_sliceStridedSlice?sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp:value:0Esequential_23/lstm_46/while/lstm_cell_46/strided_slice/stack:output:0Gsequential_23/lstm_46/while/lstm_cell_46/strided_slice/stack_1:output:0Gsequential_23/lstm_46/while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask┌
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_4MatMul)sequential_23_lstm_46_while_placeholder_2?sequential_23/lstm_46/while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         Ар
,sequential_23/lstm_46/while/lstm_cell_46/addAddV29sequential_23/lstm_46/while/lstm_cell_46/BiasAdd:output:0;sequential_23/lstm_46/while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         Аs
.sequential_23/lstm_46/while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_23/lstm_46/while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╤
,sequential_23/lstm_46/while/lstm_cell_46/MulMul0sequential_23/lstm_46/while/lstm_cell_46/add:z:07sequential_23/lstm_46/while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         А╫
.sequential_23/lstm_46/while/lstm_cell_46/Add_1AddV20sequential_23/lstm_46/while/lstm_cell_46/Mul:z:09sequential_23/lstm_46/while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         АЕ
@sequential_23/lstm_46/while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?√
>sequential_23/lstm_46/while/lstm_cell_46/clip_by_value/MinimumMinimum2sequential_23/lstm_46/while/lstm_cell_46/Add_1:z:0Isequential_23/lstm_46/while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         А}
8sequential_23/lstm_46/while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    √
6sequential_23/lstm_46/while/lstm_cell_46/clip_by_valueMaximumBsequential_23/lstm_46/while/lstm_cell_46/clip_by_value/Minimum:z:0Asequential_23/lstm_46/while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         А╛
9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_1ReadVariableOpBsequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0П
>sequential_23/lstm_46/while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   С
@sequential_23/lstm_46/while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       С
@sequential_23/lstm_46/while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ─
8sequential_23/lstm_46/while/lstm_cell_46/strided_slice_1StridedSliceAsequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_1:value:0Gsequential_23/lstm_46/while/lstm_cell_46/strided_slice_1/stack:output:0Isequential_23/lstm_46/while/lstm_cell_46/strided_slice_1/stack_1:output:0Isequential_23/lstm_46/while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▄
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_5MatMul)sequential_23_lstm_46_while_placeholder_2Asequential_23/lstm_46/while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Аф
.sequential_23/lstm_46/while/lstm_cell_46/add_2AddV2;sequential_23/lstm_46/while/lstm_cell_46/BiasAdd_1:output:0;sequential_23/lstm_46/while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         Аu
0sequential_23/lstm_46/while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_23/lstm_46/while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╫
.sequential_23/lstm_46/while/lstm_cell_46/Mul_1Mul2sequential_23/lstm_46/while/lstm_cell_46/add_2:z:09sequential_23/lstm_46/while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         А┘
.sequential_23/lstm_46/while/lstm_cell_46/Add_3AddV22sequential_23/lstm_46/while/lstm_cell_46/Mul_1:z:09sequential_23/lstm_46/while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         АЗ
Bsequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А? 
@sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1/MinimumMinimum2sequential_23/lstm_46/while/lstm_cell_46/Add_3:z:0Ksequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         А
:sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
8sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1MaximumDsequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum:z:0Csequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А╤
.sequential_23/lstm_46/while/lstm_cell_46/mul_2Mul<sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_1:z:0)sequential_23_lstm_46_while_placeholder_3*
T0*(
_output_shapes
:         А╛
9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_2ReadVariableOpBsequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0П
>sequential_23/lstm_46/while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       С
@sequential_23/lstm_46/while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  С
@sequential_23/lstm_46/while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ─
8sequential_23/lstm_46/while/lstm_cell_46/strided_slice_2StridedSliceAsequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_2:value:0Gsequential_23/lstm_46/while/lstm_cell_46/strided_slice_2/stack:output:0Isequential_23/lstm_46/while/lstm_cell_46/strided_slice_2/stack_1:output:0Isequential_23/lstm_46/while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▄
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_6MatMul)sequential_23_lstm_46_while_placeholder_2Asequential_23/lstm_46/while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Аф
.sequential_23/lstm_46/while/lstm_cell_46/add_4AddV2;sequential_23/lstm_46/while/lstm_cell_46/BiasAdd_2:output:0;sequential_23/lstm_46/while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         АЬ
-sequential_23/lstm_46/while/lstm_cell_46/ReluRelu2sequential_23/lstm_46/while/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         Ас
.sequential_23/lstm_46/while/lstm_cell_46/mul_3Mul:sequential_23/lstm_46/while/lstm_cell_46/clip_by_value:z:0;sequential_23/lstm_46/while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А╥
.sequential_23/lstm_46/while/lstm_cell_46/add_5AddV22sequential_23/lstm_46/while/lstm_cell_46/mul_2:z:02sequential_23/lstm_46/while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         А╛
9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_3ReadVariableOpBsequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0П
>sequential_23/lstm_46/while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  С
@sequential_23/lstm_46/while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        С
@sequential_23/lstm_46/while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ─
8sequential_23/lstm_46/while/lstm_cell_46/strided_slice_3StridedSliceAsequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_3:value:0Gsequential_23/lstm_46/while/lstm_cell_46/strided_slice_3/stack:output:0Isequential_23/lstm_46/while/lstm_cell_46/strided_slice_3/stack_1:output:0Isequential_23/lstm_46/while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▄
1sequential_23/lstm_46/while/lstm_cell_46/MatMul_7MatMul)sequential_23_lstm_46_while_placeholder_2Asequential_23/lstm_46/while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Аф
.sequential_23/lstm_46/while/lstm_cell_46/add_6AddV2;sequential_23/lstm_46/while/lstm_cell_46/BiasAdd_3:output:0;sequential_23/lstm_46/while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         Аu
0sequential_23/lstm_46/while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_23/lstm_46/while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╫
.sequential_23/lstm_46/while/lstm_cell_46/Mul_4Mul2sequential_23/lstm_46/while/lstm_cell_46/add_6:z:09sequential_23/lstm_46/while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         А┘
.sequential_23/lstm_46/while/lstm_cell_46/Add_7AddV22sequential_23/lstm_46/while/lstm_cell_46/Mul_4:z:09sequential_23/lstm_46/while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         АЗ
Bsequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А? 
@sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2/MinimumMinimum2sequential_23/lstm_46/while/lstm_cell_46/Add_7:z:0Ksequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         А
:sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
8sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2MaximumDsequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum:z:0Csequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АЮ
/sequential_23/lstm_46/while/lstm_cell_46/Relu_1Relu2sequential_23/lstm_46/while/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Ах
.sequential_23/lstm_46/while/lstm_cell_46/mul_5Mul<sequential_23/lstm_46/while/lstm_cell_46/clip_by_value_2:z:0=sequential_23/lstm_46/while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         АЭ
@sequential_23/lstm_46/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_23_lstm_46_while_placeholder_1'sequential_23_lstm_46_while_placeholder2sequential_23/lstm_46/while/lstm_cell_46/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥c
!sequential_23/lstm_46/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ю
sequential_23/lstm_46/while/addAddV2'sequential_23_lstm_46_while_placeholder*sequential_23/lstm_46/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_23/lstm_46/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_23/lstm_46/while/add_1AddV2Dsequential_23_lstm_46_while_sequential_23_lstm_46_while_loop_counter,sequential_23/lstm_46/while/add_1/y:output:0*
T0*
_output_shapes
: Ы
$sequential_23/lstm_46/while/IdentityIdentity%sequential_23/lstm_46/while/add_1:z:0!^sequential_23/lstm_46/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_23/lstm_46/while/Identity_1IdentityJsequential_23_lstm_46_while_sequential_23_lstm_46_while_maximum_iterations!^sequential_23/lstm_46/while/NoOp*
T0*
_output_shapes
: Ы
&sequential_23/lstm_46/while/Identity_2Identity#sequential_23/lstm_46/while/add:z:0!^sequential_23/lstm_46/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_23/lstm_46/while/Identity_3IdentityPsequential_23/lstm_46/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_23/lstm_46/while/NoOp*
T0*
_output_shapes
: ╝
&sequential_23/lstm_46/while/Identity_4Identity2sequential_23/lstm_46/while/lstm_cell_46/mul_5:z:0!^sequential_23/lstm_46/while/NoOp*
T0*(
_output_shapes
:         А╝
&sequential_23/lstm_46/while/Identity_5Identity2sequential_23/lstm_46/while/lstm_cell_46/add_5:z:0!^sequential_23/lstm_46/while/NoOp*
T0*(
_output_shapes
:         А╥
 sequential_23/lstm_46/while/NoOpNoOp8^sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp:^sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_1:^sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_2:^sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_3>^sequential_23/lstm_46/while/lstm_cell_46/split/ReadVariableOp@^sequential_23/lstm_46/while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Y
&sequential_23_lstm_46_while_identity_1/sequential_23/lstm_46/while/Identity_1:output:0"Y
&sequential_23_lstm_46_while_identity_2/sequential_23/lstm_46/while/Identity_2:output:0"Y
&sequential_23_lstm_46_while_identity_3/sequential_23/lstm_46/while/Identity_3:output:0"Y
&sequential_23_lstm_46_while_identity_4/sequential_23/lstm_46/while/Identity_4:output:0"Y
&sequential_23_lstm_46_while_identity_5/sequential_23/lstm_46/while/Identity_5:output:0"U
$sequential_23_lstm_46_while_identity-sequential_23/lstm_46/while/Identity:output:0"Ж
@sequential_23_lstm_46_while_lstm_cell_46_readvariableop_resourceBsequential_23_lstm_46_while_lstm_cell_46_readvariableop_resource_0"Ц
Hsequential_23_lstm_46_while_lstm_cell_46_split_1_readvariableop_resourceJsequential_23_lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0"Т
Fsequential_23_lstm_46_while_lstm_cell_46_split_readvariableop_resourceHsequential_23_lstm_46_while_lstm_cell_46_split_readvariableop_resource_0"И
Asequential_23_lstm_46_while_sequential_23_lstm_46_strided_slice_1Csequential_23_lstm_46_while_sequential_23_lstm_46_strided_slice_1_0"А
}sequential_23_lstm_46_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_46_tensorarrayunstack_tensorlistfromtensorsequential_23_lstm_46_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_46_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2v
9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_19sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_12v
9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_29sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_22v
9sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_39sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp_32r
7sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp7sequential_23/lstm_46/while/lstm_cell_46/ReadVariableOp2~
=sequential_23/lstm_46/while/lstm_cell_46/split/ReadVariableOp=sequential_23/lstm_46/while/lstm_cell_46/split/ReadVariableOp2В
?sequential_23/lstm_46/while/lstm_cell_46/split_1/ReadVariableOp?sequential_23/lstm_46/while/lstm_cell_46/split_1/ReadVariableOp:
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
_user_specified_name0.sequential_23/lstm_46/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_23/lstm_46/while/loop_counter
Ч	
├
while_cond_378934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_378934___redundant_placeholder04
0while_while_cond_378934___redundant_placeholder14
0while_while_cond_378934___redundant_placeholder24
0while_while_cond_378934___redundant_placeholder3
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
ы}
ж	
while_body_378612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_47_split_readvariableop_resource_0:
ААC
4while_lstm_cell_47_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_47_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_47_split_readvariableop_resource:
ААA
2while_lstm_cell_47_split_1_readvariableop_resource:	А=
*while_lstm_cell_47_readvariableop_resource:	@АИв!while/lstm_cell_47/ReadVariableOpв#while/lstm_cell_47/ReadVariableOp_1в#while/lstm_cell_47/ReadVariableOp_2в#while/lstm_cell_47/ReadVariableOp_3в'while/lstm_cell_47/split/ReadVariableOpв)while/lstm_cell_47/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_47/split/ReadVariableOpReadVariableOp2while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_47/splitSplit+while/lstm_cell_47/split/split_dim:output:0/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_47/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_47/split_1Split-while/lstm_cell_47/split_1/split_dim:output:01while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_47/BiasAddBiasAdd#while/lstm_cell_47/MatMul:product:0#while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_1BiasAdd%while/lstm_cell_47/MatMul_1:product:0#while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_2BiasAdd%while/lstm_cell_47/MatMul_2:product:0#while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_3BiasAdd%while/lstm_cell_47/MatMul_3:product:0#while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_47/ReadVariableOpReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_47/strided_sliceStridedSlice)while/lstm_cell_47/ReadVariableOp:value:0/while/lstm_cell_47/strided_slice/stack:output:01while/lstm_cell_47/strided_slice/stack_1:output:01while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_47/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_47/addAddV2#while/lstm_cell_47/BiasAdd:output:0%while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_47/MulMulwhile/lstm_cell_47/add:z:0!while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_47/Add_1AddV2while/lstm_cell_47/Mul:z:0#while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_47/clip_by_value/MinimumMinimumwhile/lstm_cell_47/Add_1:z:03while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_47/clip_by_valueMaximum,while/lstm_cell_47/clip_by_value/Minimum:z:0+while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_1ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_1StridedSlice+while/lstm_cell_47/ReadVariableOp_1:value:01while/lstm_cell_47/strided_slice_1/stack:output:03while/lstm_cell_47/strided_slice_1/stack_1:output:03while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_2AddV2%while/lstm_cell_47/BiasAdd_1:output:0%while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_1Mulwhile/lstm_cell_47/add_2:z:0#while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_3AddV2while/lstm_cell_47/Mul_1:z:0#while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_1/MinimumMinimumwhile/lstm_cell_47/Add_3:z:05while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_1Maximum.while/lstm_cell_47/clip_by_value_1/Minimum:z:0-while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_47/mul_2Mul&while/lstm_cell_47/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_2ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_2StridedSlice+while/lstm_cell_47/ReadVariableOp_2:value:01while/lstm_cell_47/strided_slice_2/stack:output:03while/lstm_cell_47/strided_slice_2/stack_1:output:03while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_4AddV2%while/lstm_cell_47/BiasAdd_2:output:0%while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_47/ReluReluwhile/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_47/mul_3Mul$while/lstm_cell_47/clip_by_value:z:0%while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_47/add_5AddV2while/lstm_cell_47/mul_2:z:0while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_3ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_3StridedSlice+while/lstm_cell_47/ReadVariableOp_3:value:01while/lstm_cell_47/strided_slice_3/stack:output:03while/lstm_cell_47/strided_slice_3/stack_1:output:03while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_6AddV2%while/lstm_cell_47/BiasAdd_3:output:0%while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_4Mulwhile/lstm_cell_47/add_6:z:0#while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_7AddV2while/lstm_cell_47/Mul_4:z:0#while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_2/MinimumMinimumwhile/lstm_cell_47/Add_7:z:05while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_2Maximum.while/lstm_cell_47/clip_by_value_2/Minimum:z:0-while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_47/Relu_1Reluwhile/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_47/mul_5Mul&while/lstm_cell_47/clip_by_value_2:z:0'while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_47/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_47/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_47/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_47/ReadVariableOp$^while/lstm_cell_47/ReadVariableOp_1$^while/lstm_cell_47/ReadVariableOp_2$^while/lstm_cell_47/ReadVariableOp_3(^while/lstm_cell_47/split/ReadVariableOp*^while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_47_readvariableop_resource,while_lstm_cell_47_readvariableop_resource_0"j
2while_lstm_cell_47_split_1_readvariableop_resource4while_lstm_cell_47_split_1_readvariableop_resource_0"f
0while_lstm_cell_47_split_readvariableop_resource2while_lstm_cell_47_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_47/ReadVariableOp_1#while/lstm_cell_47/ReadVariableOp_12J
#while/lstm_cell_47/ReadVariableOp_2#while/lstm_cell_47/ReadVariableOp_22J
#while/lstm_cell_47/ReadVariableOp_3#while/lstm_cell_47/ReadVariableOp_32F
!while/lstm_cell_47/ReadVariableOp!while/lstm_cell_47/ReadVariableOp2R
'while/lstm_cell_47/split/ReadVariableOp'while/lstm_cell_47/split/ReadVariableOp2V
)while/lstm_cell_47/split_1/ReadVariableOp)while/lstm_cell_47/split_1/ReadVariableOp:
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
с7
Ж
C__inference_lstm_46_layer_call_and_return_conditional_losses_377509

inputs&
lstm_cell_46_377428:	А"
lstm_cell_46_377430:	А'
lstm_cell_46_377432:
АА
identityИв$lstm_cell_46/StatefulPartitionedCallвwhileI
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
$lstm_cell_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_46_377428lstm_cell_46_377430lstm_cell_46_377432*
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377427n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_46_377428lstm_cell_46_377430lstm_cell_46_377432*
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
while_body_377441*
condR
while_cond_377440*M
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
NoOpNoOp%^lstm_cell_46/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_46/StatefulPartitionedCall$lstm_cell_46/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
иИ
ш
C__inference_lstm_47_layer_call_and_return_conditional_losses_378752

inputs>
*lstm_cell_47_split_readvariableop_resource:
АА;
,lstm_cell_47_split_1_readvariableop_resource:	А7
$lstm_cell_47_readvariableop_resource:	@А
identityИвlstm_cell_47/ReadVariableOpвlstm_cell_47/ReadVariableOp_1вlstm_cell_47/ReadVariableOp_2вlstm_cell_47/ReadVariableOp_3в!lstm_cell_47/split/ReadVariableOpв#lstm_cell_47/split_1/ReadVariableOpвwhileI
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
:
         АR
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
lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_47/split/ReadVariableOpReadVariableOp*lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_47/splitSplit%lstm_cell_47/split/split_dim:output:0)lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_47/MatMulMatMulstrided_slice_2:output:0lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_47/split_1/ReadVariableOpReadVariableOp,lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_47/split_1Split'lstm_cell_47/split_1/split_dim:output:0+lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_47/BiasAddBiasAddlstm_cell_47/MatMul:product:0lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_1BiasAddlstm_cell_47/MatMul_1:product:0lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_2BiasAddlstm_cell_47/MatMul_2:product:0lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_3BiasAddlstm_cell_47/MatMul_3:product:0lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_47/ReadVariableOpReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_47/strided_sliceStridedSlice#lstm_cell_47/ReadVariableOp:value:0)lstm_cell_47/strided_slice/stack:output:0+lstm_cell_47/strided_slice/stack_1:output:0+lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_47/MatMul_4MatMulzeros:output:0#lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_47/addAddV2lstm_cell_47/BiasAdd:output:0lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_47/MulMullstm_cell_47/add:z:0lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_47/Add_1AddV2lstm_cell_47/Mul:z:0lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_47/clip_by_value/MinimumMinimumlstm_cell_47/Add_1:z:0-lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_47/clip_by_valueMaximum&lstm_cell_47/clip_by_value/Minimum:z:0%lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_1ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_1StridedSlice%lstm_cell_47/ReadVariableOp_1:value:0+lstm_cell_47/strided_slice_1/stack:output:0-lstm_cell_47/strided_slice_1/stack_1:output:0-lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_5MatMulzeros:output:0%lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_2AddV2lstm_cell_47/BiasAdd_1:output:0lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_1Mullstm_cell_47/add_2:z:0lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_3AddV2lstm_cell_47/Mul_1:z:0lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_cell_47/Add_3:z:0/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_1Maximum(lstm_cell_47/clip_by_value_1/Minimum:z:0'lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_47/mul_2Mul lstm_cell_47/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_2ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_2StridedSlice%lstm_cell_47/ReadVariableOp_2:value:0+lstm_cell_47/strided_slice_2/stack:output:0-lstm_cell_47/strided_slice_2/stack_1:output:0-lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_6MatMulzeros:output:0%lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_4AddV2lstm_cell_47/BiasAdd_2:output:0lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/ReluRelulstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_47/mul_3Mullstm_cell_47/clip_by_value:z:0lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_47/add_5AddV2lstm_cell_47/mul_2:z:0lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_3ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_3StridedSlice%lstm_cell_47/ReadVariableOp_3:value:0+lstm_cell_47/strided_slice_3/stack:output:0-lstm_cell_47/strided_slice_3/stack_1:output:0-lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_7MatMulzeros:output:0%lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_6AddV2lstm_cell_47/BiasAdd_3:output:0lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_4Mullstm_cell_47/add_6:z:0lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_7AddV2lstm_cell_47/Mul_4:z:0lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_cell_47/Add_7:z:0/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_2Maximum(lstm_cell_47/clip_by_value_2/Minimum:z:0'lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_47/Relu_1Relulstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_47/mul_5Mul lstm_cell_47/clip_by_value_2:z:0!lstm_cell_47/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_47_split_readvariableop_resource,lstm_cell_47_split_1_readvariableop_resource$lstm_cell_47_readvariableop_resource*
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
while_body_378612*
condR
while_cond_378611*K
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
:
         @*
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
:         
@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_47/ReadVariableOp^lstm_cell_47/ReadVariableOp_1^lstm_cell_47/ReadVariableOp_2^lstm_cell_47/ReadVariableOp_3"^lstm_cell_47/split/ReadVariableOp$^lstm_cell_47/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         
А: : : 2>
lstm_cell_47/ReadVariableOp_1lstm_cell_47/ReadVariableOp_12>
lstm_cell_47/ReadVariableOp_2lstm_cell_47/ReadVariableOp_22>
lstm_cell_47/ReadVariableOp_3lstm_cell_47/ReadVariableOp_32:
lstm_cell_47/ReadVariableOplstm_cell_47/ReadVariableOp2F
!lstm_cell_47/split/ReadVariableOp!lstm_cell_47/split/ReadVariableOp2J
#lstm_cell_47/split_1/ReadVariableOp#lstm_cell_47/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         
А
 
_user_specified_nameinputs
Ы	
├
while_cond_381008
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_381008___redundant_placeholder04
0while_while_cond_381008___redundant_placeholder14
0while_while_cond_381008___redundant_placeholder24
0while_while_cond_381008___redundant_placeholder3
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
lstm_47_while_cond_379932,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3.
*lstm_47_while_less_lstm_47_strided_slice_1D
@lstm_47_while_lstm_47_while_cond_379932___redundant_placeholder0D
@lstm_47_while_lstm_47_while_cond_379932___redundant_placeholder1D
@lstm_47_while_lstm_47_while_cond_379932___redundant_placeholder2D
@lstm_47_while_lstm_47_while_cond_379932___redundant_placeholder3
lstm_47_while_identity
В
lstm_47/while/LessLesslstm_47_while_placeholder*lstm_47_while_less_lstm_47_strided_slice_1*
T0*
_output_shapes
: [
lstm_47/while/IdentityIdentitylstm_47/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_47_while_identitylstm_47/while/Identity:output:0*(
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
_user_specified_name" lstm_47/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_47/while/loop_counter
б~
ж	
while_body_381521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_46_split_readvariableop_resource_0:	АC
4while_lstm_cell_46_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_46_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_46_split_readvariableop_resource:	АA
2while_lstm_cell_46_split_1_readvariableop_resource:	А>
*while_lstm_cell_46_readvariableop_resource:
ААИв!while/lstm_cell_46/ReadVariableOpв#while/lstm_cell_46/ReadVariableOp_1в#while/lstm_cell_46/ReadVariableOp_2в#while/lstm_cell_46/ReadVariableOp_3в'while/lstm_cell_46/split/ReadVariableOpв)while/lstm_cell_46/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_46/split/ReadVariableOpReadVariableOp2while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_46/splitSplit+while/lstm_cell_46/split/split_dim:output:0/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_46/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_46/split_1Split-while/lstm_cell_46/split_1/split_dim:output:01while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_46/BiasAddBiasAdd#while/lstm_cell_46/MatMul:product:0#while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_1BiasAdd%while/lstm_cell_46/MatMul_1:product:0#while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_2BiasAdd%while/lstm_cell_46/MatMul_2:product:0#while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_3BiasAdd%while/lstm_cell_46/MatMul_3:product:0#while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_46/ReadVariableOpReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_46/strided_sliceStridedSlice)while/lstm_cell_46/ReadVariableOp:value:0/while/lstm_cell_46/strided_slice/stack:output:01while/lstm_cell_46/strided_slice/stack_1:output:01while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_46/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_46/addAddV2#while/lstm_cell_46/BiasAdd:output:0%while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_46/MulMulwhile/lstm_cell_46/add:z:0!while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_46/Add_1AddV2while/lstm_cell_46/Mul:z:0#while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_46/clip_by_value/MinimumMinimumwhile/lstm_cell_46/Add_1:z:03while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_46/clip_by_valueMaximum,while/lstm_cell_46/clip_by_value/Minimum:z:0+while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_1ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_1StridedSlice+while/lstm_cell_46/ReadVariableOp_1:value:01while/lstm_cell_46/strided_slice_1/stack:output:03while/lstm_cell_46/strided_slice_1/stack_1:output:03while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_2AddV2%while/lstm_cell_46/BiasAdd_1:output:0%while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_1Mulwhile/lstm_cell_46/add_2:z:0#while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_3AddV2while/lstm_cell_46/Mul_1:z:0#while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_1/MinimumMinimumwhile/lstm_cell_46/Add_3:z:05while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_1Maximum.while/lstm_cell_46/clip_by_value_1/Minimum:z:0-while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_46/mul_2Mul&while/lstm_cell_46/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_2ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_2StridedSlice+while/lstm_cell_46/ReadVariableOp_2:value:01while/lstm_cell_46/strided_slice_2/stack:output:03while/lstm_cell_46/strided_slice_2/stack_1:output:03while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_4AddV2%while/lstm_cell_46/BiasAdd_2:output:0%while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_46/ReluReluwhile/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_46/mul_3Mul$while/lstm_cell_46/clip_by_value:z:0%while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_46/add_5AddV2while/lstm_cell_46/mul_2:z:0while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_3ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_3StridedSlice+while/lstm_cell_46/ReadVariableOp_3:value:01while/lstm_cell_46/strided_slice_3/stack:output:03while/lstm_cell_46/strided_slice_3/stack_1:output:03while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_6AddV2%while/lstm_cell_46/BiasAdd_3:output:0%while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_4Mulwhile/lstm_cell_46/add_6:z:0#while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_7AddV2while/lstm_cell_46/Mul_4:z:0#while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_2/MinimumMinimumwhile/lstm_cell_46/Add_7:z:05while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_2Maximum.while/lstm_cell_46/clip_by_value_2/Minimum:z:0-while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_46/Relu_1Reluwhile/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_46/mul_5Mul&while/lstm_cell_46/clip_by_value_2:z:0'while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_46/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_46/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_46/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_46/ReadVariableOp$^while/lstm_cell_46/ReadVariableOp_1$^while/lstm_cell_46/ReadVariableOp_2$^while/lstm_cell_46/ReadVariableOp_3(^while/lstm_cell_46/split/ReadVariableOp*^while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_46_readvariableop_resource,while_lstm_cell_46_readvariableop_resource_0"j
2while_lstm_cell_46_split_1_readvariableop_resource4while_lstm_cell_46_split_1_readvariableop_resource_0"f
0while_lstm_cell_46_split_readvariableop_resource2while_lstm_cell_46_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_46/ReadVariableOp_1#while/lstm_cell_46/ReadVariableOp_12J
#while/lstm_cell_46/ReadVariableOp_2#while/lstm_cell_46/ReadVariableOp_22J
#while/lstm_cell_46/ReadVariableOp_3#while/lstm_cell_46/ReadVariableOp_32F
!while/lstm_cell_46/ReadVariableOp!while/lstm_cell_46/ReadVariableOp2R
'while/lstm_cell_46/split/ReadVariableOp'while/lstm_cell_46/split/ReadVariableOp2V
)while/lstm_cell_46/split_1/ReadVariableOp)while/lstm_cell_46/split_1/ReadVariableOp:
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383172

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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382871

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
·
╢
(__inference_lstm_47_layer_call_fn_381694

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
C__inference_lstm_47_layer_call_and_return_conditional_losses_378752o
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
:         
А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         
А
 
_user_specified_nameinputs
▄	
╔
.__inference_sequential_23_layer_call_fn_379565

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
I__inference_sequential_23_layer_call_and_return_conditional_losses_379408o
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
':         
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ы}
ж	
while_body_382589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_47_split_readvariableop_resource_0:
ААC
4while_lstm_cell_47_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_47_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_47_split_readvariableop_resource:
ААA
2while_lstm_cell_47_split_1_readvariableop_resource:	А=
*while_lstm_cell_47_readvariableop_resource:	@АИв!while/lstm_cell_47/ReadVariableOpв#while/lstm_cell_47/ReadVariableOp_1в#while/lstm_cell_47/ReadVariableOp_2в#while/lstm_cell_47/ReadVariableOp_3в'while/lstm_cell_47/split/ReadVariableOpв)while/lstm_cell_47/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_47/split/ReadVariableOpReadVariableOp2while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_47/splitSplit+while/lstm_cell_47/split/split_dim:output:0/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_47/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_47/split_1Split-while/lstm_cell_47/split_1/split_dim:output:01while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_47/BiasAddBiasAdd#while/lstm_cell_47/MatMul:product:0#while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_1BiasAdd%while/lstm_cell_47/MatMul_1:product:0#while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_2BiasAdd%while/lstm_cell_47/MatMul_2:product:0#while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_3BiasAdd%while/lstm_cell_47/MatMul_3:product:0#while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_47/ReadVariableOpReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_47/strided_sliceStridedSlice)while/lstm_cell_47/ReadVariableOp:value:0/while/lstm_cell_47/strided_slice/stack:output:01while/lstm_cell_47/strided_slice/stack_1:output:01while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_47/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_47/addAddV2#while/lstm_cell_47/BiasAdd:output:0%while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_47/MulMulwhile/lstm_cell_47/add:z:0!while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_47/Add_1AddV2while/lstm_cell_47/Mul:z:0#while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_47/clip_by_value/MinimumMinimumwhile/lstm_cell_47/Add_1:z:03while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_47/clip_by_valueMaximum,while/lstm_cell_47/clip_by_value/Minimum:z:0+while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_1ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_1StridedSlice+while/lstm_cell_47/ReadVariableOp_1:value:01while/lstm_cell_47/strided_slice_1/stack:output:03while/lstm_cell_47/strided_slice_1/stack_1:output:03while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_2AddV2%while/lstm_cell_47/BiasAdd_1:output:0%while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_1Mulwhile/lstm_cell_47/add_2:z:0#while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_3AddV2while/lstm_cell_47/Mul_1:z:0#while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_1/MinimumMinimumwhile/lstm_cell_47/Add_3:z:05while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_1Maximum.while/lstm_cell_47/clip_by_value_1/Minimum:z:0-while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_47/mul_2Mul&while/lstm_cell_47/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_2ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_2StridedSlice+while/lstm_cell_47/ReadVariableOp_2:value:01while/lstm_cell_47/strided_slice_2/stack:output:03while/lstm_cell_47/strided_slice_2/stack_1:output:03while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_4AddV2%while/lstm_cell_47/BiasAdd_2:output:0%while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_47/ReluReluwhile/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_47/mul_3Mul$while/lstm_cell_47/clip_by_value:z:0%while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_47/add_5AddV2while/lstm_cell_47/mul_2:z:0while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_3ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_3StridedSlice+while/lstm_cell_47/ReadVariableOp_3:value:01while/lstm_cell_47/strided_slice_3/stack:output:03while/lstm_cell_47/strided_slice_3/stack_1:output:03while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_6AddV2%while/lstm_cell_47/BiasAdd_3:output:0%while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_4Mulwhile/lstm_cell_47/add_6:z:0#while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_7AddV2while/lstm_cell_47/Mul_4:z:0#while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_2/MinimumMinimumwhile/lstm_cell_47/Add_7:z:05while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_2Maximum.while/lstm_cell_47/clip_by_value_2/Minimum:z:0-while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_47/Relu_1Reluwhile/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_47/mul_5Mul&while/lstm_cell_47/clip_by_value_2:z:0'while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_47/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_47/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_47/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_47/ReadVariableOp$^while/lstm_cell_47/ReadVariableOp_1$^while/lstm_cell_47/ReadVariableOp_2$^while/lstm_cell_47/ReadVariableOp_3(^while/lstm_cell_47/split/ReadVariableOp*^while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_47_readvariableop_resource,while_lstm_cell_47_readvariableop_resource_0"j
2while_lstm_cell_47_split_1_readvariableop_resource4while_lstm_cell_47_split_1_readvariableop_resource_0"f
0while_lstm_cell_47_split_readvariableop_resource2while_lstm_cell_47_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_47/ReadVariableOp_1#while/lstm_cell_47/ReadVariableOp_12J
#while/lstm_cell_47/ReadVariableOp_2#while/lstm_cell_47/ReadVariableOp_22J
#while/lstm_cell_47/ReadVariableOp_3#while/lstm_cell_47/ReadVariableOp_32F
!while/lstm_cell_47/ReadVariableOp!while/lstm_cell_47/ReadVariableOp2R
'while/lstm_cell_47/split/ReadVariableOp'while/lstm_cell_47/split/ReadVariableOp2V
)while/lstm_cell_47/split_1/ReadVariableOp)while/lstm_cell_47/split_1/ReadVariableOp:
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
Ё
▌
I__inference_sequential_23_layer_call_and_return_conditional_losses_379408

inputs!
lstm_46_379388:	А
lstm_46_379390:	А"
lstm_46_379392:
АА"
lstm_47_379395:
АА
lstm_47_379397:	А!
lstm_47_379399:	@А!
dense_23_379402:@
dense_23_379404:
identityИв dense_23/StatefulPartitionedCallвlstm_46/StatefulPartitionedCallвlstm_47/StatefulPartitionedCallГ
lstm_46/StatefulPartitionedCallStatefulPartitionedCallinputslstm_46_379388lstm_46_379390lstm_46_379392*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         
А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_46_layer_call_and_return_conditional_losses_379353а
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_379395lstm_47_379397lstm_47_379399*
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_379075Т
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_23_379402dense_23_379404*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_378770x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_23/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
МK
к
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377629

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
м
╕
(__inference_lstm_46_layer_call_fn_380604
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_377509}
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
№╕
╩	
I__inference_sequential_23_layer_call_and_return_conditional_losses_380593

inputsE
2lstm_46_lstm_cell_46_split_readvariableop_resource:	АC
4lstm_46_lstm_cell_46_split_1_readvariableop_resource:	А@
,lstm_46_lstm_cell_46_readvariableop_resource:
ААF
2lstm_47_lstm_cell_47_split_readvariableop_resource:
ААC
4lstm_47_lstm_cell_47_split_1_readvariableop_resource:	А?
,lstm_47_lstm_cell_47_readvariableop_resource:	@А9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:
identityИвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpв#lstm_46/lstm_cell_46/ReadVariableOpв%lstm_46/lstm_cell_46/ReadVariableOp_1в%lstm_46/lstm_cell_46/ReadVariableOp_2в%lstm_46/lstm_cell_46/ReadVariableOp_3в)lstm_46/lstm_cell_46/split/ReadVariableOpв+lstm_46/lstm_cell_46/split_1/ReadVariableOpвlstm_46/whileв#lstm_47/lstm_cell_47/ReadVariableOpв%lstm_47/lstm_cell_47/ReadVariableOp_1в%lstm_47/lstm_cell_47/ReadVariableOp_2в%lstm_47/lstm_cell_47/ReadVariableOp_3в)lstm_47/lstm_cell_47/split/ReadVariableOpв+lstm_47/lstm_cell_47/split_1/ReadVariableOpвlstm_47/whileQ
lstm_46/ShapeShapeinputs*
T0*
_output_shapes
::э╧e
lstm_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_46/strided_sliceStridedSlicelstm_46/Shape:output:0$lstm_46/strided_slice/stack:output:0&lstm_46/strided_slice/stack_1:output:0&lstm_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_46/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АЛ
lstm_46/zeros/packedPacklstm_46/strided_slice:output:0lstm_46/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_46/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
lstm_46/zerosFilllstm_46/zeros/packed:output:0lstm_46/zeros/Const:output:0*
T0*(
_output_shapes
:         А[
lstm_46/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АП
lstm_46/zeros_1/packedPacklstm_46/strided_slice:output:0!lstm_46/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_46/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
lstm_46/zeros_1Filllstm_46/zeros_1/packed:output:0lstm_46/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аk
lstm_46/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_46/transpose	Transposeinputslstm_46/transpose/perm:output:0*
T0*+
_output_shapes
:
         b
lstm_46/Shape_1Shapelstm_46/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_46/strided_slice_1StridedSlicelstm_46/Shape_1:output:0&lstm_46/strided_slice_1/stack:output:0(lstm_46/strided_slice_1/stack_1:output:0(lstm_46/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_46/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_46/TensorArrayV2TensorListReserve,lstm_46/TensorArrayV2/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       °
/lstm_46/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_46/transpose:y:0Flstm_46/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
lstm_46/strided_slice_2StridedSlicelstm_46/transpose:y:0&lstm_46/strided_slice_2/stack:output:0(lstm_46/strided_slice_2/stack_1:output:0(lstm_46/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskf
$lstm_46/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Э
)lstm_46/lstm_cell_46/split/ReadVariableOpReadVariableOp2lstm_46_lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0с
lstm_46/lstm_cell_46/splitSplit-lstm_46/lstm_cell_46/split/split_dim:output:01lstm_46/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЯ
lstm_46/lstm_cell_46/MatMulMatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Аб
lstm_46/lstm_cell_46/MatMul_1MatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Аб
lstm_46/lstm_cell_46/MatMul_2MatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Аб
lstm_46/lstm_cell_46/MatMul_3MatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аh
&lstm_46/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_46/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4lstm_46_lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╫
lstm_46/lstm_cell_46/split_1Split/lstm_46/lstm_cell_46/split_1/split_dim:output:03lstm_46/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitи
lstm_46/lstm_cell_46/BiasAddBiasAdd%lstm_46/lstm_cell_46/MatMul:product:0%lstm_46/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Ам
lstm_46/lstm_cell_46/BiasAdd_1BiasAdd'lstm_46/lstm_cell_46/MatMul_1:product:0%lstm_46/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Ам
lstm_46/lstm_cell_46/BiasAdd_2BiasAdd'lstm_46/lstm_cell_46/MatMul_2:product:0%lstm_46/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Ам
lstm_46/lstm_cell_46/BiasAdd_3BiasAdd'lstm_46/lstm_cell_46/MatMul_3:product:0%lstm_46/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АТ
#lstm_46/lstm_cell_46/ReadVariableOpReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0y
(lstm_46/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_46/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*lstm_46/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"lstm_46/lstm_cell_46/strided_sliceStridedSlice+lstm_46/lstm_cell_46/ReadVariableOp:value:01lstm_46/lstm_cell_46/strided_slice/stack:output:03lstm_46/lstm_cell_46/strided_slice/stack_1:output:03lstm_46/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЯ
lstm_46/lstm_cell_46/MatMul_4MatMullstm_46/zeros:output:0+lstm_46/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         Ад
lstm_46/lstm_cell_46/addAddV2%lstm_46/lstm_cell_46/BiasAdd:output:0'lstm_46/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А_
lstm_46/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_46/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
lstm_46/lstm_cell_46/MulMullstm_46/lstm_cell_46/add:z:0#lstm_46/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АЫ
lstm_46/lstm_cell_46/Add_1AddV2lstm_46/lstm_cell_46/Mul:z:0%lstm_46/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аq
,lstm_46/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┐
*lstm_46/lstm_cell_46/clip_by_value/MinimumMinimumlstm_46/lstm_cell_46/Add_1:z:05lstm_46/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$lstm_46/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"lstm_46/lstm_cell_46/clip_by_valueMaximum.lstm_46/lstm_cell_46/clip_by_value/Minimum:z:0-lstm_46/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АФ
%lstm_46/lstm_cell_46/ReadVariableOp_1ReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_46/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_46/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_46/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_46/lstm_cell_46/strided_slice_1StridedSlice-lstm_46/lstm_cell_46/ReadVariableOp_1:value:03lstm_46/lstm_cell_46/strided_slice_1/stack:output:05lstm_46/lstm_cell_46/strided_slice_1/stack_1:output:05lstm_46/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_46/lstm_cell_46/MatMul_5MatMullstm_46/zeros:output:0-lstm_46/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Аи
lstm_46/lstm_cell_46/add_2AddV2'lstm_46/lstm_cell_46/BiasAdd_1:output:0'lstm_46/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         Аa
lstm_46/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_46/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_46/lstm_cell_46/Mul_1Mullstm_46/lstm_cell_46/add_2:z:0%lstm_46/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЭ
lstm_46/lstm_cell_46/Add_3AddV2lstm_46/lstm_cell_46/Mul_1:z:0%lstm_46/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аs
.lstm_46/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_46/lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_46/lstm_cell_46/Add_3:z:07lstm_46/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_46/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_46/lstm_cell_46/clip_by_value_1Maximum0lstm_46/lstm_cell_46/clip_by_value_1/Minimum:z:0/lstm_46/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АШ
lstm_46/lstm_cell_46/mul_2Mul(lstm_46/lstm_cell_46/clip_by_value_1:z:0lstm_46/zeros_1:output:0*
T0*(
_output_shapes
:         АФ
%lstm_46/lstm_cell_46/ReadVariableOp_2ReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_46/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_46/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_46/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_46/lstm_cell_46/strided_slice_2StridedSlice-lstm_46/lstm_cell_46/ReadVariableOp_2:value:03lstm_46/lstm_cell_46/strided_slice_2/stack:output:05lstm_46/lstm_cell_46/strided_slice_2/stack_1:output:05lstm_46/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_46/lstm_cell_46/MatMul_6MatMullstm_46/zeros:output:0-lstm_46/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Аи
lstm_46/lstm_cell_46/add_4AddV2'lstm_46/lstm_cell_46/BiasAdd_2:output:0'lstm_46/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аt
lstm_46/lstm_cell_46/ReluRelulstm_46/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         Ае
lstm_46/lstm_cell_46/mul_3Mul&lstm_46/lstm_cell_46/clip_by_value:z:0'lstm_46/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АЦ
lstm_46/lstm_cell_46/add_5AddV2lstm_46/lstm_cell_46/mul_2:z:0lstm_46/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АФ
%lstm_46/lstm_cell_46/ReadVariableOp_3ReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_46/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_46/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_46/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_46/lstm_cell_46/strided_slice_3StridedSlice-lstm_46/lstm_cell_46/ReadVariableOp_3:value:03lstm_46/lstm_cell_46/strided_slice_3/stack:output:05lstm_46/lstm_cell_46/strided_slice_3/stack_1:output:05lstm_46/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_46/lstm_cell_46/MatMul_7MatMullstm_46/zeros:output:0-lstm_46/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Аи
lstm_46/lstm_cell_46/add_6AddV2'lstm_46/lstm_cell_46/BiasAdd_3:output:0'lstm_46/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         Аa
lstm_46/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_46/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_46/lstm_cell_46/Mul_4Mullstm_46/lstm_cell_46/add_6:z:0%lstm_46/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЭ
lstm_46/lstm_cell_46/Add_7AddV2lstm_46/lstm_cell_46/Mul_4:z:0%lstm_46/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аs
.lstm_46/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_46/lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_46/lstm_cell_46/Add_7:z:07lstm_46/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_46/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_46/lstm_cell_46/clip_by_value_2Maximum0lstm_46/lstm_cell_46/clip_by_value_2/Minimum:z:0/lstm_46/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аv
lstm_46/lstm_cell_46/Relu_1Relulstm_46/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Ай
lstm_46/lstm_cell_46/mul_5Mul(lstm_46/lstm_cell_46/clip_by_value_2:z:0)lstm_46/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         Аv
%lstm_46/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╨
lstm_46/TensorArrayV2_1TensorListReserve.lstm_46/TensorArrayV2_1/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_46/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_46/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_46/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ь
lstm_46/whileWhile#lstm_46/while/loop_counter:output:0)lstm_46/while/maximum_iterations:output:0lstm_46/time:output:0 lstm_46/TensorArrayV2_1:handle:0lstm_46/zeros:output:0lstm_46/zeros_1:output:0 lstm_46/strided_slice_1:output:0?lstm_46/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_46_lstm_cell_46_split_readvariableop_resource4lstm_46_lstm_cell_46_split_1_readvariableop_resource,lstm_46_lstm_cell_46_readvariableop_resource*
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
lstm_46_while_body_380195*%
condR
lstm_46_while_cond_380194*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Й
8lstm_46/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   █
*lstm_46/TensorArrayV2Stack/TensorListStackTensorListStacklstm_46/while:output:3Alstm_46/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
         А*
element_dtype0p
lstm_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_46/strided_slice_3StridedSlice3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_46/strided_slice_3/stack:output:0(lstm_46/strided_slice_3/stack_1:output:0(lstm_46/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskm
lstm_46/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
lstm_46/transpose_1	Transpose3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_46/transpose_1/perm:output:0*
T0*,
_output_shapes
:         
Аb
lstm_47/ShapeShapelstm_46/transpose_1:y:0*
T0*
_output_shapes
::э╧e
lstm_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_47/strided_sliceStridedSlicelstm_47/Shape:output:0$lstm_47/strided_slice/stack:output:0&lstm_47/strided_slice/stack_1:output:0&lstm_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_47/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Л
lstm_47/zeros/packedPacklstm_47/strided_slice:output:0lstm_47/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_47/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
lstm_47/zerosFilllstm_47/zeros/packed:output:0lstm_47/zeros/Const:output:0*
T0*'
_output_shapes
:         @Z
lstm_47/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@П
lstm_47/zeros_1/packedPacklstm_47/strided_slice:output:0!lstm_47/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_47/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
lstm_47/zeros_1Filllstm_47/zeros_1/packed:output:0lstm_47/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @k
lstm_47/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          П
lstm_47/transpose	Transposelstm_46/transpose_1:y:0lstm_47/transpose/perm:output:0*
T0*,
_output_shapes
:
         Аb
lstm_47/Shape_1Shapelstm_47/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_47/strided_slice_1StridedSlicelstm_47/Shape_1:output:0&lstm_47/strided_slice_1/stack:output:0(lstm_47/strided_slice_1/stack_1:output:0(lstm_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_47/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_47/TensorArrayV2TensorListReserve,lstm_47/TensorArrayV2/element_shape:output:0 lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   °
/lstm_47/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_47/transpose:y:0Flstm_47/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
lstm_47/strided_slice_2StridedSlicelstm_47/transpose:y:0&lstm_47/strided_slice_2/stack:output:0(lstm_47/strided_slice_2/stack_1:output:0(lstm_47/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskf
$lstm_47/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
)lstm_47/lstm_cell_47/split/ReadVariableOpReadVariableOp2lstm_47_lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0с
lstm_47/lstm_cell_47/splitSplit-lstm_47/lstm_cell_47/split/split_dim:output:01lstm_47/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЮ
lstm_47/lstm_cell_47/MatMulMatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @а
lstm_47/lstm_cell_47/MatMul_1MatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @а
lstm_47/lstm_cell_47/MatMul_2MatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @а
lstm_47/lstm_cell_47/MatMul_3MatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @h
&lstm_47/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_47/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4lstm_47_lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
lstm_47/lstm_cell_47/split_1Split/lstm_47/lstm_cell_47/split_1/split_dim:output:03lstm_47/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitз
lstm_47/lstm_cell_47/BiasAddBiasAdd%lstm_47/lstm_cell_47/MatMul:product:0%lstm_47/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @л
lstm_47/lstm_cell_47/BiasAdd_1BiasAdd'lstm_47/lstm_cell_47/MatMul_1:product:0%lstm_47/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @л
lstm_47/lstm_cell_47/BiasAdd_2BiasAdd'lstm_47/lstm_cell_47/MatMul_2:product:0%lstm_47/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @л
lstm_47/lstm_cell_47/BiasAdd_3BiasAdd'lstm_47/lstm_cell_47/MatMul_3:product:0%lstm_47/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @С
#lstm_47/lstm_cell_47/ReadVariableOpReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0y
(lstm_47/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_47/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm_47/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"lstm_47/lstm_cell_47/strided_sliceStridedSlice+lstm_47/lstm_cell_47/ReadVariableOp:value:01lstm_47/lstm_cell_47/strided_slice/stack:output:03lstm_47/lstm_cell_47/strided_slice/stack_1:output:03lstm_47/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЮ
lstm_47/lstm_cell_47/MatMul_4MatMullstm_47/zeros:output:0+lstm_47/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @г
lstm_47/lstm_cell_47/addAddV2%lstm_47/lstm_cell_47/BiasAdd:output:0'lstm_47/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @_
lstm_47/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_47/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
lstm_47/lstm_cell_47/MulMullstm_47/lstm_cell_47/add:z:0#lstm_47/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ъ
lstm_47/lstm_cell_47/Add_1AddV2lstm_47/lstm_cell_47/Mul:z:0%lstm_47/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @q
,lstm_47/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╛
*lstm_47/lstm_cell_47/clip_by_value/MinimumMinimumlstm_47/lstm_cell_47/Add_1:z:05lstm_47/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$lstm_47/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"lstm_47/lstm_cell_47/clip_by_valueMaximum.lstm_47/lstm_cell_47/clip_by_value/Minimum:z:0-lstm_47/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @У
%lstm_47/lstm_cell_47/ReadVariableOp_1ReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_47/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   }
,lstm_47/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_47/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_47/lstm_cell_47/strided_slice_1StridedSlice-lstm_47/lstm_cell_47/ReadVariableOp_1:value:03lstm_47/lstm_cell_47/strided_slice_1/stack:output:05lstm_47/lstm_cell_47/strided_slice_1/stack_1:output:05lstm_47/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_47/lstm_cell_47/MatMul_5MatMullstm_47/zeros:output:0-lstm_47/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @з
lstm_47/lstm_cell_47/add_2AddV2'lstm_47/lstm_cell_47/BiasAdd_1:output:0'lstm_47/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @a
lstm_47/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_47/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_47/lstm_cell_47/Mul_1Mullstm_47/lstm_cell_47/add_2:z:0%lstm_47/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ь
lstm_47/lstm_cell_47/Add_3AddV2lstm_47/lstm_cell_47/Mul_1:z:0%lstm_47/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @s
.lstm_47/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_47/lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_47/lstm_cell_47/Add_3:z:07lstm_47/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_47/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_47/lstm_cell_47/clip_by_value_1Maximum0lstm_47/lstm_cell_47/clip_by_value_1/Minimum:z:0/lstm_47/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @Ч
lstm_47/lstm_cell_47/mul_2Mul(lstm_47/lstm_cell_47/clip_by_value_1:z:0lstm_47/zeros_1:output:0*
T0*'
_output_shapes
:         @У
%lstm_47/lstm_cell_47/ReadVariableOp_2ReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_47/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_47/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_47/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_47/lstm_cell_47/strided_slice_2StridedSlice-lstm_47/lstm_cell_47/ReadVariableOp_2:value:03lstm_47/lstm_cell_47/strided_slice_2/stack:output:05lstm_47/lstm_cell_47/strided_slice_2/stack_1:output:05lstm_47/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_47/lstm_cell_47/MatMul_6MatMullstm_47/zeros:output:0-lstm_47/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @з
lstm_47/lstm_cell_47/add_4AddV2'lstm_47/lstm_cell_47/BiasAdd_2:output:0'lstm_47/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @s
lstm_47/lstm_cell_47/ReluRelulstm_47/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @д
lstm_47/lstm_cell_47/mul_3Mul&lstm_47/lstm_cell_47/clip_by_value:z:0'lstm_47/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @Х
lstm_47/lstm_cell_47/add_5AddV2lstm_47/lstm_cell_47/mul_2:z:0lstm_47/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @У
%lstm_47/lstm_cell_47/ReadVariableOp_3ReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_47/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_47/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_47/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_47/lstm_cell_47/strided_slice_3StridedSlice-lstm_47/lstm_cell_47/ReadVariableOp_3:value:03lstm_47/lstm_cell_47/strided_slice_3/stack:output:05lstm_47/lstm_cell_47/strided_slice_3/stack_1:output:05lstm_47/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_47/lstm_cell_47/MatMul_7MatMullstm_47/zeros:output:0-lstm_47/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @з
lstm_47/lstm_cell_47/add_6AddV2'lstm_47/lstm_cell_47/BiasAdd_3:output:0'lstm_47/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @a
lstm_47/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_47/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_47/lstm_cell_47/Mul_4Mullstm_47/lstm_cell_47/add_6:z:0%lstm_47/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ь
lstm_47/lstm_cell_47/Add_7AddV2lstm_47/lstm_cell_47/Mul_4:z:0%lstm_47/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @s
.lstm_47/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_47/lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_47/lstm_cell_47/Add_7:z:07lstm_47/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_47/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_47/lstm_cell_47/clip_by_value_2Maximum0lstm_47/lstm_cell_47/clip_by_value_2/Minimum:z:0/lstm_47/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @u
lstm_47/lstm_cell_47/Relu_1Relulstm_47/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @и
lstm_47/lstm_cell_47/mul_5Mul(lstm_47/lstm_cell_47/clip_by_value_2:z:0)lstm_47/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @v
%lstm_47/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╨
lstm_47/TensorArrayV2_1TensorListReserve.lstm_47/TensorArrayV2_1/element_shape:output:0 lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_47/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_47/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_47/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
lstm_47/whileWhile#lstm_47/while/loop_counter:output:0)lstm_47/while/maximum_iterations:output:0lstm_47/time:output:0 lstm_47/TensorArrayV2_1:handle:0lstm_47/zeros:output:0lstm_47/zeros_1:output:0 lstm_47/strided_slice_1:output:0?lstm_47/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_47_lstm_cell_47_split_readvariableop_resource4lstm_47_lstm_cell_47_split_1_readvariableop_resource,lstm_47_lstm_cell_47_readvariableop_resource*
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
lstm_47_while_body_380447*%
condR
lstm_47_while_cond_380446*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Й
8lstm_47/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┌
*lstm_47/TensorArrayV2Stack/TensorListStackTensorListStacklstm_47/while:output:3Alstm_47/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         @*
element_dtype0p
lstm_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
lstm_47/strided_slice_3StridedSlice3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_47/strided_slice_3/stack:output:0(lstm_47/strided_slice_3/stack_1:output:0(lstm_47/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskm
lstm_47/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
lstm_47/transpose_1	Transpose3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_47/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
@Ж
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Х
dense_23/MatMulMatMul lstm_47/strided_slice_3:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp$^lstm_46/lstm_cell_46/ReadVariableOp&^lstm_46/lstm_cell_46/ReadVariableOp_1&^lstm_46/lstm_cell_46/ReadVariableOp_2&^lstm_46/lstm_cell_46/ReadVariableOp_3*^lstm_46/lstm_cell_46/split/ReadVariableOp,^lstm_46/lstm_cell_46/split_1/ReadVariableOp^lstm_46/while$^lstm_47/lstm_cell_47/ReadVariableOp&^lstm_47/lstm_cell_47/ReadVariableOp_1&^lstm_47/lstm_cell_47/ReadVariableOp_2&^lstm_47/lstm_cell_47/ReadVariableOp_3*^lstm_47/lstm_cell_47/split/ReadVariableOp,^lstm_47/lstm_cell_47/split_1/ReadVariableOp^lstm_47/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2N
%lstm_46/lstm_cell_46/ReadVariableOp_1%lstm_46/lstm_cell_46/ReadVariableOp_12N
%lstm_46/lstm_cell_46/ReadVariableOp_2%lstm_46/lstm_cell_46/ReadVariableOp_22N
%lstm_46/lstm_cell_46/ReadVariableOp_3%lstm_46/lstm_cell_46/ReadVariableOp_32J
#lstm_46/lstm_cell_46/ReadVariableOp#lstm_46/lstm_cell_46/ReadVariableOp2V
)lstm_46/lstm_cell_46/split/ReadVariableOp)lstm_46/lstm_cell_46/split/ReadVariableOp2Z
+lstm_46/lstm_cell_46/split_1/ReadVariableOp+lstm_46/lstm_cell_46/split_1/ReadVariableOp2
lstm_46/whilelstm_46/while2N
%lstm_47/lstm_cell_47/ReadVariableOp_1%lstm_47/lstm_cell_47/ReadVariableOp_12N
%lstm_47/lstm_cell_47/ReadVariableOp_2%lstm_47/lstm_cell_47/ReadVariableOp_22N
%lstm_47/lstm_cell_47/ReadVariableOp_3%lstm_47/lstm_cell_47/ReadVariableOp_32J
#lstm_47/lstm_cell_47/ReadVariableOp#lstm_47/lstm_cell_47/ReadVariableOp2V
)lstm_47/lstm_cell_47/split/ReadVariableOp)lstm_47/lstm_cell_47/split/ReadVariableOp2Z
+lstm_47/lstm_cell_47/split_1/ReadVariableOp+lstm_47/lstm_cell_47/split_1/ReadVariableOp2
lstm_47/whilelstm_47/while:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
Ч	
├
while_cond_382588
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_382588___redundant_placeholder04
0while_while_cond_382588___redundant_placeholder14
0while_while_cond_382588___redundant_placeholder24
0while_while_cond_382588___redundant_placeholder3
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
Ч	
├
while_cond_381820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_381820___redundant_placeholder04
0while_while_cond_381820___redundant_placeholder14
0while_while_cond_381820___redundant_placeholder24
0while_while_cond_381820___redundant_placeholder3
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
╙#
х
while_body_377688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_46_377712_0:	А*
while_lstm_cell_46_377714_0:	А/
while_lstm_cell_46_377716_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_46_377712:	А(
while_lstm_cell_46_377714:	А-
while_lstm_cell_46_377716:
ААИв*while/lstm_cell_46/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╢
*while/lstm_cell_46/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_46_377712_0while_lstm_cell_46_377714_0while_lstm_cell_46_377716_0*
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377629▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_46/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_46/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         АС
while/Identity_5Identity3while/lstm_cell_46/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Аy

while/NoOpNoOp+^while/lstm_cell_46/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_46_377712while_lstm_cell_46_377712_0"8
while_lstm_cell_46_377714while_lstm_cell_46_377714_0"8
while_lstm_cell_46_377716while_lstm_cell_46_377716_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2X
*while/lstm_cell_46/StatefulPartitionedCall*while/lstm_cell_46/StatefulPartitionedCall:
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
б~
ж	
while_body_379213
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_46_split_readvariableop_resource_0:	АC
4while_lstm_cell_46_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_46_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_46_split_readvariableop_resource:	АA
2while_lstm_cell_46_split_1_readvariableop_resource:	А>
*while_lstm_cell_46_readvariableop_resource:
ААИв!while/lstm_cell_46/ReadVariableOpв#while/lstm_cell_46/ReadVariableOp_1в#while/lstm_cell_46/ReadVariableOp_2в#while/lstm_cell_46/ReadVariableOp_3в'while/lstm_cell_46/split/ReadVariableOpв)while/lstm_cell_46/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_46/split/ReadVariableOpReadVariableOp2while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_46/splitSplit+while/lstm_cell_46/split/split_dim:output:0/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_46/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_46/split_1Split-while/lstm_cell_46/split_1/split_dim:output:01while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_46/BiasAddBiasAdd#while/lstm_cell_46/MatMul:product:0#while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_1BiasAdd%while/lstm_cell_46/MatMul_1:product:0#while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_2BiasAdd%while/lstm_cell_46/MatMul_2:product:0#while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_3BiasAdd%while/lstm_cell_46/MatMul_3:product:0#while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_46/ReadVariableOpReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_46/strided_sliceStridedSlice)while/lstm_cell_46/ReadVariableOp:value:0/while/lstm_cell_46/strided_slice/stack:output:01while/lstm_cell_46/strided_slice/stack_1:output:01while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_46/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_46/addAddV2#while/lstm_cell_46/BiasAdd:output:0%while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_46/MulMulwhile/lstm_cell_46/add:z:0!while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_46/Add_1AddV2while/lstm_cell_46/Mul:z:0#while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_46/clip_by_value/MinimumMinimumwhile/lstm_cell_46/Add_1:z:03while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_46/clip_by_valueMaximum,while/lstm_cell_46/clip_by_value/Minimum:z:0+while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_1ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_1StridedSlice+while/lstm_cell_46/ReadVariableOp_1:value:01while/lstm_cell_46/strided_slice_1/stack:output:03while/lstm_cell_46/strided_slice_1/stack_1:output:03while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_2AddV2%while/lstm_cell_46/BiasAdd_1:output:0%while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_1Mulwhile/lstm_cell_46/add_2:z:0#while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_3AddV2while/lstm_cell_46/Mul_1:z:0#while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_1/MinimumMinimumwhile/lstm_cell_46/Add_3:z:05while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_1Maximum.while/lstm_cell_46/clip_by_value_1/Minimum:z:0-while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_46/mul_2Mul&while/lstm_cell_46/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_2ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_2StridedSlice+while/lstm_cell_46/ReadVariableOp_2:value:01while/lstm_cell_46/strided_slice_2/stack:output:03while/lstm_cell_46/strided_slice_2/stack_1:output:03while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_4AddV2%while/lstm_cell_46/BiasAdd_2:output:0%while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_46/ReluReluwhile/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_46/mul_3Mul$while/lstm_cell_46/clip_by_value:z:0%while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_46/add_5AddV2while/lstm_cell_46/mul_2:z:0while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_3ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_3StridedSlice+while/lstm_cell_46/ReadVariableOp_3:value:01while/lstm_cell_46/strided_slice_3/stack:output:03while/lstm_cell_46/strided_slice_3/stack_1:output:03while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_6AddV2%while/lstm_cell_46/BiasAdd_3:output:0%while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_4Mulwhile/lstm_cell_46/add_6:z:0#while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_7AddV2while/lstm_cell_46/Mul_4:z:0#while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_2/MinimumMinimumwhile/lstm_cell_46/Add_7:z:05while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_2Maximum.while/lstm_cell_46/clip_by_value_2/Minimum:z:0-while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_46/Relu_1Reluwhile/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_46/mul_5Mul&while/lstm_cell_46/clip_by_value_2:z:0'while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_46/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_46/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_46/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_46/ReadVariableOp$^while/lstm_cell_46/ReadVariableOp_1$^while/lstm_cell_46/ReadVariableOp_2$^while/lstm_cell_46/ReadVariableOp_3(^while/lstm_cell_46/split/ReadVariableOp*^while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_46_readvariableop_resource,while_lstm_cell_46_readvariableop_resource_0"j
2while_lstm_cell_46_split_1_readvariableop_resource4while_lstm_cell_46_split_1_readvariableop_resource_0"f
0while_lstm_cell_46_split_readvariableop_resource2while_lstm_cell_46_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_46/ReadVariableOp_1#while/lstm_cell_46/ReadVariableOp_12J
#while/lstm_cell_46/ReadVariableOp_2#while/lstm_cell_46/ReadVariableOp_22J
#while/lstm_cell_46/ReadVariableOp_3#while/lstm_cell_46/ReadVariableOp_32F
!while/lstm_cell_46/ReadVariableOp!while/lstm_cell_46/ReadVariableOp2R
'while/lstm_cell_46/split/ReadVariableOp'while/lstm_cell_46/split/ReadVariableOp2V
)while/lstm_cell_46/split_1/ReadVariableOp)while/lstm_cell_46/split_1/ReadVariableOp:
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
ы}
ж	
while_body_382077
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_47_split_readvariableop_resource_0:
ААC
4while_lstm_cell_47_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_47_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_47_split_readvariableop_resource:
ААA
2while_lstm_cell_47_split_1_readvariableop_resource:	А=
*while_lstm_cell_47_readvariableop_resource:	@АИв!while/lstm_cell_47/ReadVariableOpв#while/lstm_cell_47/ReadVariableOp_1в#while/lstm_cell_47/ReadVariableOp_2в#while/lstm_cell_47/ReadVariableOp_3в'while/lstm_cell_47/split/ReadVariableOpв)while/lstm_cell_47/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_47/split/ReadVariableOpReadVariableOp2while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_47/splitSplit+while/lstm_cell_47/split/split_dim:output:0/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_47/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_47/split_1Split-while/lstm_cell_47/split_1/split_dim:output:01while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_47/BiasAddBiasAdd#while/lstm_cell_47/MatMul:product:0#while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_1BiasAdd%while/lstm_cell_47/MatMul_1:product:0#while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_2BiasAdd%while/lstm_cell_47/MatMul_2:product:0#while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_3BiasAdd%while/lstm_cell_47/MatMul_3:product:0#while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_47/ReadVariableOpReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_47/strided_sliceStridedSlice)while/lstm_cell_47/ReadVariableOp:value:0/while/lstm_cell_47/strided_slice/stack:output:01while/lstm_cell_47/strided_slice/stack_1:output:01while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_47/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_47/addAddV2#while/lstm_cell_47/BiasAdd:output:0%while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_47/MulMulwhile/lstm_cell_47/add:z:0!while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_47/Add_1AddV2while/lstm_cell_47/Mul:z:0#while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_47/clip_by_value/MinimumMinimumwhile/lstm_cell_47/Add_1:z:03while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_47/clip_by_valueMaximum,while/lstm_cell_47/clip_by_value/Minimum:z:0+while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_1ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_1StridedSlice+while/lstm_cell_47/ReadVariableOp_1:value:01while/lstm_cell_47/strided_slice_1/stack:output:03while/lstm_cell_47/strided_slice_1/stack_1:output:03while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_2AddV2%while/lstm_cell_47/BiasAdd_1:output:0%while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_1Mulwhile/lstm_cell_47/add_2:z:0#while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_3AddV2while/lstm_cell_47/Mul_1:z:0#while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_1/MinimumMinimumwhile/lstm_cell_47/Add_3:z:05while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_1Maximum.while/lstm_cell_47/clip_by_value_1/Minimum:z:0-while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_47/mul_2Mul&while/lstm_cell_47/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_2ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_2StridedSlice+while/lstm_cell_47/ReadVariableOp_2:value:01while/lstm_cell_47/strided_slice_2/stack:output:03while/lstm_cell_47/strided_slice_2/stack_1:output:03while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_4AddV2%while/lstm_cell_47/BiasAdd_2:output:0%while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_47/ReluReluwhile/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_47/mul_3Mul$while/lstm_cell_47/clip_by_value:z:0%while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_47/add_5AddV2while/lstm_cell_47/mul_2:z:0while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_3ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_3StridedSlice+while/lstm_cell_47/ReadVariableOp_3:value:01while/lstm_cell_47/strided_slice_3/stack:output:03while/lstm_cell_47/strided_slice_3/stack_1:output:03while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_6AddV2%while/lstm_cell_47/BiasAdd_3:output:0%while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_4Mulwhile/lstm_cell_47/add_6:z:0#while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_7AddV2while/lstm_cell_47/Mul_4:z:0#while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_2/MinimumMinimumwhile/lstm_cell_47/Add_7:z:05while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_2Maximum.while/lstm_cell_47/clip_by_value_2/Minimum:z:0-while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_47/Relu_1Reluwhile/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_47/mul_5Mul&while/lstm_cell_47/clip_by_value_2:z:0'while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_47/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_47/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_47/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_47/ReadVariableOp$^while/lstm_cell_47/ReadVariableOp_1$^while/lstm_cell_47/ReadVariableOp_2$^while/lstm_cell_47/ReadVariableOp_3(^while/lstm_cell_47/split/ReadVariableOp*^while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_47_readvariableop_resource,while_lstm_cell_47_readvariableop_resource_0"j
2while_lstm_cell_47_split_1_readvariableop_resource4while_lstm_cell_47_split_1_readvariableop_resource_0"f
0while_lstm_cell_47_split_readvariableop_resource2while_lstm_cell_47_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_47/ReadVariableOp_1#while/lstm_cell_47/ReadVariableOp_12J
#while/lstm_cell_47/ReadVariableOp_2#while/lstm_cell_47/ReadVariableOp_22J
#while/lstm_cell_47/ReadVariableOp_3#while/lstm_cell_47/ReadVariableOp_32F
!while/lstm_cell_47/ReadVariableOp!while/lstm_cell_47/ReadVariableOp2R
'while/lstm_cell_47/split/ReadVariableOp'while/lstm_cell_47/split/ReadVariableOp2V
)while/lstm_cell_47/split_1/ReadVariableOp)while/lstm_cell_47/split_1/ReadVariableOp:
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
Ы	
├
while_cond_379212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_379212___redundant_placeholder04
0while_while_cond_379212___redundant_placeholder14
0while_while_cond_379212___redundant_placeholder24
0while_while_cond_379212___redundant_placeholder3
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
ТР
╛
lstm_46_while_body_379681,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3+
'lstm_46_while_lstm_46_strided_slice_1_0g
clstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_46_while_lstm_cell_46_split_readvariableop_resource_0:	АK
<lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0:	АH
4lstm_46_while_lstm_cell_46_readvariableop_resource_0:
АА
lstm_46_while_identity
lstm_46_while_identity_1
lstm_46_while_identity_2
lstm_46_while_identity_3
lstm_46_while_identity_4
lstm_46_while_identity_5)
%lstm_46_while_lstm_46_strided_slice_1e
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorK
8lstm_46_while_lstm_cell_46_split_readvariableop_resource:	АI
:lstm_46_while_lstm_cell_46_split_1_readvariableop_resource:	АF
2lstm_46_while_lstm_cell_46_readvariableop_resource:
ААИв)lstm_46/while/lstm_cell_46/ReadVariableOpв+lstm_46/while/lstm_cell_46/ReadVariableOp_1в+lstm_46/while/lstm_cell_46/ReadVariableOp_2в+lstm_46/while/lstm_cell_46/ReadVariableOp_3в/lstm_46/while/lstm_cell_46/split/ReadVariableOpв1lstm_46/while/lstm_cell_46/split_1/ReadVariableOpР
?lstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_46/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0lstm_46_while_placeholderHlstm_46/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0l
*lstm_46/while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :л
/lstm_46/while/lstm_cell_46/split/ReadVariableOpReadVariableOp:lstm_46_while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0є
 lstm_46/while/lstm_cell_46/splitSplit3lstm_46/while/lstm_cell_46/split/split_dim:output:07lstm_46/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split├
!lstm_46/while/lstm_cell_46/MatMulMatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         А┼
#lstm_46/while/lstm_cell_46/MatMul_1MatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         А┼
#lstm_46/while/lstm_cell_46/MatMul_2MatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         А┼
#lstm_46/while/lstm_cell_46/MatMul_3MatMul8lstm_46/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_46/while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аn
,lstm_46/while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_46/while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp<lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0щ
"lstm_46/while/lstm_cell_46/split_1Split5lstm_46/while/lstm_cell_46/split_1/split_dim:output:09lstm_46/while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_split║
"lstm_46/while/lstm_cell_46/BiasAddBiasAdd+lstm_46/while/lstm_cell_46/MatMul:product:0+lstm_46/while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         А╛
$lstm_46/while/lstm_cell_46/BiasAdd_1BiasAdd-lstm_46/while/lstm_cell_46/MatMul_1:product:0+lstm_46/while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         А╛
$lstm_46/while/lstm_cell_46/BiasAdd_2BiasAdd-lstm_46/while/lstm_cell_46/MatMul_2:product:0+lstm_46/while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         А╛
$lstm_46/while/lstm_cell_46/BiasAdd_3BiasAdd-lstm_46/while/lstm_cell_46/MatMul_3:product:0+lstm_46/while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         Аа
)lstm_46/while/lstm_cell_46/ReadVariableOpReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0
.lstm_46/while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_46/while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Б
0lstm_46/while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ї
(lstm_46/while/lstm_cell_46/strided_sliceStridedSlice1lstm_46/while/lstm_cell_46/ReadVariableOp:value:07lstm_46/while/lstm_cell_46/strided_slice/stack:output:09lstm_46/while/lstm_cell_46/strided_slice/stack_1:output:09lstm_46/while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask░
#lstm_46/while/lstm_cell_46/MatMul_4MatMullstm_46_while_placeholder_21lstm_46/while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         А╢
lstm_46/while/lstm_cell_46/addAddV2+lstm_46/while/lstm_cell_46/BiasAdd:output:0-lstm_46/while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         Аe
 lstm_46/while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_46/while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?з
lstm_46/while/lstm_cell_46/MulMul"lstm_46/while/lstm_cell_46/add:z:0)lstm_46/while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         Ан
 lstm_46/while/lstm_cell_46/Add_1AddV2"lstm_46/while/lstm_cell_46/Mul:z:0+lstm_46/while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аw
2lstm_46/while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
0lstm_46/while/lstm_cell_46/clip_by_value/MinimumMinimum$lstm_46/while/lstm_cell_46/Add_1:z:0;lstm_46/while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аo
*lstm_46/while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(lstm_46/while/lstm_cell_46/clip_by_valueMaximum4lstm_46/while/lstm_cell_46/clip_by_value/Minimum:z:03lstm_46/while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         Ав
+lstm_46/while/lstm_cell_46/ReadVariableOp_1ReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_46/while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_46/while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_46/while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_46/while/lstm_cell_46/strided_slice_1StridedSlice3lstm_46/while/lstm_cell_46/ReadVariableOp_1:value:09lstm_46/while/lstm_cell_46/strided_slice_1/stack:output:0;lstm_46/while/lstm_cell_46/strided_slice_1/stack_1:output:0;lstm_46/while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_46/while/lstm_cell_46/MatMul_5MatMullstm_46_while_placeholder_23lstm_46/while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         А║
 lstm_46/while/lstm_cell_46/add_2AddV2-lstm_46/while/lstm_cell_46/BiasAdd_1:output:0-lstm_46/while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         Аg
"lstm_46/while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_46/while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_46/while/lstm_cell_46/Mul_1Mul$lstm_46/while/lstm_cell_46/add_2:z:0+lstm_46/while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         Ап
 lstm_46/while/lstm_cell_46/Add_3AddV2$lstm_46/while/lstm_cell_46/Mul_1:z:0+lstm_46/while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аy
4lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_46/while/lstm_cell_46/clip_by_value_1/MinimumMinimum$lstm_46/while/lstm_cell_46/Add_3:z:0=lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_46/while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_46/while/lstm_cell_46/clip_by_value_1Maximum6lstm_46/while/lstm_cell_46/clip_by_value_1/Minimum:z:05lstm_46/while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         Аз
 lstm_46/while/lstm_cell_46/mul_2Mul.lstm_46/while/lstm_cell_46/clip_by_value_1:z:0lstm_46_while_placeholder_3*
T0*(
_output_shapes
:         Ав
+lstm_46/while/lstm_cell_46/ReadVariableOp_2ReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_46/while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Г
2lstm_46/while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_46/while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_46/while/lstm_cell_46/strided_slice_2StridedSlice3lstm_46/while/lstm_cell_46/ReadVariableOp_2:value:09lstm_46/while/lstm_cell_46/strided_slice_2/stack:output:0;lstm_46/while/lstm_cell_46/strided_slice_2/stack_1:output:0;lstm_46/while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_46/while/lstm_cell_46/MatMul_6MatMullstm_46_while_placeholder_23lstm_46/while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         А║
 lstm_46/while/lstm_cell_46/add_4AddV2-lstm_46/while/lstm_cell_46/BiasAdd_2:output:0-lstm_46/while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         АА
lstm_46/while/lstm_cell_46/ReluRelu$lstm_46/while/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         А╖
 lstm_46/while/lstm_cell_46/mul_3Mul,lstm_46/while/lstm_cell_46/clip_by_value:z:0-lstm_46/while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         Аи
 lstm_46/while/lstm_cell_46/add_5AddV2$lstm_46/while/lstm_cell_46/mul_2:z:0$lstm_46/while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         Ав
+lstm_46/while/lstm_cell_46/ReadVariableOp_3ReadVariableOp4lstm_46_while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Б
0lstm_46/while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  Г
2lstm_46/while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_46/while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ■
*lstm_46/while/lstm_cell_46/strided_slice_3StridedSlice3lstm_46/while/lstm_cell_46/ReadVariableOp_3:value:09lstm_46/while/lstm_cell_46/strided_slice_3/stack:output:0;lstm_46/while/lstm_cell_46/strided_slice_3/stack_1:output:0;lstm_46/while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask▓
#lstm_46/while/lstm_cell_46/MatMul_7MatMullstm_46_while_placeholder_23lstm_46/while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         А║
 lstm_46/while/lstm_cell_46/add_6AddV2-lstm_46/while/lstm_cell_46/BiasAdd_3:output:0-lstm_46/while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         Аg
"lstm_46/while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_46/while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?н
 lstm_46/while/lstm_cell_46/Mul_4Mul$lstm_46/while/lstm_cell_46/add_6:z:0+lstm_46/while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         Ап
 lstm_46/while/lstm_cell_46/Add_7AddV2$lstm_46/while/lstm_cell_46/Mul_4:z:0+lstm_46/while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аy
4lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╒
2lstm_46/while/lstm_cell_46/clip_by_value_2/MinimumMinimum$lstm_46/while/lstm_cell_46/Add_7:z:0=lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аq
,lstm_46/while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*lstm_46/while/lstm_cell_46/clip_by_value_2Maximum6lstm_46/while/lstm_cell_46/clip_by_value_2/Minimum:z:05lstm_46/while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АВ
!lstm_46/while/lstm_cell_46/Relu_1Relu$lstm_46/while/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         А╗
 lstm_46/while/lstm_cell_46/mul_5Mul.lstm_46/while/lstm_cell_46/clip_by_value_2:z:0/lstm_46/while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         Ах
2lstm_46/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_46_while_placeholder_1lstm_46_while_placeholder$lstm_46/while/lstm_cell_46/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_46/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_46/while/addAddV2lstm_46_while_placeholderlstm_46/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_46/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_46/while/add_1AddV2(lstm_46_while_lstm_46_while_loop_counterlstm_46/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_46/while/IdentityIdentitylstm_46/while/add_1:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: К
lstm_46/while/Identity_1Identity.lstm_46_while_lstm_46_while_maximum_iterations^lstm_46/while/NoOp*
T0*
_output_shapes
: q
lstm_46/while/Identity_2Identitylstm_46/while/add:z:0^lstm_46/while/NoOp*
T0*
_output_shapes
: Ю
lstm_46/while/Identity_3IdentityBlstm_46/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_46/while/NoOp*
T0*
_output_shapes
: Т
lstm_46/while/Identity_4Identity$lstm_46/while/lstm_cell_46/mul_5:z:0^lstm_46/while/NoOp*
T0*(
_output_shapes
:         АТ
lstm_46/while/Identity_5Identity$lstm_46/while/lstm_cell_46/add_5:z:0^lstm_46/while/NoOp*
T0*(
_output_shapes
:         АЁ
lstm_46/while/NoOpNoOp*^lstm_46/while/lstm_cell_46/ReadVariableOp,^lstm_46/while/lstm_cell_46/ReadVariableOp_1,^lstm_46/while/lstm_cell_46/ReadVariableOp_2,^lstm_46/while/lstm_cell_46/ReadVariableOp_30^lstm_46/while/lstm_cell_46/split/ReadVariableOp2^lstm_46/while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_46_while_identity_1!lstm_46/while/Identity_1:output:0"=
lstm_46_while_identity_2!lstm_46/while/Identity_2:output:0"=
lstm_46_while_identity_3!lstm_46/while/Identity_3:output:0"=
lstm_46_while_identity_4!lstm_46/while/Identity_4:output:0"=
lstm_46_while_identity_5!lstm_46/while/Identity_5:output:0"9
lstm_46_while_identitylstm_46/while/Identity:output:0"P
%lstm_46_while_lstm_46_strided_slice_1'lstm_46_while_lstm_46_strided_slice_1_0"j
2lstm_46_while_lstm_cell_46_readvariableop_resource4lstm_46_while_lstm_cell_46_readvariableop_resource_0"z
:lstm_46_while_lstm_cell_46_split_1_readvariableop_resource<lstm_46_while_lstm_cell_46_split_1_readvariableop_resource_0"v
8lstm_46_while_lstm_cell_46_split_readvariableop_resource:lstm_46_while_lstm_cell_46_split_readvariableop_resource_0"╚
alstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensorclstm_46_while_tensorarrayv2read_tensorlistgetitem_lstm_46_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2Z
+lstm_46/while/lstm_cell_46/ReadVariableOp_1+lstm_46/while/lstm_cell_46/ReadVariableOp_12Z
+lstm_46/while/lstm_cell_46/ReadVariableOp_2+lstm_46/while/lstm_cell_46/ReadVariableOp_22Z
+lstm_46/while/lstm_cell_46/ReadVariableOp_3+lstm_46/while/lstm_cell_46/ReadVariableOp_32V
)lstm_46/while/lstm_cell_46/ReadVariableOp)lstm_46/while/lstm_cell_46/ReadVariableOp2b
/lstm_46/while/lstm_cell_46/split/ReadVariableOp/lstm_46/while/lstm_cell_46/split/ReadVariableOp2f
1lstm_46/while/lstm_cell_46/split_1/ReadVariableOp1lstm_46/while/lstm_cell_46/split_1/ReadVariableOp:
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
_user_specified_name" lstm_46/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_46/while/loop_counter
·
╢
(__inference_lstm_47_layer_call_fn_381705

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
C__inference_lstm_47_layer_call_and_return_conditional_losses_379075o
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
:         
А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         
А
 
_user_specified_nameinputs
Ы	
├
while_cond_378348
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_378348___redundant_placeholder04
0while_while_cond_378348___redundant_placeholder14
0while_while_cond_378348___redundant_placeholder24
0while_while_cond_378348___redundant_placeholder3
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
╘J
к
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_377889

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
╙#
х
while_body_377441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_46_377465_0:	А*
while_lstm_cell_46_377467_0:	А/
while_lstm_cell_46_377469_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_46_377465:	А(
while_lstm_cell_46_377467:	А-
while_lstm_cell_46_377469:
ААИв*while/lstm_cell_46/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╢
*while/lstm_cell_46/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_46_377465_0while_lstm_cell_46_377467_0while_lstm_cell_46_377469_0*
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377427▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_46/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_46/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         АС
while/Identity_5Identity3while/lstm_cell_46/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Аy

while/NoOpNoOp+^while/lstm_cell_46/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_46_377465while_lstm_cell_46_377465_0"8
while_lstm_cell_46_377467while_lstm_cell_46_377467_0"8
while_lstm_cell_46_377469while_lstm_cell_46_377469_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2X
*while/lstm_cell_46/StatefulPartitionedCall*while/lstm_cell_46/StatefulPartitionedCall:
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
.__inference_sequential_23_layer_call_fn_379544

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
I__inference_sequential_23_layer_call_and_return_conditional_losses_378777o
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
':         
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ы}
ж	
while_body_378935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_47_split_readvariableop_resource_0:
ААC
4while_lstm_cell_47_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_47_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_47_split_readvariableop_resource:
ААA
2while_lstm_cell_47_split_1_readvariableop_resource:	А=
*while_lstm_cell_47_readvariableop_resource:	@АИв!while/lstm_cell_47/ReadVariableOpв#while/lstm_cell_47/ReadVariableOp_1в#while/lstm_cell_47/ReadVariableOp_2в#while/lstm_cell_47/ReadVariableOp_3в'while/lstm_cell_47/split/ReadVariableOpв)while/lstm_cell_47/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_47/split/ReadVariableOpReadVariableOp2while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_47/splitSplit+while/lstm_cell_47/split/split_dim:output:0/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_47/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_47/split_1Split-while/lstm_cell_47/split_1/split_dim:output:01while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_47/BiasAddBiasAdd#while/lstm_cell_47/MatMul:product:0#while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_1BiasAdd%while/lstm_cell_47/MatMul_1:product:0#while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_2BiasAdd%while/lstm_cell_47/MatMul_2:product:0#while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_3BiasAdd%while/lstm_cell_47/MatMul_3:product:0#while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_47/ReadVariableOpReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_47/strided_sliceStridedSlice)while/lstm_cell_47/ReadVariableOp:value:0/while/lstm_cell_47/strided_slice/stack:output:01while/lstm_cell_47/strided_slice/stack_1:output:01while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_47/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_47/addAddV2#while/lstm_cell_47/BiasAdd:output:0%while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_47/MulMulwhile/lstm_cell_47/add:z:0!while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_47/Add_1AddV2while/lstm_cell_47/Mul:z:0#while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_47/clip_by_value/MinimumMinimumwhile/lstm_cell_47/Add_1:z:03while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_47/clip_by_valueMaximum,while/lstm_cell_47/clip_by_value/Minimum:z:0+while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_1ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_1StridedSlice+while/lstm_cell_47/ReadVariableOp_1:value:01while/lstm_cell_47/strided_slice_1/stack:output:03while/lstm_cell_47/strided_slice_1/stack_1:output:03while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_2AddV2%while/lstm_cell_47/BiasAdd_1:output:0%while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_1Mulwhile/lstm_cell_47/add_2:z:0#while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_3AddV2while/lstm_cell_47/Mul_1:z:0#while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_1/MinimumMinimumwhile/lstm_cell_47/Add_3:z:05while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_1Maximum.while/lstm_cell_47/clip_by_value_1/Minimum:z:0-while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_47/mul_2Mul&while/lstm_cell_47/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_2ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_2StridedSlice+while/lstm_cell_47/ReadVariableOp_2:value:01while/lstm_cell_47/strided_slice_2/stack:output:03while/lstm_cell_47/strided_slice_2/stack_1:output:03while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_4AddV2%while/lstm_cell_47/BiasAdd_2:output:0%while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_47/ReluReluwhile/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_47/mul_3Mul$while/lstm_cell_47/clip_by_value:z:0%while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_47/add_5AddV2while/lstm_cell_47/mul_2:z:0while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_3ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_3StridedSlice+while/lstm_cell_47/ReadVariableOp_3:value:01while/lstm_cell_47/strided_slice_3/stack:output:03while/lstm_cell_47/strided_slice_3/stack_1:output:03while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_6AddV2%while/lstm_cell_47/BiasAdd_3:output:0%while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_4Mulwhile/lstm_cell_47/add_6:z:0#while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_7AddV2while/lstm_cell_47/Mul_4:z:0#while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_2/MinimumMinimumwhile/lstm_cell_47/Add_7:z:05while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_2Maximum.while/lstm_cell_47/clip_by_value_2/Minimum:z:0-while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_47/Relu_1Reluwhile/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_47/mul_5Mul&while/lstm_cell_47/clip_by_value_2:z:0'while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_47/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_47/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_47/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_47/ReadVariableOp$^while/lstm_cell_47/ReadVariableOp_1$^while/lstm_cell_47/ReadVariableOp_2$^while/lstm_cell_47/ReadVariableOp_3(^while/lstm_cell_47/split/ReadVariableOp*^while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_47_readvariableop_resource,while_lstm_cell_47_readvariableop_resource_0"j
2while_lstm_cell_47_split_1_readvariableop_resource4while_lstm_cell_47_split_1_readvariableop_resource_0"f
0while_lstm_cell_47_split_readvariableop_resource2while_lstm_cell_47_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_47/ReadVariableOp_1#while/lstm_cell_47/ReadVariableOp_12J
#while/lstm_cell_47/ReadVariableOp_2#while/lstm_cell_47/ReadVariableOp_22J
#while/lstm_cell_47/ReadVariableOp_3#while/lstm_cell_47/ReadVariableOp_32F
!while/lstm_cell_47/ReadVariableOp!while/lstm_cell_47/ReadVariableOp2R
'while/lstm_cell_47/split/ReadVariableOp'while/lstm_cell_47/split/ReadVariableOp2V
)while/lstm_cell_47/split_1/ReadVariableOp)while/lstm_cell_47/split_1/ReadVariableOp:
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
while_cond_382076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_382076___redundant_placeholder04
0while_while_cond_382076___redundant_placeholder14
0while_while_cond_382076___redundant_placeholder24
0while_while_cond_382076___redundant_placeholder3
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
Ч	
├
while_cond_382332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_382332___redundant_placeholder04
0while_while_cond_382332___redundant_placeholder14
0while_while_cond_382332___redundant_placeholder24
0while_while_cond_382332___redundant_placeholder3
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
Э·
∙
!__inference__wrapped_model_377303
lstm_46_inputS
@sequential_23_lstm_46_lstm_cell_46_split_readvariableop_resource:	АQ
Bsequential_23_lstm_46_lstm_cell_46_split_1_readvariableop_resource:	АN
:sequential_23_lstm_46_lstm_cell_46_readvariableop_resource:
ААT
@sequential_23_lstm_47_lstm_cell_47_split_readvariableop_resource:
ААQ
Bsequential_23_lstm_47_lstm_cell_47_split_1_readvariableop_resource:	АM
:sequential_23_lstm_47_lstm_cell_47_readvariableop_resource:	@АG
5sequential_23_dense_23_matmul_readvariableop_resource:@D
6sequential_23_dense_23_biasadd_readvariableop_resource:
identityИв-sequential_23/dense_23/BiasAdd/ReadVariableOpв,sequential_23/dense_23/MatMul/ReadVariableOpв1sequential_23/lstm_46/lstm_cell_46/ReadVariableOpв3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_1в3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_2в3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_3в7sequential_23/lstm_46/lstm_cell_46/split/ReadVariableOpв9sequential_23/lstm_46/lstm_cell_46/split_1/ReadVariableOpвsequential_23/lstm_46/whileв1sequential_23/lstm_47/lstm_cell_47/ReadVariableOpв3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_1в3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_2в3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_3в7sequential_23/lstm_47/lstm_cell_47/split/ReadVariableOpв9sequential_23/lstm_47/lstm_cell_47/split_1/ReadVariableOpвsequential_23/lstm_47/whilef
sequential_23/lstm_46/ShapeShapelstm_46_input*
T0*
_output_shapes
::э╧s
)sequential_23/lstm_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_23/lstm_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_23/lstm_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_23/lstm_46/strided_sliceStridedSlice$sequential_23/lstm_46/Shape:output:02sequential_23/lstm_46/strided_slice/stack:output:04sequential_23/lstm_46/strided_slice/stack_1:output:04sequential_23/lstm_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_23/lstm_46/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А╡
"sequential_23/lstm_46/zeros/packedPack,sequential_23/lstm_46/strided_slice:output:0-sequential_23/lstm_46/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_23/lstm_46/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    п
sequential_23/lstm_46/zerosFill+sequential_23/lstm_46/zeros/packed:output:0*sequential_23/lstm_46/zeros/Const:output:0*
T0*(
_output_shapes
:         Аi
&sequential_23/lstm_46/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :А╣
$sequential_23/lstm_46/zeros_1/packedPack,sequential_23/lstm_46/strided_slice:output:0/sequential_23/lstm_46/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_23/lstm_46/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╡
sequential_23/lstm_46/zeros_1Fill-sequential_23/lstm_46/zeros_1/packed:output:0,sequential_23/lstm_46/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аy
$sequential_23/lstm_46/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
sequential_23/lstm_46/transpose	Transposelstm_46_input-sequential_23/lstm_46/transpose/perm:output:0*
T0*+
_output_shapes
:
         ~
sequential_23/lstm_46/Shape_1Shape#sequential_23/lstm_46/transpose:y:0*
T0*
_output_shapes
::э╧u
+sequential_23/lstm_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_23/lstm_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_23/lstm_46/strided_slice_1StridedSlice&sequential_23/lstm_46/Shape_1:output:04sequential_23/lstm_46/strided_slice_1/stack:output:06sequential_23/lstm_46/strided_slice_1/stack_1:output:06sequential_23/lstm_46/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_23/lstm_46/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
#sequential_23/lstm_46/TensorArrayV2TensorListReserve:sequential_23/lstm_46/TensorArrayV2/element_shape:output:0.sequential_23/lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ь
Ksequential_23/lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       в
=sequential_23/lstm_46/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_23/lstm_46/transpose:y:0Tsequential_23/lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥u
+sequential_23/lstm_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_23/lstm_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╫
%sequential_23/lstm_46/strided_slice_2StridedSlice#sequential_23/lstm_46/transpose:y:04sequential_23/lstm_46/strided_slice_2/stack:output:06sequential_23/lstm_46/strided_slice_2/stack_1:output:06sequential_23/lstm_46/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskt
2sequential_23/lstm_46/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╣
7sequential_23/lstm_46/lstm_cell_46/split/ReadVariableOpReadVariableOp@sequential_23_lstm_46_lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
(sequential_23/lstm_46/lstm_cell_46/splitSplit;sequential_23/lstm_46/lstm_cell_46/split/split_dim:output:0?sequential_23/lstm_46/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_split╔
)sequential_23/lstm_46/lstm_cell_46/MatMulMatMul.sequential_23/lstm_46/strided_slice_2:output:01sequential_23/lstm_46/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         А╦
+sequential_23/lstm_46/lstm_cell_46/MatMul_1MatMul.sequential_23/lstm_46/strided_slice_2:output:01sequential_23/lstm_46/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         А╦
+sequential_23/lstm_46/lstm_cell_46/MatMul_2MatMul.sequential_23/lstm_46/strided_slice_2:output:01sequential_23/lstm_46/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         А╦
+sequential_23/lstm_46/lstm_cell_46/MatMul_3MatMul.sequential_23/lstm_46/strided_slice_2:output:01sequential_23/lstm_46/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аv
4sequential_23/lstm_46/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╣
9sequential_23/lstm_46/lstm_cell_46/split_1/ReadVariableOpReadVariableOpBsequential_23_lstm_46_lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Б
*sequential_23/lstm_46/lstm_cell_46/split_1Split=sequential_23/lstm_46/lstm_cell_46/split_1/split_dim:output:0Asequential_23/lstm_46/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_split╥
*sequential_23/lstm_46/lstm_cell_46/BiasAddBiasAdd3sequential_23/lstm_46/lstm_cell_46/MatMul:product:03sequential_23/lstm_46/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         А╓
,sequential_23/lstm_46/lstm_cell_46/BiasAdd_1BiasAdd5sequential_23/lstm_46/lstm_cell_46/MatMul_1:product:03sequential_23/lstm_46/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         А╓
,sequential_23/lstm_46/lstm_cell_46/BiasAdd_2BiasAdd5sequential_23/lstm_46/lstm_cell_46/MatMul_2:product:03sequential_23/lstm_46/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         А╓
,sequential_23/lstm_46/lstm_cell_46/BiasAdd_3BiasAdd5sequential_23/lstm_46/lstm_cell_46/MatMul_3:product:03sequential_23/lstm_46/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         Ао
1sequential_23/lstm_46/lstm_cell_46/ReadVariableOpReadVariableOp:sequential_23_lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0З
6sequential_23/lstm_46/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Й
8sequential_23/lstm_46/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Й
8sequential_23/lstm_46/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ь
0sequential_23/lstm_46/lstm_cell_46/strided_sliceStridedSlice9sequential_23/lstm_46/lstm_cell_46/ReadVariableOp:value:0?sequential_23/lstm_46/lstm_cell_46/strided_slice/stack:output:0Asequential_23/lstm_46/lstm_cell_46/strided_slice/stack_1:output:0Asequential_23/lstm_46/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╔
+sequential_23/lstm_46/lstm_cell_46/MatMul_4MatMul$sequential_23/lstm_46/zeros:output:09sequential_23/lstm_46/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         А╬
&sequential_23/lstm_46/lstm_cell_46/addAddV23sequential_23/lstm_46/lstm_cell_46/BiasAdd:output:05sequential_23/lstm_46/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         Аm
(sequential_23/lstm_46/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_23/lstm_46/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?┐
&sequential_23/lstm_46/lstm_cell_46/MulMul*sequential_23/lstm_46/lstm_cell_46/add:z:01sequential_23/lstm_46/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         А┼
(sequential_23/lstm_46/lstm_cell_46/Add_1AddV2*sequential_23/lstm_46/lstm_cell_46/Mul:z:03sequential_23/lstm_46/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         А
:sequential_23/lstm_46/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?щ
8sequential_23/lstm_46/lstm_cell_46/clip_by_value/MinimumMinimum,sequential_23/lstm_46/lstm_cell_46/Add_1:z:0Csequential_23/lstm_46/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аw
2sequential_23/lstm_46/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    щ
0sequential_23/lstm_46/lstm_cell_46/clip_by_valueMaximum<sequential_23/lstm_46/lstm_cell_46/clip_by_value/Minimum:z:0;sequential_23/lstm_46/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         А░
3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_1ReadVariableOp:sequential_23_lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
8sequential_23/lstm_46/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Л
:sequential_23/lstm_46/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Л
:sequential_23/lstm_46/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
2sequential_23/lstm_46/lstm_cell_46/strided_slice_1StridedSlice;sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_1:value:0Asequential_23/lstm_46/lstm_cell_46/strided_slice_1/stack:output:0Csequential_23/lstm_46/lstm_cell_46/strided_slice_1/stack_1:output:0Csequential_23/lstm_46/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╦
+sequential_23/lstm_46/lstm_cell_46/MatMul_5MatMul$sequential_23/lstm_46/zeros:output:0;sequential_23/lstm_46/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         А╥
(sequential_23/lstm_46/lstm_cell_46/add_2AddV25sequential_23/lstm_46/lstm_cell_46/BiasAdd_1:output:05sequential_23/lstm_46/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         Аo
*sequential_23/lstm_46/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_23/lstm_46/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
(sequential_23/lstm_46/lstm_cell_46/Mul_1Mul,sequential_23/lstm_46/lstm_cell_46/add_2:z:03sequential_23/lstm_46/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         А╟
(sequential_23/lstm_46/lstm_cell_46/Add_3AddV2,sequential_23/lstm_46/lstm_cell_46/Mul_1:z:03sequential_23/lstm_46/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         АБ
<sequential_23/lstm_46/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?э
:sequential_23/lstm_46/lstm_cell_46/clip_by_value_1/MinimumMinimum,sequential_23/lstm_46/lstm_cell_46/Add_3:z:0Esequential_23/lstm_46/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аy
4sequential_23/lstm_46/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    я
2sequential_23/lstm_46/lstm_cell_46/clip_by_value_1Maximum>sequential_23/lstm_46/lstm_cell_46/clip_by_value_1/Minimum:z:0=sequential_23/lstm_46/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         А┬
(sequential_23/lstm_46/lstm_cell_46/mul_2Mul6sequential_23/lstm_46/lstm_cell_46/clip_by_value_1:z:0&sequential_23/lstm_46/zeros_1:output:0*
T0*(
_output_shapes
:         А░
3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_2ReadVariableOp:sequential_23_lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
8sequential_23/lstm_46/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Л
:sequential_23/lstm_46/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  Л
:sequential_23/lstm_46/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
2sequential_23/lstm_46/lstm_cell_46/strided_slice_2StridedSlice;sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_2:value:0Asequential_23/lstm_46/lstm_cell_46/strided_slice_2/stack:output:0Csequential_23/lstm_46/lstm_cell_46/strided_slice_2/stack_1:output:0Csequential_23/lstm_46/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╦
+sequential_23/lstm_46/lstm_cell_46/MatMul_6MatMul$sequential_23/lstm_46/zeros:output:0;sequential_23/lstm_46/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         А╥
(sequential_23/lstm_46/lstm_cell_46/add_4AddV25sequential_23/lstm_46/lstm_cell_46/BiasAdd_2:output:05sequential_23/lstm_46/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         АР
'sequential_23/lstm_46/lstm_cell_46/ReluRelu,sequential_23/lstm_46/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         А╧
(sequential_23/lstm_46/lstm_cell_46/mul_3Mul4sequential_23/lstm_46/lstm_cell_46/clip_by_value:z:05sequential_23/lstm_46/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А└
(sequential_23/lstm_46/lstm_cell_46/add_5AddV2,sequential_23/lstm_46/lstm_cell_46/mul_2:z:0,sequential_23/lstm_46/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         А░
3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_3ReadVariableOp:sequential_23_lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
8sequential_23/lstm_46/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  Л
:sequential_23/lstm_46/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Л
:sequential_23/lstm_46/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
2sequential_23/lstm_46/lstm_cell_46/strided_slice_3StridedSlice;sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_3:value:0Asequential_23/lstm_46/lstm_cell_46/strided_slice_3/stack:output:0Csequential_23/lstm_46/lstm_cell_46/strided_slice_3/stack_1:output:0Csequential_23/lstm_46/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_mask╦
+sequential_23/lstm_46/lstm_cell_46/MatMul_7MatMul$sequential_23/lstm_46/zeros:output:0;sequential_23/lstm_46/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         А╥
(sequential_23/lstm_46/lstm_cell_46/add_6AddV25sequential_23/lstm_46/lstm_cell_46/BiasAdd_3:output:05sequential_23/lstm_46/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         Аo
*sequential_23/lstm_46/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_23/lstm_46/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
(sequential_23/lstm_46/lstm_cell_46/Mul_4Mul,sequential_23/lstm_46/lstm_cell_46/add_6:z:03sequential_23/lstm_46/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         А╟
(sequential_23/lstm_46/lstm_cell_46/Add_7AddV2,sequential_23/lstm_46/lstm_cell_46/Mul_4:z:03sequential_23/lstm_46/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         АБ
<sequential_23/lstm_46/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?э
:sequential_23/lstm_46/lstm_cell_46/clip_by_value_2/MinimumMinimum,sequential_23/lstm_46/lstm_cell_46/Add_7:z:0Esequential_23/lstm_46/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аy
4sequential_23/lstm_46/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    я
2sequential_23/lstm_46/lstm_cell_46/clip_by_value_2Maximum>sequential_23/lstm_46/lstm_cell_46/clip_by_value_2/Minimum:z:0=sequential_23/lstm_46/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         АТ
)sequential_23/lstm_46/lstm_cell_46/Relu_1Relu,sequential_23/lstm_46/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         А╙
(sequential_23/lstm_46/lstm_cell_46/mul_5Mul6sequential_23/lstm_46/lstm_cell_46/clip_by_value_2:z:07sequential_23/lstm_46/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         АД
3sequential_23/lstm_46/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ·
%sequential_23/lstm_46/TensorArrayV2_1TensorListReserve<sequential_23/lstm_46/TensorArrayV2_1/element_shape:output:0.sequential_23/lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥\
sequential_23/lstm_46/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_23/lstm_46/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_23/lstm_46/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ░
sequential_23/lstm_46/whileWhile1sequential_23/lstm_46/while/loop_counter:output:07sequential_23/lstm_46/while/maximum_iterations:output:0#sequential_23/lstm_46/time:output:0.sequential_23/lstm_46/TensorArrayV2_1:handle:0$sequential_23/lstm_46/zeros:output:0&sequential_23/lstm_46/zeros_1:output:0.sequential_23/lstm_46/strided_slice_1:output:0Msequential_23/lstm_46/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_23_lstm_46_lstm_cell_46_split_readvariableop_resourceBsequential_23_lstm_46_lstm_cell_46_split_1_readvariableop_resource:sequential_23_lstm_46_lstm_cell_46_readvariableop_resource*
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
'sequential_23_lstm_46_while_body_376905*3
cond+R)
'sequential_23_lstm_46_while_cond_376904*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Ч
Fsequential_23/lstm_46/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   Е
8sequential_23/lstm_46/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_23/lstm_46/while:output:3Osequential_23/lstm_46/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
         А*
element_dtype0~
+sequential_23/lstm_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_23/lstm_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
%sequential_23/lstm_46/strided_slice_3StridedSliceAsequential_23/lstm_46/TensorArrayV2Stack/TensorListStack:tensor:04sequential_23/lstm_46/strided_slice_3/stack:output:06sequential_23/lstm_46/strided_slice_3/stack_1:output:06sequential_23/lstm_46/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_mask{
&sequential_23/lstm_46/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┘
!sequential_23/lstm_46/transpose_1	TransposeAsequential_23/lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_23/lstm_46/transpose_1/perm:output:0*
T0*,
_output_shapes
:         
А~
sequential_23/lstm_47/ShapeShape%sequential_23/lstm_46/transpose_1:y:0*
T0*
_output_shapes
::э╧s
)sequential_23/lstm_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_23/lstm_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_23/lstm_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_23/lstm_47/strided_sliceStridedSlice$sequential_23/lstm_47/Shape:output:02sequential_23/lstm_47/strided_slice/stack:output:04sequential_23/lstm_47/strided_slice/stack_1:output:04sequential_23/lstm_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_23/lstm_47/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@╡
"sequential_23/lstm_47/zeros/packedPack,sequential_23/lstm_47/strided_slice:output:0-sequential_23/lstm_47/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_23/lstm_47/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    о
sequential_23/lstm_47/zerosFill+sequential_23/lstm_47/zeros/packed:output:0*sequential_23/lstm_47/zeros/Const:output:0*
T0*'
_output_shapes
:         @h
&sequential_23/lstm_47/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@╣
$sequential_23/lstm_47/zeros_1/packedPack,sequential_23/lstm_47/strided_slice:output:0/sequential_23/lstm_47/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_23/lstm_47/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_23/lstm_47/zeros_1Fill-sequential_23/lstm_47/zeros_1/packed:output:0,sequential_23/lstm_47/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @y
$sequential_23/lstm_47/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╣
sequential_23/lstm_47/transpose	Transpose%sequential_23/lstm_46/transpose_1:y:0-sequential_23/lstm_47/transpose/perm:output:0*
T0*,
_output_shapes
:
         А~
sequential_23/lstm_47/Shape_1Shape#sequential_23/lstm_47/transpose:y:0*
T0*
_output_shapes
::э╧u
+sequential_23/lstm_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_23/lstm_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_23/lstm_47/strided_slice_1StridedSlice&sequential_23/lstm_47/Shape_1:output:04sequential_23/lstm_47/strided_slice_1/stack:output:06sequential_23/lstm_47/strided_slice_1/stack_1:output:06sequential_23/lstm_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_23/lstm_47/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
#sequential_23/lstm_47/TensorArrayV2TensorListReserve:sequential_23/lstm_47/TensorArrayV2/element_shape:output:0.sequential_23/lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ь
Ksequential_23/lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   в
=sequential_23/lstm_47/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_23/lstm_47/transpose:y:0Tsequential_23/lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥u
+sequential_23/lstm_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_23/lstm_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
%sequential_23/lstm_47/strided_slice_2StridedSlice#sequential_23/lstm_47/transpose:y:04sequential_23/lstm_47/strided_slice_2/stack:output:06sequential_23/lstm_47/strided_slice_2/stack_1:output:06sequential_23/lstm_47/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskt
2sequential_23/lstm_47/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
7sequential_23/lstm_47/lstm_cell_47/split/ReadVariableOpReadVariableOp@sequential_23_lstm_47_lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Л
(sequential_23/lstm_47/lstm_cell_47/splitSplit;sequential_23/lstm_47/lstm_cell_47/split/split_dim:output:0?sequential_23/lstm_47/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split╚
)sequential_23/lstm_47/lstm_cell_47/MatMulMatMul.sequential_23/lstm_47/strided_slice_2:output:01sequential_23/lstm_47/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @╩
+sequential_23/lstm_47/lstm_cell_47/MatMul_1MatMul.sequential_23/lstm_47/strided_slice_2:output:01sequential_23/lstm_47/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @╩
+sequential_23/lstm_47/lstm_cell_47/MatMul_2MatMul.sequential_23/lstm_47/strided_slice_2:output:01sequential_23/lstm_47/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @╩
+sequential_23/lstm_47/lstm_cell_47/MatMul_3MatMul.sequential_23/lstm_47/strided_slice_2:output:01sequential_23/lstm_47/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @v
4sequential_23/lstm_47/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╣
9sequential_23/lstm_47/lstm_cell_47/split_1/ReadVariableOpReadVariableOpBsequential_23_lstm_47_lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0¤
*sequential_23/lstm_47/lstm_cell_47/split_1Split=sequential_23/lstm_47/lstm_cell_47/split_1/split_dim:output:0Asequential_23/lstm_47/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split╤
*sequential_23/lstm_47/lstm_cell_47/BiasAddBiasAdd3sequential_23/lstm_47/lstm_cell_47/MatMul:product:03sequential_23/lstm_47/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @╒
,sequential_23/lstm_47/lstm_cell_47/BiasAdd_1BiasAdd5sequential_23/lstm_47/lstm_cell_47/MatMul_1:product:03sequential_23/lstm_47/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @╒
,sequential_23/lstm_47/lstm_cell_47/BiasAdd_2BiasAdd5sequential_23/lstm_47/lstm_cell_47/MatMul_2:product:03sequential_23/lstm_47/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @╒
,sequential_23/lstm_47/lstm_cell_47/BiasAdd_3BiasAdd5sequential_23/lstm_47/lstm_cell_47/MatMul_3:product:03sequential_23/lstm_47/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @н
1sequential_23/lstm_47/lstm_cell_47/ReadVariableOpReadVariableOp:sequential_23_lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0З
6sequential_23/lstm_47/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Й
8sequential_23/lstm_47/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Й
8sequential_23/lstm_47/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ъ
0sequential_23/lstm_47/lstm_cell_47/strided_sliceStridedSlice9sequential_23/lstm_47/lstm_cell_47/ReadVariableOp:value:0?sequential_23/lstm_47/lstm_cell_47/strided_slice/stack:output:0Asequential_23/lstm_47/lstm_cell_47/strided_slice/stack_1:output:0Asequential_23/lstm_47/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╚
+sequential_23/lstm_47/lstm_cell_47/MatMul_4MatMul$sequential_23/lstm_47/zeros:output:09sequential_23/lstm_47/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @═
&sequential_23/lstm_47/lstm_cell_47/addAddV23sequential_23/lstm_47/lstm_cell_47/BiasAdd:output:05sequential_23/lstm_47/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @m
(sequential_23/lstm_47/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_23/lstm_47/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╛
&sequential_23/lstm_47/lstm_cell_47/MulMul*sequential_23/lstm_47/lstm_cell_47/add:z:01sequential_23/lstm_47/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @─
(sequential_23/lstm_47/lstm_cell_47/Add_1AddV2*sequential_23/lstm_47/lstm_cell_47/Mul:z:03sequential_23/lstm_47/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @
:sequential_23/lstm_47/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ш
8sequential_23/lstm_47/lstm_cell_47/clip_by_value/MinimumMinimum,sequential_23/lstm_47/lstm_cell_47/Add_1:z:0Csequential_23/lstm_47/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @w
2sequential_23/lstm_47/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ш
0sequential_23/lstm_47/lstm_cell_47/clip_by_valueMaximum<sequential_23/lstm_47/lstm_cell_47/clip_by_value/Minimum:z:0;sequential_23/lstm_47/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @п
3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_1ReadVariableOp:sequential_23_lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0Й
8sequential_23/lstm_47/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   Л
:sequential_23/lstm_47/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Л
:sequential_23/lstm_47/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_23/lstm_47/lstm_cell_47/strided_slice_1StridedSlice;sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_1:value:0Asequential_23/lstm_47/lstm_cell_47/strided_slice_1/stack:output:0Csequential_23/lstm_47/lstm_cell_47/strided_slice_1/stack_1:output:0Csequential_23/lstm_47/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╩
+sequential_23/lstm_47/lstm_cell_47/MatMul_5MatMul$sequential_23/lstm_47/zeros:output:0;sequential_23/lstm_47/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @╤
(sequential_23/lstm_47/lstm_cell_47/add_2AddV25sequential_23/lstm_47/lstm_cell_47/BiasAdd_1:output:05sequential_23/lstm_47/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @o
*sequential_23/lstm_47/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_23/lstm_47/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?─
(sequential_23/lstm_47/lstm_cell_47/Mul_1Mul,sequential_23/lstm_47/lstm_cell_47/add_2:z:03sequential_23/lstm_47/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @╞
(sequential_23/lstm_47/lstm_cell_47/Add_3AddV2,sequential_23/lstm_47/lstm_cell_47/Mul_1:z:03sequential_23/lstm_47/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @Б
<sequential_23/lstm_47/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ь
:sequential_23/lstm_47/lstm_cell_47/clip_by_value_1/MinimumMinimum,sequential_23/lstm_47/lstm_cell_47/Add_3:z:0Esequential_23/lstm_47/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @y
4sequential_23/lstm_47/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
2sequential_23/lstm_47/lstm_cell_47/clip_by_value_1Maximum>sequential_23/lstm_47/lstm_cell_47/clip_by_value_1/Minimum:z:0=sequential_23/lstm_47/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @┴
(sequential_23/lstm_47/lstm_cell_47/mul_2Mul6sequential_23/lstm_47/lstm_cell_47/clip_by_value_1:z:0&sequential_23/lstm_47/zeros_1:output:0*
T0*'
_output_shapes
:         @п
3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_2ReadVariableOp:sequential_23_lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0Й
8sequential_23/lstm_47/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Л
:sequential_23/lstm_47/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   Л
:sequential_23/lstm_47/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_23/lstm_47/lstm_cell_47/strided_slice_2StridedSlice;sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_2:value:0Asequential_23/lstm_47/lstm_cell_47/strided_slice_2/stack:output:0Csequential_23/lstm_47/lstm_cell_47/strided_slice_2/stack_1:output:0Csequential_23/lstm_47/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╩
+sequential_23/lstm_47/lstm_cell_47/MatMul_6MatMul$sequential_23/lstm_47/zeros:output:0;sequential_23/lstm_47/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @╤
(sequential_23/lstm_47/lstm_cell_47/add_4AddV25sequential_23/lstm_47/lstm_cell_47/BiasAdd_2:output:05sequential_23/lstm_47/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @П
'sequential_23/lstm_47/lstm_cell_47/ReluRelu,sequential_23/lstm_47/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @╬
(sequential_23/lstm_47/lstm_cell_47/mul_3Mul4sequential_23/lstm_47/lstm_cell_47/clip_by_value:z:05sequential_23/lstm_47/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @┐
(sequential_23/lstm_47/lstm_cell_47/add_5AddV2,sequential_23/lstm_47/lstm_cell_47/mul_2:z:0,sequential_23/lstm_47/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @п
3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_3ReadVariableOp:sequential_23_lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0Й
8sequential_23/lstm_47/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   Л
:sequential_23/lstm_47/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Л
:sequential_23/lstm_47/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_23/lstm_47/lstm_cell_47/strided_slice_3StridedSlice;sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_3:value:0Asequential_23/lstm_47/lstm_cell_47/strided_slice_3/stack:output:0Csequential_23/lstm_47/lstm_cell_47/strided_slice_3/stack_1:output:0Csequential_23/lstm_47/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask╩
+sequential_23/lstm_47/lstm_cell_47/MatMul_7MatMul$sequential_23/lstm_47/zeros:output:0;sequential_23/lstm_47/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @╤
(sequential_23/lstm_47/lstm_cell_47/add_6AddV25sequential_23/lstm_47/lstm_cell_47/BiasAdd_3:output:05sequential_23/lstm_47/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @o
*sequential_23/lstm_47/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>o
*sequential_23/lstm_47/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?─
(sequential_23/lstm_47/lstm_cell_47/Mul_4Mul,sequential_23/lstm_47/lstm_cell_47/add_6:z:03sequential_23/lstm_47/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @╞
(sequential_23/lstm_47/lstm_cell_47/Add_7AddV2,sequential_23/lstm_47/lstm_cell_47/Mul_4:z:03sequential_23/lstm_47/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @Б
<sequential_23/lstm_47/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ь
:sequential_23/lstm_47/lstm_cell_47/clip_by_value_2/MinimumMinimum,sequential_23/lstm_47/lstm_cell_47/Add_7:z:0Esequential_23/lstm_47/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @y
4sequential_23/lstm_47/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
2sequential_23/lstm_47/lstm_cell_47/clip_by_value_2Maximum>sequential_23/lstm_47/lstm_cell_47/clip_by_value_2/Minimum:z:0=sequential_23/lstm_47/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @С
)sequential_23/lstm_47/lstm_cell_47/Relu_1Relu,sequential_23/lstm_47/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @╥
(sequential_23/lstm_47/lstm_cell_47/mul_5Mul6sequential_23/lstm_47/lstm_cell_47/clip_by_value_2:z:07sequential_23/lstm_47/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @Д
3sequential_23/lstm_47/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ·
%sequential_23/lstm_47/TensorArrayV2_1TensorListReserve<sequential_23/lstm_47/TensorArrayV2_1/element_shape:output:0.sequential_23/lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥\
sequential_23/lstm_47/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_23/lstm_47/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_23/lstm_47/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : м
sequential_23/lstm_47/whileWhile1sequential_23/lstm_47/while/loop_counter:output:07sequential_23/lstm_47/while/maximum_iterations:output:0#sequential_23/lstm_47/time:output:0.sequential_23/lstm_47/TensorArrayV2_1:handle:0$sequential_23/lstm_47/zeros:output:0&sequential_23/lstm_47/zeros_1:output:0.sequential_23/lstm_47/strided_slice_1:output:0Msequential_23/lstm_47/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_23_lstm_47_lstm_cell_47_split_readvariableop_resourceBsequential_23_lstm_47_lstm_cell_47_split_1_readvariableop_resource:sequential_23_lstm_47_lstm_cell_47_readvariableop_resource*
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
'sequential_23_lstm_47_while_body_377157*3
cond+R)
'sequential_23_lstm_47_while_cond_377156*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Ч
Fsequential_23/lstm_47/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Д
8sequential_23/lstm_47/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_23/lstm_47/while:output:3Osequential_23/lstm_47/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         @*
element_dtype0~
+sequential_23/lstm_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_23/lstm_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
%sequential_23/lstm_47/strided_slice_3StridedSliceAsequential_23/lstm_47/TensorArrayV2Stack/TensorListStack:tensor:04sequential_23/lstm_47/strided_slice_3/stack:output:06sequential_23/lstm_47/strided_slice_3/stack_1:output:06sequential_23/lstm_47/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask{
&sequential_23/lstm_47/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╪
!sequential_23/lstm_47/transpose_1	TransposeAsequential_23/lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_23/lstm_47/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
@в
,sequential_23/dense_23/MatMul/ReadVariableOpReadVariableOp5sequential_23_dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0┐
sequential_23/dense_23/MatMulMatMul.sequential_23/lstm_47/strided_slice_3:output:04sequential_23/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_23/dense_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_23/dense_23/BiasAddBiasAdd'sequential_23/dense_23/MatMul:product:05sequential_23/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
IdentityIdentity'sequential_23/dense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ∙
NoOpNoOp.^sequential_23/dense_23/BiasAdd/ReadVariableOp-^sequential_23/dense_23/MatMul/ReadVariableOp2^sequential_23/lstm_46/lstm_cell_46/ReadVariableOp4^sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_14^sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_24^sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_38^sequential_23/lstm_46/lstm_cell_46/split/ReadVariableOp:^sequential_23/lstm_46/lstm_cell_46/split_1/ReadVariableOp^sequential_23/lstm_46/while2^sequential_23/lstm_47/lstm_cell_47/ReadVariableOp4^sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_14^sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_24^sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_38^sequential_23/lstm_47/lstm_cell_47/split/ReadVariableOp:^sequential_23/lstm_47/lstm_cell_47/split_1/ReadVariableOp^sequential_23/lstm_47/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2^
-sequential_23/dense_23/BiasAdd/ReadVariableOp-sequential_23/dense_23/BiasAdd/ReadVariableOp2\
,sequential_23/dense_23/MatMul/ReadVariableOp,sequential_23/dense_23/MatMul/ReadVariableOp2j
3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_13sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_12j
3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_23sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_22j
3sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_33sequential_23/lstm_46/lstm_cell_46/ReadVariableOp_32f
1sequential_23/lstm_46/lstm_cell_46/ReadVariableOp1sequential_23/lstm_46/lstm_cell_46/ReadVariableOp2r
7sequential_23/lstm_46/lstm_cell_46/split/ReadVariableOp7sequential_23/lstm_46/lstm_cell_46/split/ReadVariableOp2v
9sequential_23/lstm_46/lstm_cell_46/split_1/ReadVariableOp9sequential_23/lstm_46/lstm_cell_46/split_1/ReadVariableOp2:
sequential_23/lstm_46/whilesequential_23/lstm_46/while2j
3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_13sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_12j
3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_23sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_22j
3sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_33sequential_23/lstm_47/lstm_cell_47/ReadVariableOp_32f
1sequential_23/lstm_47/lstm_cell_47/ReadVariableOp1sequential_23/lstm_47/lstm_cell_47/ReadVariableOp2r
7sequential_23/lstm_47/lstm_cell_47/split/ReadVariableOp7sequential_23/lstm_47/lstm_cell_47/split/ReadVariableOp2v
9sequential_23/lstm_47/lstm_cell_47/split_1/ReadVariableOp9sequential_23/lstm_47/lstm_cell_47/split_1/ReadVariableOp2:
sequential_23/lstm_47/whilesequential_23/lstm_47/while:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_46_input
Ч	
├
while_cond_378149
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_378149___redundant_placeholder04
0while_while_cond_378149___redundant_placeholder14
0while_while_cond_378149___redundant_placeholder24
0while_while_cond_378149___redundant_placeholder3
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
ъ
C__inference_lstm_47_layer_call_and_return_conditional_losses_382217
inputs_0>
*lstm_cell_47_split_readvariableop_resource:
АА;
,lstm_cell_47_split_1_readvariableop_resource:	А7
$lstm_cell_47_readvariableop_resource:	@А
identityИвlstm_cell_47/ReadVariableOpвlstm_cell_47/ReadVariableOp_1вlstm_cell_47/ReadVariableOp_2вlstm_cell_47/ReadVariableOp_3в!lstm_cell_47/split/ReadVariableOpв#lstm_cell_47/split_1/ReadVariableOpвwhileK
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
lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_47/split/ReadVariableOpReadVariableOp*lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_47/splitSplit%lstm_cell_47/split/split_dim:output:0)lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_47/MatMulMatMulstrided_slice_2:output:0lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_47/split_1/ReadVariableOpReadVariableOp,lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_47/split_1Split'lstm_cell_47/split_1/split_dim:output:0+lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_47/BiasAddBiasAddlstm_cell_47/MatMul:product:0lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_1BiasAddlstm_cell_47/MatMul_1:product:0lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_2BiasAddlstm_cell_47/MatMul_2:product:0lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_3BiasAddlstm_cell_47/MatMul_3:product:0lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_47/ReadVariableOpReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_47/strided_sliceStridedSlice#lstm_cell_47/ReadVariableOp:value:0)lstm_cell_47/strided_slice/stack:output:0+lstm_cell_47/strided_slice/stack_1:output:0+lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_47/MatMul_4MatMulzeros:output:0#lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_47/addAddV2lstm_cell_47/BiasAdd:output:0lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_47/MulMullstm_cell_47/add:z:0lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_47/Add_1AddV2lstm_cell_47/Mul:z:0lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_47/clip_by_value/MinimumMinimumlstm_cell_47/Add_1:z:0-lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_47/clip_by_valueMaximum&lstm_cell_47/clip_by_value/Minimum:z:0%lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_1ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_1StridedSlice%lstm_cell_47/ReadVariableOp_1:value:0+lstm_cell_47/strided_slice_1/stack:output:0-lstm_cell_47/strided_slice_1/stack_1:output:0-lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_5MatMulzeros:output:0%lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_2AddV2lstm_cell_47/BiasAdd_1:output:0lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_1Mullstm_cell_47/add_2:z:0lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_3AddV2lstm_cell_47/Mul_1:z:0lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_cell_47/Add_3:z:0/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_1Maximum(lstm_cell_47/clip_by_value_1/Minimum:z:0'lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_47/mul_2Mul lstm_cell_47/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_2ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_2StridedSlice%lstm_cell_47/ReadVariableOp_2:value:0+lstm_cell_47/strided_slice_2/stack:output:0-lstm_cell_47/strided_slice_2/stack_1:output:0-lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_6MatMulzeros:output:0%lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_4AddV2lstm_cell_47/BiasAdd_2:output:0lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/ReluRelulstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_47/mul_3Mullstm_cell_47/clip_by_value:z:0lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_47/add_5AddV2lstm_cell_47/mul_2:z:0lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_3ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_3StridedSlice%lstm_cell_47/ReadVariableOp_3:value:0+lstm_cell_47/strided_slice_3/stack:output:0-lstm_cell_47/strided_slice_3/stack_1:output:0-lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_7MatMulzeros:output:0%lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_6AddV2lstm_cell_47/BiasAdd_3:output:0lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_4Mullstm_cell_47/add_6:z:0lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_7AddV2lstm_cell_47/Mul_4:z:0lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_cell_47/Add_7:z:0/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_2Maximum(lstm_cell_47/clip_by_value_2/Minimum:z:0'lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_47/Relu_1Relulstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_47/mul_5Mul lstm_cell_47/clip_by_value_2:z:0!lstm_cell_47/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_47_split_readvariableop_resource,lstm_cell_47_split_1_readvariableop_resource$lstm_cell_47_readvariableop_resource*
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
while_body_382077*
condR
while_cond_382076*K
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
NoOpNoOp^lstm_cell_47/ReadVariableOp^lstm_cell_47/ReadVariableOp_1^lstm_cell_47/ReadVariableOp_2^lstm_cell_47/ReadVariableOp_3"^lstm_cell_47/split/ReadVariableOp$^lstm_cell_47/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2>
lstm_cell_47/ReadVariableOp_1lstm_cell_47/ReadVariableOp_12>
lstm_cell_47/ReadVariableOp_2lstm_cell_47/ReadVariableOp_22>
lstm_cell_47/ReadVariableOp_3lstm_cell_47/ReadVariableOp_32:
lstm_cell_47/ReadVariableOplstm_cell_47/ReadVariableOp2F
!lstm_cell_47/split/ReadVariableOp!lstm_cell_47/split/ReadVariableOp2J
#lstm_cell_47/split_1/ReadVariableOp#lstm_cell_47/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  А
"
_user_specified_name
inputs_0
╦#
х
while_body_377903
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_47_377927_0:
АА*
while_lstm_cell_47_377929_0:	А.
while_lstm_cell_47_377931_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_47_377927:
АА(
while_lstm_cell_47_377929:	А,
while_lstm_cell_47_377931:	@АИв*while/lstm_cell_47/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0│
*while/lstm_cell_47/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_47_377927_0while_lstm_cell_47_377929_0while_lstm_cell_47_377931_0*
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_377889▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_47/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_47/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @Р
while/Identity_5Identity3while/lstm_cell_47/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @y

while/NoOpNoOp+^while/lstm_cell_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"8
while_lstm_cell_47_377927while_lstm_cell_47_377927_0"8
while_lstm_cell_47_377929while_lstm_cell_47_377929_0"8
while_lstm_cell_47_377931while_lstm_cell_47_377931_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2X
*while/lstm_cell_47/StatefulPartitionedCall*while/lstm_cell_47/StatefulPartitionedCall:
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
Е
ф
I__inference_sequential_23_layer_call_and_return_conditional_losses_379494
lstm_46_input!
lstm_46_379474:	А
lstm_46_379476:	А"
lstm_46_379478:
АА"
lstm_47_379481:
АА
lstm_47_379483:	А!
lstm_47_379485:	@А!
dense_23_379488:@
dense_23_379490:
identityИв dense_23/StatefulPartitionedCallвlstm_46/StatefulPartitionedCallвlstm_47/StatefulPartitionedCallК
lstm_46/StatefulPartitionedCallStatefulPartitionedCalllstm_46_inputlstm_46_379474lstm_46_379476lstm_46_379478*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         
А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_lstm_46_layer_call_and_return_conditional_losses_379353а
lstm_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_46/StatefulPartitionedCall:output:0lstm_47_379481lstm_47_379483lstm_47_379485*
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_379075Т
 dense_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_47/StatefulPartitionedCall:output:0dense_23_379488dense_23_379490*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_378770x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp!^dense_23/StatefulPartitionedCall ^lstm_46/StatefulPartitionedCall ^lstm_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2B
lstm_46/StatefulPartitionedCalllstm_46/StatefulPartitionedCall2B
lstm_47/StatefulPartitionedCalllstm_47/StatefulPartitionedCall:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_46_input
ЪK
м
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382960

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
Ч	
├
while_cond_378611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_378611___redundant_placeholder04
0while_while_cond_378611___redundant_placeholder14
0while_while_cond_378611___redundant_placeholder24
0while_while_cond_378611___redundant_placeholder3
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
ё	
╨
.__inference_sequential_23_layer_call_fn_378796
lstm_46_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_46_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_23_layer_call_and_return_conditional_losses_378777o
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
':         
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_46_input
с7
Ж
C__inference_lstm_46_layer_call_and_return_conditional_losses_377756

inputs&
lstm_cell_46_377675:	А"
lstm_cell_46_377677:	А'
lstm_cell_46_377679:
АА
identityИв$lstm_cell_46/StatefulPartitionedCallвwhileI
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
$lstm_cell_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_46_377675lstm_cell_46_377677lstm_cell_46_377679*
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377629n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_46_377675lstm_cell_46_377677lstm_cell_46_377679*
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
while_body_377688*
condR
while_cond_377687*M
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
NoOpNoOp%^lstm_cell_46/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_46/StatefulPartitionedCall$lstm_cell_46/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
░
√
'sequential_23_lstm_46_while_cond_376904H
Dsequential_23_lstm_46_while_sequential_23_lstm_46_while_loop_counterN
Jsequential_23_lstm_46_while_sequential_23_lstm_46_while_maximum_iterations+
'sequential_23_lstm_46_while_placeholder-
)sequential_23_lstm_46_while_placeholder_1-
)sequential_23_lstm_46_while_placeholder_2-
)sequential_23_lstm_46_while_placeholder_3J
Fsequential_23_lstm_46_while_less_sequential_23_lstm_46_strided_slice_1`
\sequential_23_lstm_46_while_sequential_23_lstm_46_while_cond_376904___redundant_placeholder0`
\sequential_23_lstm_46_while_sequential_23_lstm_46_while_cond_376904___redundant_placeholder1`
\sequential_23_lstm_46_while_sequential_23_lstm_46_while_cond_376904___redundant_placeholder2`
\sequential_23_lstm_46_while_sequential_23_lstm_46_while_cond_376904___redundant_placeholder3(
$sequential_23_lstm_46_while_identity
║
 sequential_23/lstm_46/while/LessLess'sequential_23_lstm_46_while_placeholderFsequential_23_lstm_46_while_less_sequential_23_lstm_46_strided_slice_1*
T0*
_output_shapes
: w
$sequential_23/lstm_46/while/IdentityIdentity$sequential_23/lstm_46/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_23_lstm_46_while_identity-sequential_23/lstm_46/while/Identity:output:0*(
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
_user_specified_name0.sequential_23/lstm_46/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_23/lstm_46/while/loop_counter
Єо
к
'sequential_23_lstm_47_while_body_377157H
Dsequential_23_lstm_47_while_sequential_23_lstm_47_while_loop_counterN
Jsequential_23_lstm_47_while_sequential_23_lstm_47_while_maximum_iterations+
'sequential_23_lstm_47_while_placeholder-
)sequential_23_lstm_47_while_placeholder_1-
)sequential_23_lstm_47_while_placeholder_2-
)sequential_23_lstm_47_while_placeholder_3G
Csequential_23_lstm_47_while_sequential_23_lstm_47_strided_slice_1_0Г
sequential_23_lstm_47_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_47_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_23_lstm_47_while_lstm_cell_47_split_readvariableop_resource_0:
ААY
Jsequential_23_lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0:	АU
Bsequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource_0:	@А(
$sequential_23_lstm_47_while_identity*
&sequential_23_lstm_47_while_identity_1*
&sequential_23_lstm_47_while_identity_2*
&sequential_23_lstm_47_while_identity_3*
&sequential_23_lstm_47_while_identity_4*
&sequential_23_lstm_47_while_identity_5E
Asequential_23_lstm_47_while_sequential_23_lstm_47_strided_slice_1Б
}sequential_23_lstm_47_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_47_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_23_lstm_47_while_lstm_cell_47_split_readvariableop_resource:
ААW
Hsequential_23_lstm_47_while_lstm_cell_47_split_1_readvariableop_resource:	АS
@sequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource:	@АИв7sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOpв9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_1в9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_2в9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_3в=sequential_23/lstm_47/while/lstm_cell_47/split/ReadVariableOpв?sequential_23/lstm_47/while/lstm_cell_47/split_1/ReadVariableOpЮ
Msequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   Х
?sequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_23_lstm_47_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_47_tensorarrayunstack_tensorlistfromtensor_0'sequential_23_lstm_47_while_placeholderVsequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0z
8sequential_23/lstm_47/while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╚
=sequential_23/lstm_47/while/lstm_cell_47/split/ReadVariableOpReadVariableOpHsequential_23_lstm_47_while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0Э
.sequential_23/lstm_47/while/lstm_cell_47/splitSplitAsequential_23/lstm_47/while/lstm_cell_47/split/split_dim:output:0Esequential_23/lstm_47/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitь
/sequential_23/lstm_47/while/lstm_cell_47/MatMulMatMulFsequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_47/while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @ю
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_1MatMulFsequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_47/while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @ю
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_2MatMulFsequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_47/while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @ю
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_3MatMulFsequential_23/lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_23/lstm_47/while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @|
:sequential_23/lstm_47/while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ╟
?sequential_23/lstm_47/while/lstm_cell_47/split_1/ReadVariableOpReadVariableOpJsequential_23_lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0П
0sequential_23/lstm_47/while/lstm_cell_47/split_1SplitCsequential_23/lstm_47/while/lstm_cell_47/split_1/split_dim:output:0Gsequential_23/lstm_47/while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitу
0sequential_23/lstm_47/while/lstm_cell_47/BiasAddBiasAdd9sequential_23/lstm_47/while/lstm_cell_47/MatMul:product:09sequential_23/lstm_47/while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @ч
2sequential_23/lstm_47/while/lstm_cell_47/BiasAdd_1BiasAdd;sequential_23/lstm_47/while/lstm_cell_47/MatMul_1:product:09sequential_23/lstm_47/while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @ч
2sequential_23/lstm_47/while/lstm_cell_47/BiasAdd_2BiasAdd;sequential_23/lstm_47/while/lstm_cell_47/MatMul_2:product:09sequential_23/lstm_47/while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @ч
2sequential_23/lstm_47/while/lstm_cell_47/BiasAdd_3BiasAdd;sequential_23/lstm_47/while/lstm_cell_47/MatMul_3:product:09sequential_23/lstm_47/while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @╗
7sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOpReadVariableOpBsequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Н
<sequential_23/lstm_47/while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        П
>sequential_23/lstm_47/while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   П
>sequential_23/lstm_47/while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
6sequential_23/lstm_47/while/lstm_cell_47/strided_sliceStridedSlice?sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp:value:0Esequential_23/lstm_47/while/lstm_cell_47/strided_slice/stack:output:0Gsequential_23/lstm_47/while/lstm_cell_47/strided_slice/stack_1:output:0Gsequential_23/lstm_47/while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask┘
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_4MatMul)sequential_23_lstm_47_while_placeholder_2?sequential_23/lstm_47/while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @▀
,sequential_23/lstm_47/while/lstm_cell_47/addAddV29sequential_23/lstm_47/while/lstm_cell_47/BiasAdd:output:0;sequential_23/lstm_47/while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @s
.sequential_23/lstm_47/while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_23/lstm_47/while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╨
,sequential_23/lstm_47/while/lstm_cell_47/MulMul0sequential_23/lstm_47/while/lstm_cell_47/add:z:07sequential_23/lstm_47/while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @╓
.sequential_23/lstm_47/while/lstm_cell_47/Add_1AddV20sequential_23/lstm_47/while/lstm_cell_47/Mul:z:09sequential_23/lstm_47/while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @Е
@sequential_23/lstm_47/while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?·
>sequential_23/lstm_47/while/lstm_cell_47/clip_by_value/MinimumMinimum2sequential_23/lstm_47/while/lstm_cell_47/Add_1:z:0Isequential_23/lstm_47/while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @}
8sequential_23/lstm_47/while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ·
6sequential_23/lstm_47/while/lstm_cell_47/clip_by_valueMaximumBsequential_23/lstm_47/while/lstm_cell_47/clip_by_value/Minimum:z:0Asequential_23/lstm_47/while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @╜
9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_1ReadVariableOpBsequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0П
>sequential_23/lstm_47/while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   С
@sequential_23/lstm_47/while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   С
@sequential_23/lstm_47/while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_23/lstm_47/while/lstm_cell_47/strided_slice_1StridedSliceAsequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_1:value:0Gsequential_23/lstm_47/while/lstm_cell_47/strided_slice_1/stack:output:0Isequential_23/lstm_47/while/lstm_cell_47/strided_slice_1/stack_1:output:0Isequential_23/lstm_47/while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask█
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_5MatMul)sequential_23_lstm_47_while_placeholder_2Asequential_23/lstm_47/while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @у
.sequential_23/lstm_47/while/lstm_cell_47/add_2AddV2;sequential_23/lstm_47/while/lstm_cell_47/BiasAdd_1:output:0;sequential_23/lstm_47/while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @u
0sequential_23/lstm_47/while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_23/lstm_47/while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╓
.sequential_23/lstm_47/while/lstm_cell_47/Mul_1Mul2sequential_23/lstm_47/while/lstm_cell_47/add_2:z:09sequential_23/lstm_47/while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @╪
.sequential_23/lstm_47/while/lstm_cell_47/Add_3AddV22sequential_23/lstm_47/while/lstm_cell_47/Mul_1:z:09sequential_23/lstm_47/while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @З
Bsequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?■
@sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1/MinimumMinimum2sequential_23/lstm_47/while/lstm_cell_47/Add_3:z:0Ksequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @
:sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
8sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1MaximumDsequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum:z:0Csequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @╨
.sequential_23/lstm_47/while/lstm_cell_47/mul_2Mul<sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_1:z:0)sequential_23_lstm_47_while_placeholder_3*
T0*'
_output_shapes
:         @╜
9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_2ReadVariableOpBsequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0П
>sequential_23/lstm_47/while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   С
@sequential_23/lstm_47/while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   С
@sequential_23/lstm_47/while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_23/lstm_47/while/lstm_cell_47/strided_slice_2StridedSliceAsequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_2:value:0Gsequential_23/lstm_47/while/lstm_cell_47/strided_slice_2/stack:output:0Isequential_23/lstm_47/while/lstm_cell_47/strided_slice_2/stack_1:output:0Isequential_23/lstm_47/while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask█
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_6MatMul)sequential_23_lstm_47_while_placeholder_2Asequential_23/lstm_47/while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @у
.sequential_23/lstm_47/while/lstm_cell_47/add_4AddV2;sequential_23/lstm_47/while/lstm_cell_47/BiasAdd_2:output:0;sequential_23/lstm_47/while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @Ы
-sequential_23/lstm_47/while/lstm_cell_47/ReluRelu2sequential_23/lstm_47/while/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @р
.sequential_23/lstm_47/while/lstm_cell_47/mul_3Mul:sequential_23/lstm_47/while/lstm_cell_47/clip_by_value:z:0;sequential_23/lstm_47/while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @╤
.sequential_23/lstm_47/while/lstm_cell_47/add_5AddV22sequential_23/lstm_47/while/lstm_cell_47/mul_2:z:02sequential_23/lstm_47/while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @╜
9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_3ReadVariableOpBsequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0П
>sequential_23/lstm_47/while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   С
@sequential_23/lstm_47/while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        С
@sequential_23/lstm_47/while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_23/lstm_47/while/lstm_cell_47/strided_slice_3StridedSliceAsequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_3:value:0Gsequential_23/lstm_47/while/lstm_cell_47/strided_slice_3/stack:output:0Isequential_23/lstm_47/while/lstm_cell_47/strided_slice_3/stack_1:output:0Isequential_23/lstm_47/while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask█
1sequential_23/lstm_47/while/lstm_cell_47/MatMul_7MatMul)sequential_23_lstm_47_while_placeholder_2Asequential_23/lstm_47/while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @у
.sequential_23/lstm_47/while/lstm_cell_47/add_6AddV2;sequential_23/lstm_47/while/lstm_cell_47/BiasAdd_3:output:0;sequential_23/lstm_47/while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @u
0sequential_23/lstm_47/while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>u
0sequential_23/lstm_47/while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?╓
.sequential_23/lstm_47/while/lstm_cell_47/Mul_4Mul2sequential_23/lstm_47/while/lstm_cell_47/add_6:z:09sequential_23/lstm_47/while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @╪
.sequential_23/lstm_47/while/lstm_cell_47/Add_7AddV22sequential_23/lstm_47/while/lstm_cell_47/Mul_4:z:09sequential_23/lstm_47/while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @З
Bsequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?■
@sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2/MinimumMinimum2sequential_23/lstm_47/while/lstm_cell_47/Add_7:z:0Ksequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @
:sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    А
8sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2MaximumDsequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum:z:0Csequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @Э
/sequential_23/lstm_47/while/lstm_cell_47/Relu_1Relu2sequential_23/lstm_47/while/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @ф
.sequential_23/lstm_47/while/lstm_cell_47/mul_5Mul<sequential_23/lstm_47/while/lstm_cell_47/clip_by_value_2:z:0=sequential_23/lstm_47/while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @Э
@sequential_23/lstm_47/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_23_lstm_47_while_placeholder_1'sequential_23_lstm_47_while_placeholder2sequential_23/lstm_47/while/lstm_cell_47/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥c
!sequential_23/lstm_47/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ю
sequential_23/lstm_47/while/addAddV2'sequential_23_lstm_47_while_placeholder*sequential_23/lstm_47/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_23/lstm_47/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_23/lstm_47/while/add_1AddV2Dsequential_23_lstm_47_while_sequential_23_lstm_47_while_loop_counter,sequential_23/lstm_47/while/add_1/y:output:0*
T0*
_output_shapes
: Ы
$sequential_23/lstm_47/while/IdentityIdentity%sequential_23/lstm_47/while/add_1:z:0!^sequential_23/lstm_47/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_23/lstm_47/while/Identity_1IdentityJsequential_23_lstm_47_while_sequential_23_lstm_47_while_maximum_iterations!^sequential_23/lstm_47/while/NoOp*
T0*
_output_shapes
: Ы
&sequential_23/lstm_47/while/Identity_2Identity#sequential_23/lstm_47/while/add:z:0!^sequential_23/lstm_47/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_23/lstm_47/while/Identity_3IdentityPsequential_23/lstm_47/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_23/lstm_47/while/NoOp*
T0*
_output_shapes
: ╗
&sequential_23/lstm_47/while/Identity_4Identity2sequential_23/lstm_47/while/lstm_cell_47/mul_5:z:0!^sequential_23/lstm_47/while/NoOp*
T0*'
_output_shapes
:         @╗
&sequential_23/lstm_47/while/Identity_5Identity2sequential_23/lstm_47/while/lstm_cell_47/add_5:z:0!^sequential_23/lstm_47/while/NoOp*
T0*'
_output_shapes
:         @╥
 sequential_23/lstm_47/while/NoOpNoOp8^sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp:^sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_1:^sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_2:^sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_3>^sequential_23/lstm_47/while/lstm_cell_47/split/ReadVariableOp@^sequential_23/lstm_47/while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Y
&sequential_23_lstm_47_while_identity_1/sequential_23/lstm_47/while/Identity_1:output:0"Y
&sequential_23_lstm_47_while_identity_2/sequential_23/lstm_47/while/Identity_2:output:0"Y
&sequential_23_lstm_47_while_identity_3/sequential_23/lstm_47/while/Identity_3:output:0"Y
&sequential_23_lstm_47_while_identity_4/sequential_23/lstm_47/while/Identity_4:output:0"Y
&sequential_23_lstm_47_while_identity_5/sequential_23/lstm_47/while/Identity_5:output:0"U
$sequential_23_lstm_47_while_identity-sequential_23/lstm_47/while/Identity:output:0"Ж
@sequential_23_lstm_47_while_lstm_cell_47_readvariableop_resourceBsequential_23_lstm_47_while_lstm_cell_47_readvariableop_resource_0"Ц
Hsequential_23_lstm_47_while_lstm_cell_47_split_1_readvariableop_resourceJsequential_23_lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0"Т
Fsequential_23_lstm_47_while_lstm_cell_47_split_readvariableop_resourceHsequential_23_lstm_47_while_lstm_cell_47_split_readvariableop_resource_0"И
Asequential_23_lstm_47_while_sequential_23_lstm_47_strided_slice_1Csequential_23_lstm_47_while_sequential_23_lstm_47_strided_slice_1_0"А
}sequential_23_lstm_47_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_47_tensorarrayunstack_tensorlistfromtensorsequential_23_lstm_47_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_47_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2v
9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_19sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_12v
9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_29sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_22v
9sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_39sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp_32r
7sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp7sequential_23/lstm_47/while/lstm_cell_47/ReadVariableOp2~
=sequential_23/lstm_47/while/lstm_cell_47/split/ReadVariableOp=sequential_23/lstm_47/while/lstm_cell_47/split/ReadVariableOp2В
?sequential_23/lstm_47/while/lstm_cell_47/split_1/ReadVariableOp?sequential_23/lstm_47/while/lstm_cell_47/split_1/ReadVariableOp:
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
_user_specified_name0.sequential_23/lstm_47/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_23/lstm_47/while/loop_counter
╟	
ї
D__inference_dense_23_layer_call_and_return_conditional_losses_378770

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
тJ
м
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383083

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
иИ
ш
C__inference_lstm_47_layer_call_and_return_conditional_losses_382473

inputs>
*lstm_cell_47_split_readvariableop_resource:
АА;
,lstm_cell_47_split_1_readvariableop_resource:	А7
$lstm_cell_47_readvariableop_resource:	@А
identityИвlstm_cell_47/ReadVariableOpвlstm_cell_47/ReadVariableOp_1вlstm_cell_47/ReadVariableOp_2вlstm_cell_47/ReadVariableOp_3в!lstm_cell_47/split/ReadVariableOpв#lstm_cell_47/split_1/ReadVariableOpвwhileI
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
:
         АR
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
lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_47/split/ReadVariableOpReadVariableOp*lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_47/splitSplit%lstm_cell_47/split/split_dim:output:0)lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_47/MatMulMatMulstrided_slice_2:output:0lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_47/split_1/ReadVariableOpReadVariableOp,lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_47/split_1Split'lstm_cell_47/split_1/split_dim:output:0+lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_47/BiasAddBiasAddlstm_cell_47/MatMul:product:0lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_1BiasAddlstm_cell_47/MatMul_1:product:0lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_2BiasAddlstm_cell_47/MatMul_2:product:0lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_3BiasAddlstm_cell_47/MatMul_3:product:0lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_47/ReadVariableOpReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_47/strided_sliceStridedSlice#lstm_cell_47/ReadVariableOp:value:0)lstm_cell_47/strided_slice/stack:output:0+lstm_cell_47/strided_slice/stack_1:output:0+lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_47/MatMul_4MatMulzeros:output:0#lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_47/addAddV2lstm_cell_47/BiasAdd:output:0lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_47/MulMullstm_cell_47/add:z:0lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_47/Add_1AddV2lstm_cell_47/Mul:z:0lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_47/clip_by_value/MinimumMinimumlstm_cell_47/Add_1:z:0-lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_47/clip_by_valueMaximum&lstm_cell_47/clip_by_value/Minimum:z:0%lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_1ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_1StridedSlice%lstm_cell_47/ReadVariableOp_1:value:0+lstm_cell_47/strided_slice_1/stack:output:0-lstm_cell_47/strided_slice_1/stack_1:output:0-lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_5MatMulzeros:output:0%lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_2AddV2lstm_cell_47/BiasAdd_1:output:0lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_1Mullstm_cell_47/add_2:z:0lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_3AddV2lstm_cell_47/Mul_1:z:0lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_cell_47/Add_3:z:0/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_1Maximum(lstm_cell_47/clip_by_value_1/Minimum:z:0'lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_47/mul_2Mul lstm_cell_47/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_2ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_2StridedSlice%lstm_cell_47/ReadVariableOp_2:value:0+lstm_cell_47/strided_slice_2/stack:output:0-lstm_cell_47/strided_slice_2/stack_1:output:0-lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_6MatMulzeros:output:0%lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_4AddV2lstm_cell_47/BiasAdd_2:output:0lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/ReluRelulstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_47/mul_3Mullstm_cell_47/clip_by_value:z:0lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_47/add_5AddV2lstm_cell_47/mul_2:z:0lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_3ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_3StridedSlice%lstm_cell_47/ReadVariableOp_3:value:0+lstm_cell_47/strided_slice_3/stack:output:0-lstm_cell_47/strided_slice_3/stack_1:output:0-lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_7MatMulzeros:output:0%lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_6AddV2lstm_cell_47/BiasAdd_3:output:0lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_4Mullstm_cell_47/add_6:z:0lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_7AddV2lstm_cell_47/Mul_4:z:0lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_cell_47/Add_7:z:0/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_2Maximum(lstm_cell_47/clip_by_value_2/Minimum:z:0'lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_47/Relu_1Relulstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_47/mul_5Mul lstm_cell_47/clip_by_value_2:z:0!lstm_cell_47/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_47_split_readvariableop_resource,lstm_cell_47_split_1_readvariableop_resource$lstm_cell_47_readvariableop_resource*
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
while_body_382333*
condR
while_cond_382332*K
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
:
         @*
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
:         
@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_47/ReadVariableOp^lstm_cell_47/ReadVariableOp_1^lstm_cell_47/ReadVariableOp_2^lstm_cell_47/ReadVariableOp_3"^lstm_cell_47/split/ReadVariableOp$^lstm_cell_47/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         
А: : : 2>
lstm_cell_47/ReadVariableOp_1lstm_cell_47/ReadVariableOp_12>
lstm_cell_47/ReadVariableOp_2lstm_cell_47/ReadVariableOp_22>
lstm_cell_47/ReadVariableOp_3lstm_cell_47/ReadVariableOp_32:
lstm_cell_47/ReadVariableOplstm_cell_47/ReadVariableOp2F
!lstm_cell_47/split/ReadVariableOp!lstm_cell_47/split/ReadVariableOp2J
#lstm_cell_47/split_1/ReadVariableOp#lstm_cell_47/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         
А
 
_user_specified_nameinputs
┬
Ц
)__inference_dense_23_layer_call_fn_382738

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
D__inference_dense_23_layer_call_and_return_conditional_losses_378770o
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
╥7
Ж
C__inference_lstm_47_layer_call_and_return_conditional_losses_378218

inputs'
lstm_cell_47_378137:
АА"
lstm_cell_47_378139:	А&
lstm_cell_47_378141:	@А
identityИв$lstm_cell_47/StatefulPartitionedCallвwhileI
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
$lstm_cell_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_47_378137lstm_cell_47_378139lstm_cell_47_378141*
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_378091n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_47_378137lstm_cell_47_378139lstm_cell_47_378141*
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
while_body_378150*
condR
while_cond_378149*K
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
NoOpNoOp%^lstm_cell_47/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2L
$lstm_cell_47/StatefulPartitionedCall$lstm_cell_47/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╘J
к
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_378091

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
Т
╕
(__inference_lstm_47_layer_call_fn_381672
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_377971o
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
ы}
ж	
while_body_381821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_47_split_readvariableop_resource_0:
ААC
4while_lstm_cell_47_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_47_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_47_split_readvariableop_resource:
ААA
2while_lstm_cell_47_split_1_readvariableop_resource:	А=
*while_lstm_cell_47_readvariableop_resource:	@АИв!while/lstm_cell_47/ReadVariableOpв#while/lstm_cell_47/ReadVariableOp_1в#while/lstm_cell_47/ReadVariableOp_2в#while/lstm_cell_47/ReadVariableOp_3в'while/lstm_cell_47/split/ReadVariableOpв)while/lstm_cell_47/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_47/split/ReadVariableOpReadVariableOp2while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_47/splitSplit+while/lstm_cell_47/split/split_dim:output:0/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_47/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_47/split_1Split-while/lstm_cell_47/split_1/split_dim:output:01while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_47/BiasAddBiasAdd#while/lstm_cell_47/MatMul:product:0#while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_1BiasAdd%while/lstm_cell_47/MatMul_1:product:0#while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_2BiasAdd%while/lstm_cell_47/MatMul_2:product:0#while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_3BiasAdd%while/lstm_cell_47/MatMul_3:product:0#while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_47/ReadVariableOpReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_47/strided_sliceStridedSlice)while/lstm_cell_47/ReadVariableOp:value:0/while/lstm_cell_47/strided_slice/stack:output:01while/lstm_cell_47/strided_slice/stack_1:output:01while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_47/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_47/addAddV2#while/lstm_cell_47/BiasAdd:output:0%while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_47/MulMulwhile/lstm_cell_47/add:z:0!while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_47/Add_1AddV2while/lstm_cell_47/Mul:z:0#while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_47/clip_by_value/MinimumMinimumwhile/lstm_cell_47/Add_1:z:03while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_47/clip_by_valueMaximum,while/lstm_cell_47/clip_by_value/Minimum:z:0+while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_1ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_1StridedSlice+while/lstm_cell_47/ReadVariableOp_1:value:01while/lstm_cell_47/strided_slice_1/stack:output:03while/lstm_cell_47/strided_slice_1/stack_1:output:03while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_2AddV2%while/lstm_cell_47/BiasAdd_1:output:0%while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_1Mulwhile/lstm_cell_47/add_2:z:0#while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_3AddV2while/lstm_cell_47/Mul_1:z:0#while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_1/MinimumMinimumwhile/lstm_cell_47/Add_3:z:05while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_1Maximum.while/lstm_cell_47/clip_by_value_1/Minimum:z:0-while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_47/mul_2Mul&while/lstm_cell_47/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_2ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_2StridedSlice+while/lstm_cell_47/ReadVariableOp_2:value:01while/lstm_cell_47/strided_slice_2/stack:output:03while/lstm_cell_47/strided_slice_2/stack_1:output:03while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_4AddV2%while/lstm_cell_47/BiasAdd_2:output:0%while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_47/ReluReluwhile/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_47/mul_3Mul$while/lstm_cell_47/clip_by_value:z:0%while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_47/add_5AddV2while/lstm_cell_47/mul_2:z:0while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_3ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_3StridedSlice+while/lstm_cell_47/ReadVariableOp_3:value:01while/lstm_cell_47/strided_slice_3/stack:output:03while/lstm_cell_47/strided_slice_3/stack_1:output:03while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_6AddV2%while/lstm_cell_47/BiasAdd_3:output:0%while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_4Mulwhile/lstm_cell_47/add_6:z:0#while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_7AddV2while/lstm_cell_47/Mul_4:z:0#while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_2/MinimumMinimumwhile/lstm_cell_47/Add_7:z:05while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_2Maximum.while/lstm_cell_47/clip_by_value_2/Minimum:z:0-while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_47/Relu_1Reluwhile/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_47/mul_5Mul&while/lstm_cell_47/clip_by_value_2:z:0'while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_47/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_47/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_47/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_47/ReadVariableOp$^while/lstm_cell_47/ReadVariableOp_1$^while/lstm_cell_47/ReadVariableOp_2$^while/lstm_cell_47/ReadVariableOp_3(^while/lstm_cell_47/split/ReadVariableOp*^while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_47_readvariableop_resource,while_lstm_cell_47_readvariableop_resource_0"j
2while_lstm_cell_47_split_1_readvariableop_resource4while_lstm_cell_47_split_1_readvariableop_resource_0"f
0while_lstm_cell_47_split_readvariableop_resource2while_lstm_cell_47_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_47/ReadVariableOp_1#while/lstm_cell_47/ReadVariableOp_12J
#while/lstm_cell_47/ReadVariableOp_2#while/lstm_cell_47/ReadVariableOp_22J
#while/lstm_cell_47/ReadVariableOp_3#while/lstm_cell_47/ReadVariableOp_32F
!while/lstm_cell_47/ReadVariableOp!while/lstm_cell_47/ReadVariableOp2R
'while/lstm_cell_47/split/ReadVariableOp'while/lstm_cell_47/split/ReadVariableOp2V
)while/lstm_cell_47/split_1/ReadVariableOp)while/lstm_cell_47/split_1/ReadVariableOp:
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
Ы	
├
while_cond_381264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_381264___redundant_placeholder04
0while_while_cond_381264___redundant_placeholder14
0while_while_cond_381264___redundant_placeholder24
0while_while_cond_381264___redundant_placeholder3
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
ш
C__inference_lstm_46_layer_call_and_return_conditional_losses_381661

inputs=
*lstm_cell_46_split_readvariableop_resource:	А;
,lstm_cell_46_split_1_readvariableop_resource:	А8
$lstm_cell_46_readvariableop_resource:
АА
identityИвlstm_cell_46/ReadVariableOpвlstm_cell_46/ReadVariableOp_1вlstm_cell_46/ReadVariableOp_2вlstm_cell_46/ReadVariableOp_3в!lstm_cell_46/split/ReadVariableOpв#lstm_cell_46/split_1/ReadVariableOpвwhileI
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
:
         R
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
lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_46/split/ReadVariableOpReadVariableOp*lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_46/splitSplit%lstm_cell_46/split/split_dim:output:0)lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_46/MatMulMatMulstrided_slice_2:output:0lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_46/split_1/ReadVariableOpReadVariableOp,lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_46/split_1Split'lstm_cell_46/split_1/split_dim:output:0+lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_46/BiasAddBiasAddlstm_cell_46/MatMul:product:0lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_1BiasAddlstm_cell_46/MatMul_1:product:0lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_2BiasAddlstm_cell_46/MatMul_2:product:0lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_3BiasAddlstm_cell_46/MatMul_3:product:0lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_46/ReadVariableOpReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_46/strided_sliceStridedSlice#lstm_cell_46/ReadVariableOp:value:0)lstm_cell_46/strided_slice/stack:output:0+lstm_cell_46/strided_slice/stack_1:output:0+lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_46/MatMul_4MatMulzeros:output:0#lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_46/addAddV2lstm_cell_46/BiasAdd:output:0lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_46/MulMullstm_cell_46/add:z:0lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_46/Add_1AddV2lstm_cell_46/Mul:z:0lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_46/clip_by_value/MinimumMinimumlstm_cell_46/Add_1:z:0-lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_46/clip_by_valueMaximum&lstm_cell_46/clip_by_value/Minimum:z:0%lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_1ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_1StridedSlice%lstm_cell_46/ReadVariableOp_1:value:0+lstm_cell_46/strided_slice_1/stack:output:0-lstm_cell_46/strided_slice_1/stack_1:output:0-lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_5MatMulzeros:output:0%lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_2AddV2lstm_cell_46/BiasAdd_1:output:0lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_1Mullstm_cell_46/add_2:z:0lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_3AddV2lstm_cell_46/Mul_1:z:0lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_cell_46/Add_3:z:0/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_1Maximum(lstm_cell_46/clip_by_value_1/Minimum:z:0'lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_46/mul_2Mul lstm_cell_46/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_2ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_2StridedSlice%lstm_cell_46/ReadVariableOp_2:value:0+lstm_cell_46/strided_slice_2/stack:output:0-lstm_cell_46/strided_slice_2/stack_1:output:0-lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_6MatMulzeros:output:0%lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_4AddV2lstm_cell_46/BiasAdd_2:output:0lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_46/ReluRelulstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_46/mul_3Mullstm_cell_46/clip_by_value:z:0lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_46/add_5AddV2lstm_cell_46/mul_2:z:0lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_3ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_3StridedSlice%lstm_cell_46/ReadVariableOp_3:value:0+lstm_cell_46/strided_slice_3/stack:output:0-lstm_cell_46/strided_slice_3/stack_1:output:0-lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_7MatMulzeros:output:0%lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_6AddV2lstm_cell_46/BiasAdd_3:output:0lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_4Mullstm_cell_46/add_6:z:0lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_7AddV2lstm_cell_46/Mul_4:z:0lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_cell_46/Add_7:z:0/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_2Maximum(lstm_cell_46/clip_by_value_2/Minimum:z:0'lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_46/Relu_1Relulstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_46/mul_5Mul lstm_cell_46/clip_by_value_2:z:0!lstm_cell_46/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_46_split_readvariableop_resource,lstm_cell_46_split_1_readvariableop_resource$lstm_cell_46_readvariableop_resource*
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
while_body_381521*
condR
while_cond_381520*M
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
:
         А*
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
:         
Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         
АЦ
NoOpNoOp^lstm_cell_46/ReadVariableOp^lstm_cell_46/ReadVariableOp_1^lstm_cell_46/ReadVariableOp_2^lstm_cell_46/ReadVariableOp_3"^lstm_cell_46/split/ReadVariableOp$^lstm_cell_46/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2>
lstm_cell_46/ReadVariableOp_1lstm_cell_46/ReadVariableOp_12>
lstm_cell_46/ReadVariableOp_2lstm_cell_46/ReadVariableOp_22>
lstm_cell_46/ReadVariableOp_3lstm_cell_46/ReadVariableOp_32:
lstm_cell_46/ReadVariableOplstm_cell_46/ReadVariableOp2F
!lstm_cell_46/split/ReadVariableOp!lstm_cell_46/split/ReadVariableOp2J
#lstm_cell_46/split_1/ReadVariableOp#lstm_cell_46/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
б~
ж	
while_body_381009
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_46_split_readvariableop_resource_0:	АC
4while_lstm_cell_46_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_46_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_46_split_readvariableop_resource:	АA
2while_lstm_cell_46_split_1_readvariableop_resource:	А>
*while_lstm_cell_46_readvariableop_resource:
ААИв!while/lstm_cell_46/ReadVariableOpв#while/lstm_cell_46/ReadVariableOp_1в#while/lstm_cell_46/ReadVariableOp_2в#while/lstm_cell_46/ReadVariableOp_3в'while/lstm_cell_46/split/ReadVariableOpв)while/lstm_cell_46/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_46/split/ReadVariableOpReadVariableOp2while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_46/splitSplit+while/lstm_cell_46/split/split_dim:output:0/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_46/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_46/split_1Split-while/lstm_cell_46/split_1/split_dim:output:01while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_46/BiasAddBiasAdd#while/lstm_cell_46/MatMul:product:0#while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_1BiasAdd%while/lstm_cell_46/MatMul_1:product:0#while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_2BiasAdd%while/lstm_cell_46/MatMul_2:product:0#while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_3BiasAdd%while/lstm_cell_46/MatMul_3:product:0#while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_46/ReadVariableOpReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_46/strided_sliceStridedSlice)while/lstm_cell_46/ReadVariableOp:value:0/while/lstm_cell_46/strided_slice/stack:output:01while/lstm_cell_46/strided_slice/stack_1:output:01while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_46/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_46/addAddV2#while/lstm_cell_46/BiasAdd:output:0%while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_46/MulMulwhile/lstm_cell_46/add:z:0!while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_46/Add_1AddV2while/lstm_cell_46/Mul:z:0#while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_46/clip_by_value/MinimumMinimumwhile/lstm_cell_46/Add_1:z:03while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_46/clip_by_valueMaximum,while/lstm_cell_46/clip_by_value/Minimum:z:0+while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_1ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_1StridedSlice+while/lstm_cell_46/ReadVariableOp_1:value:01while/lstm_cell_46/strided_slice_1/stack:output:03while/lstm_cell_46/strided_slice_1/stack_1:output:03while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_2AddV2%while/lstm_cell_46/BiasAdd_1:output:0%while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_1Mulwhile/lstm_cell_46/add_2:z:0#while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_3AddV2while/lstm_cell_46/Mul_1:z:0#while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_1/MinimumMinimumwhile/lstm_cell_46/Add_3:z:05while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_1Maximum.while/lstm_cell_46/clip_by_value_1/Minimum:z:0-while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_46/mul_2Mul&while/lstm_cell_46/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_2ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_2StridedSlice+while/lstm_cell_46/ReadVariableOp_2:value:01while/lstm_cell_46/strided_slice_2/stack:output:03while/lstm_cell_46/strided_slice_2/stack_1:output:03while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_4AddV2%while/lstm_cell_46/BiasAdd_2:output:0%while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_46/ReluReluwhile/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_46/mul_3Mul$while/lstm_cell_46/clip_by_value:z:0%while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_46/add_5AddV2while/lstm_cell_46/mul_2:z:0while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_3ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_3StridedSlice+while/lstm_cell_46/ReadVariableOp_3:value:01while/lstm_cell_46/strided_slice_3/stack:output:03while/lstm_cell_46/strided_slice_3/stack_1:output:03while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_6AddV2%while/lstm_cell_46/BiasAdd_3:output:0%while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_4Mulwhile/lstm_cell_46/add_6:z:0#while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_7AddV2while/lstm_cell_46/Mul_4:z:0#while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_2/MinimumMinimumwhile/lstm_cell_46/Add_7:z:05while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_2Maximum.while/lstm_cell_46/clip_by_value_2/Minimum:z:0-while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_46/Relu_1Reluwhile/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_46/mul_5Mul&while/lstm_cell_46/clip_by_value_2:z:0'while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_46/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_46/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_46/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_46/ReadVariableOp$^while/lstm_cell_46/ReadVariableOp_1$^while/lstm_cell_46/ReadVariableOp_2$^while/lstm_cell_46/ReadVariableOp_3(^while/lstm_cell_46/split/ReadVariableOp*^while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_46_readvariableop_resource,while_lstm_cell_46_readvariableop_resource_0"j
2while_lstm_cell_46_split_1_readvariableop_resource4while_lstm_cell_46_split_1_readvariableop_resource_0"f
0while_lstm_cell_46_split_readvariableop_resource2while_lstm_cell_46_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_46/ReadVariableOp_1#while/lstm_cell_46/ReadVariableOp_12J
#while/lstm_cell_46/ReadVariableOp_2#while/lstm_cell_46/ReadVariableOp_22J
#while/lstm_cell_46/ReadVariableOp_3#while/lstm_cell_46/ReadVariableOp_32F
!while/lstm_cell_46/ReadVariableOp!while/lstm_cell_46/ReadVariableOp2R
'while/lstm_cell_46/split/ReadVariableOp'while/lstm_cell_46/split/ReadVariableOp2V
)while/lstm_cell_46/split_1/ReadVariableOp)while/lstm_cell_46/split_1/ReadVariableOp:
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
Т
╕
(__inference_lstm_47_layer_call_fn_381683
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_378218o
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
Ы	
├
while_cond_380752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_380752___redundant_placeholder04
0while_while_cond_380752___redundant_placeholder14
0while_while_cond_380752___redundant_placeholder24
0while_while_cond_380752___redundant_placeholder3
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
М
у
lstm_46_while_cond_379680,
(lstm_46_while_lstm_46_while_loop_counter2
.lstm_46_while_lstm_46_while_maximum_iterations
lstm_46_while_placeholder
lstm_46_while_placeholder_1
lstm_46_while_placeholder_2
lstm_46_while_placeholder_3.
*lstm_46_while_less_lstm_46_strided_slice_1D
@lstm_46_while_lstm_46_while_cond_379680___redundant_placeholder0D
@lstm_46_while_lstm_46_while_cond_379680___redundant_placeholder1D
@lstm_46_while_lstm_46_while_cond_379680___redundant_placeholder2D
@lstm_46_while_lstm_46_while_cond_379680___redundant_placeholder3
lstm_46_while_identity
В
lstm_46/while/LessLesslstm_46_while_placeholder*lstm_46_while_less_lstm_46_strided_slice_1*
T0*
_output_shapes
: [
lstm_46/while/IdentityIdentitylstm_46/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_46_while_identitylstm_46/while/Identity:output:0*(
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
_user_specified_name" lstm_46/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_46/while/loop_counter
№╕
╩	
I__inference_sequential_23_layer_call_and_return_conditional_losses_380079

inputsE
2lstm_46_lstm_cell_46_split_readvariableop_resource:	АC
4lstm_46_lstm_cell_46_split_1_readvariableop_resource:	А@
,lstm_46_lstm_cell_46_readvariableop_resource:
ААF
2lstm_47_lstm_cell_47_split_readvariableop_resource:
ААC
4lstm_47_lstm_cell_47_split_1_readvariableop_resource:	А?
,lstm_47_lstm_cell_47_readvariableop_resource:	@А9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:
identityИвdense_23/BiasAdd/ReadVariableOpвdense_23/MatMul/ReadVariableOpв#lstm_46/lstm_cell_46/ReadVariableOpв%lstm_46/lstm_cell_46/ReadVariableOp_1в%lstm_46/lstm_cell_46/ReadVariableOp_2в%lstm_46/lstm_cell_46/ReadVariableOp_3в)lstm_46/lstm_cell_46/split/ReadVariableOpв+lstm_46/lstm_cell_46/split_1/ReadVariableOpвlstm_46/whileв#lstm_47/lstm_cell_47/ReadVariableOpв%lstm_47/lstm_cell_47/ReadVariableOp_1в%lstm_47/lstm_cell_47/ReadVariableOp_2в%lstm_47/lstm_cell_47/ReadVariableOp_3в)lstm_47/lstm_cell_47/split/ReadVariableOpв+lstm_47/lstm_cell_47/split_1/ReadVariableOpвlstm_47/whileQ
lstm_46/ShapeShapeinputs*
T0*
_output_shapes
::э╧e
lstm_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_46/strided_sliceStridedSlicelstm_46/Shape:output:0$lstm_46/strided_slice/stack:output:0&lstm_46/strided_slice/stack_1:output:0&lstm_46/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_46/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АЛ
lstm_46/zeros/packedPacklstm_46/strided_slice:output:0lstm_46/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_46/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Е
lstm_46/zerosFilllstm_46/zeros/packed:output:0lstm_46/zeros/Const:output:0*
T0*(
_output_shapes
:         А[
lstm_46/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :АП
lstm_46/zeros_1/packedPacklstm_46/strided_slice:output:0!lstm_46/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_46/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
lstm_46/zeros_1Filllstm_46/zeros_1/packed:output:0lstm_46/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Аk
lstm_46/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_46/transpose	Transposeinputslstm_46/transpose/perm:output:0*
T0*+
_output_shapes
:
         b
lstm_46/Shape_1Shapelstm_46/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_46/strided_slice_1StridedSlicelstm_46/Shape_1:output:0&lstm_46/strided_slice_1/stack:output:0(lstm_46/strided_slice_1/stack_1:output:0(lstm_46/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_46/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_46/TensorArrayV2TensorListReserve,lstm_46/TensorArrayV2/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_46/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       °
/lstm_46/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_46/transpose:y:0Flstm_46/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
lstm_46/strided_slice_2StridedSlicelstm_46/transpose:y:0&lstm_46/strided_slice_2/stack:output:0(lstm_46/strided_slice_2/stack_1:output:0(lstm_46/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskf
$lstm_46/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Э
)lstm_46/lstm_cell_46/split/ReadVariableOpReadVariableOp2lstm_46_lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0с
lstm_46/lstm_cell_46/splitSplit-lstm_46/lstm_cell_46/split/split_dim:output:01lstm_46/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЯ
lstm_46/lstm_cell_46/MatMulMatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Аб
lstm_46/lstm_cell_46/MatMul_1MatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Аб
lstm_46/lstm_cell_46/MatMul_2MatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Аб
lstm_46/lstm_cell_46/MatMul_3MatMul lstm_46/strided_slice_2:output:0#lstm_46/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аh
&lstm_46/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_46/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4lstm_46_lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╫
lstm_46/lstm_cell_46/split_1Split/lstm_46/lstm_cell_46/split_1/split_dim:output:03lstm_46/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitи
lstm_46/lstm_cell_46/BiasAddBiasAdd%lstm_46/lstm_cell_46/MatMul:product:0%lstm_46/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Ам
lstm_46/lstm_cell_46/BiasAdd_1BiasAdd'lstm_46/lstm_cell_46/MatMul_1:product:0%lstm_46/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Ам
lstm_46/lstm_cell_46/BiasAdd_2BiasAdd'lstm_46/lstm_cell_46/MatMul_2:product:0%lstm_46/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Ам
lstm_46/lstm_cell_46/BiasAdd_3BiasAdd'lstm_46/lstm_cell_46/MatMul_3:product:0%lstm_46/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АТ
#lstm_46/lstm_cell_46/ReadVariableOpReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0y
(lstm_46/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_46/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*lstm_46/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"lstm_46/lstm_cell_46/strided_sliceStridedSlice+lstm_46/lstm_cell_46/ReadVariableOp:value:01lstm_46/lstm_cell_46/strided_slice/stack:output:03lstm_46/lstm_cell_46/strided_slice/stack_1:output:03lstm_46/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЯ
lstm_46/lstm_cell_46/MatMul_4MatMullstm_46/zeros:output:0+lstm_46/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         Ад
lstm_46/lstm_cell_46/addAddV2%lstm_46/lstm_cell_46/BiasAdd:output:0'lstm_46/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А_
lstm_46/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_46/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
lstm_46/lstm_cell_46/MulMullstm_46/lstm_cell_46/add:z:0#lstm_46/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АЫ
lstm_46/lstm_cell_46/Add_1AddV2lstm_46/lstm_cell_46/Mul:z:0%lstm_46/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аq
,lstm_46/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┐
*lstm_46/lstm_cell_46/clip_by_value/MinimumMinimumlstm_46/lstm_cell_46/Add_1:z:05lstm_46/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$lstm_46/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"lstm_46/lstm_cell_46/clip_by_valueMaximum.lstm_46/lstm_cell_46/clip_by_value/Minimum:z:0-lstm_46/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АФ
%lstm_46/lstm_cell_46/ReadVariableOp_1ReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_46/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_46/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_46/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_46/lstm_cell_46/strided_slice_1StridedSlice-lstm_46/lstm_cell_46/ReadVariableOp_1:value:03lstm_46/lstm_cell_46/strided_slice_1/stack:output:05lstm_46/lstm_cell_46/strided_slice_1/stack_1:output:05lstm_46/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_46/lstm_cell_46/MatMul_5MatMullstm_46/zeros:output:0-lstm_46/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Аи
lstm_46/lstm_cell_46/add_2AddV2'lstm_46/lstm_cell_46/BiasAdd_1:output:0'lstm_46/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         Аa
lstm_46/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_46/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_46/lstm_cell_46/Mul_1Mullstm_46/lstm_cell_46/add_2:z:0%lstm_46/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЭ
lstm_46/lstm_cell_46/Add_3AddV2lstm_46/lstm_cell_46/Mul_1:z:0%lstm_46/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аs
.lstm_46/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_46/lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_46/lstm_cell_46/Add_3:z:07lstm_46/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_46/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_46/lstm_cell_46/clip_by_value_1Maximum0lstm_46/lstm_cell_46/clip_by_value_1/Minimum:z:0/lstm_46/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АШ
lstm_46/lstm_cell_46/mul_2Mul(lstm_46/lstm_cell_46/clip_by_value_1:z:0lstm_46/zeros_1:output:0*
T0*(
_output_shapes
:         АФ
%lstm_46/lstm_cell_46/ReadVariableOp_2ReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_46/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_46/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_46/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_46/lstm_cell_46/strided_slice_2StridedSlice-lstm_46/lstm_cell_46/ReadVariableOp_2:value:03lstm_46/lstm_cell_46/strided_slice_2/stack:output:05lstm_46/lstm_cell_46/strided_slice_2/stack_1:output:05lstm_46/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_46/lstm_cell_46/MatMul_6MatMullstm_46/zeros:output:0-lstm_46/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Аи
lstm_46/lstm_cell_46/add_4AddV2'lstm_46/lstm_cell_46/BiasAdd_2:output:0'lstm_46/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аt
lstm_46/lstm_cell_46/ReluRelulstm_46/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         Ае
lstm_46/lstm_cell_46/mul_3Mul&lstm_46/lstm_cell_46/clip_by_value:z:0'lstm_46/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АЦ
lstm_46/lstm_cell_46/add_5AddV2lstm_46/lstm_cell_46/mul_2:z:0lstm_46/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АФ
%lstm_46/lstm_cell_46/ReadVariableOp_3ReadVariableOp,lstm_46_lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0{
*lstm_46/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  }
,lstm_46/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_46/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
$lstm_46/lstm_cell_46/strided_slice_3StridedSlice-lstm_46/lstm_cell_46/ReadVariableOp_3:value:03lstm_46/lstm_cell_46/strided_slice_3/stack:output:05lstm_46/lstm_cell_46/strided_slice_3/stack_1:output:05lstm_46/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskб
lstm_46/lstm_cell_46/MatMul_7MatMullstm_46/zeros:output:0-lstm_46/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Аи
lstm_46/lstm_cell_46/add_6AddV2'lstm_46/lstm_cell_46/BiasAdd_3:output:0'lstm_46/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         Аa
lstm_46/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_46/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ы
lstm_46/lstm_cell_46/Mul_4Mullstm_46/lstm_cell_46/add_6:z:0%lstm_46/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЭ
lstm_46/lstm_cell_46/Add_7AddV2lstm_46/lstm_cell_46/Mul_4:z:0%lstm_46/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аs
.lstm_46/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?├
,lstm_46/lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_46/lstm_cell_46/Add_7:z:07lstm_46/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аk
&lstm_46/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┼
$lstm_46/lstm_cell_46/clip_by_value_2Maximum0lstm_46/lstm_cell_46/clip_by_value_2/Minimum:z:0/lstm_46/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аv
lstm_46/lstm_cell_46/Relu_1Relulstm_46/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Ай
lstm_46/lstm_cell_46/mul_5Mul(lstm_46/lstm_cell_46/clip_by_value_2:z:0)lstm_46/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         Аv
%lstm_46/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╨
lstm_46/TensorArrayV2_1TensorListReserve.lstm_46/TensorArrayV2_1/element_shape:output:0 lstm_46/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_46/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_46/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_46/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ь
lstm_46/whileWhile#lstm_46/while/loop_counter:output:0)lstm_46/while/maximum_iterations:output:0lstm_46/time:output:0 lstm_46/TensorArrayV2_1:handle:0lstm_46/zeros:output:0lstm_46/zeros_1:output:0 lstm_46/strided_slice_1:output:0?lstm_46/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_46_lstm_cell_46_split_readvariableop_resource4lstm_46_lstm_cell_46_split_1_readvariableop_resource,lstm_46_lstm_cell_46_readvariableop_resource*
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
lstm_46_while_body_379681*%
condR
lstm_46_while_cond_379680*M
output_shapes<
:: : : : :         А:         А: : : : : *
parallel_iterations Й
8lstm_46/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   █
*lstm_46/TensorArrayV2Stack/TensorListStackTensorListStacklstm_46/while:output:3Alstm_46/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:
         А*
element_dtype0p
lstm_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_46/strided_slice_3StridedSlice3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_46/strided_slice_3/stack:output:0(lstm_46/strided_slice_3/stack_1:output:0(lstm_46/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskm
lstm_46/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
lstm_46/transpose_1	Transpose3lstm_46/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_46/transpose_1/perm:output:0*
T0*,
_output_shapes
:         
Аb
lstm_47/ShapeShapelstm_46/transpose_1:y:0*
T0*
_output_shapes
::э╧e
lstm_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
lstm_47/strided_sliceStridedSlicelstm_47/Shape:output:0$lstm_47/strided_slice/stack:output:0&lstm_47/strided_slice/stack_1:output:0&lstm_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_47/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Л
lstm_47/zeros/packedPacklstm_47/strided_slice:output:0lstm_47/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_47/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
lstm_47/zerosFilllstm_47/zeros/packed:output:0lstm_47/zeros/Const:output:0*
T0*'
_output_shapes
:         @Z
lstm_47/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@П
lstm_47/zeros_1/packedPacklstm_47/strided_slice:output:0!lstm_47/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_47/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
lstm_47/zeros_1Filllstm_47/zeros_1/packed:output:0lstm_47/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @k
lstm_47/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          П
lstm_47/transpose	Transposelstm_46/transpose_1:y:0lstm_47/transpose/perm:output:0*
T0*,
_output_shapes
:
         Аb
lstm_47/Shape_1Shapelstm_47/transpose:y:0*
T0*
_output_shapes
::э╧g
lstm_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_47/strided_slice_1StridedSlicelstm_47/Shape_1:output:0&lstm_47/strided_slice_1/stack:output:0(lstm_47/strided_slice_1/stack_1:output:0(lstm_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_47/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_47/TensorArrayV2TensorListReserve,lstm_47/TensorArrayV2/element_shape:output:0 lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=lstm_47/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   °
/lstm_47/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_47/transpose:y:0Flstm_47/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
lstm_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
lstm_47/strided_slice_2StridedSlicelstm_47/transpose:y:0&lstm_47/strided_slice_2/stack:output:0(lstm_47/strided_slice_2/stack_1:output:0(lstm_47/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         А*
shrink_axis_maskf
$lstm_47/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
)lstm_47/lstm_cell_47/split/ReadVariableOpReadVariableOp2lstm_47_lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0с
lstm_47/lstm_cell_47/splitSplit-lstm_47/lstm_cell_47/split/split_dim:output:01lstm_47/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЮ
lstm_47/lstm_cell_47/MatMulMatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @а
lstm_47/lstm_cell_47/MatMul_1MatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @а
lstm_47/lstm_cell_47/MatMul_2MatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @а
lstm_47/lstm_cell_47/MatMul_3MatMul lstm_47/strided_slice_2:output:0#lstm_47/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @h
&lstm_47/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
+lstm_47/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4lstm_47_lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
lstm_47/lstm_cell_47/split_1Split/lstm_47/lstm_cell_47/split_1/split_dim:output:03lstm_47/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitз
lstm_47/lstm_cell_47/BiasAddBiasAdd%lstm_47/lstm_cell_47/MatMul:product:0%lstm_47/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @л
lstm_47/lstm_cell_47/BiasAdd_1BiasAdd'lstm_47/lstm_cell_47/MatMul_1:product:0%lstm_47/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @л
lstm_47/lstm_cell_47/BiasAdd_2BiasAdd'lstm_47/lstm_cell_47/MatMul_2:product:0%lstm_47/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @л
lstm_47/lstm_cell_47/BiasAdd_3BiasAdd'lstm_47/lstm_cell_47/MatMul_3:product:0%lstm_47/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @С
#lstm_47/lstm_cell_47/ReadVariableOpReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0y
(lstm_47/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_47/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm_47/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"lstm_47/lstm_cell_47/strided_sliceStridedSlice+lstm_47/lstm_cell_47/ReadVariableOp:value:01lstm_47/lstm_cell_47/strided_slice/stack:output:03lstm_47/lstm_cell_47/strided_slice/stack_1:output:03lstm_47/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЮ
lstm_47/lstm_cell_47/MatMul_4MatMullstm_47/zeros:output:0+lstm_47/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @г
lstm_47/lstm_cell_47/addAddV2%lstm_47/lstm_cell_47/BiasAdd:output:0'lstm_47/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @_
lstm_47/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_47/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
lstm_47/lstm_cell_47/MulMullstm_47/lstm_cell_47/add:z:0#lstm_47/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ъ
lstm_47/lstm_cell_47/Add_1AddV2lstm_47/lstm_cell_47/Mul:z:0%lstm_47/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @q
,lstm_47/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╛
*lstm_47/lstm_cell_47/clip_by_value/MinimumMinimumlstm_47/lstm_cell_47/Add_1:z:05lstm_47/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$lstm_47/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"lstm_47/lstm_cell_47/clip_by_valueMaximum.lstm_47/lstm_cell_47/clip_by_value/Minimum:z:0-lstm_47/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @У
%lstm_47/lstm_cell_47/ReadVariableOp_1ReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_47/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   }
,lstm_47/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_47/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_47/lstm_cell_47/strided_slice_1StridedSlice-lstm_47/lstm_cell_47/ReadVariableOp_1:value:03lstm_47/lstm_cell_47/strided_slice_1/stack:output:05lstm_47/lstm_cell_47/strided_slice_1/stack_1:output:05lstm_47/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_47/lstm_cell_47/MatMul_5MatMullstm_47/zeros:output:0-lstm_47/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @з
lstm_47/lstm_cell_47/add_2AddV2'lstm_47/lstm_cell_47/BiasAdd_1:output:0'lstm_47/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @a
lstm_47/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_47/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_47/lstm_cell_47/Mul_1Mullstm_47/lstm_cell_47/add_2:z:0%lstm_47/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ь
lstm_47/lstm_cell_47/Add_3AddV2lstm_47/lstm_cell_47/Mul_1:z:0%lstm_47/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @s
.lstm_47/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_47/lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_47/lstm_cell_47/Add_3:z:07lstm_47/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_47/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_47/lstm_cell_47/clip_by_value_1Maximum0lstm_47/lstm_cell_47/clip_by_value_1/Minimum:z:0/lstm_47/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @Ч
lstm_47/lstm_cell_47/mul_2Mul(lstm_47/lstm_cell_47/clip_by_value_1:z:0lstm_47/zeros_1:output:0*
T0*'
_output_shapes
:         @У
%lstm_47/lstm_cell_47/ReadVariableOp_2ReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_47/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm_47/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_47/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_47/lstm_cell_47/strided_slice_2StridedSlice-lstm_47/lstm_cell_47/ReadVariableOp_2:value:03lstm_47/lstm_cell_47/strided_slice_2/stack:output:05lstm_47/lstm_cell_47/strided_slice_2/stack_1:output:05lstm_47/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_47/lstm_cell_47/MatMul_6MatMullstm_47/zeros:output:0-lstm_47/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @з
lstm_47/lstm_cell_47/add_4AddV2'lstm_47/lstm_cell_47/BiasAdd_2:output:0'lstm_47/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @s
lstm_47/lstm_cell_47/ReluRelulstm_47/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @д
lstm_47/lstm_cell_47/mul_3Mul&lstm_47/lstm_cell_47/clip_by_value:z:0'lstm_47/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @Х
lstm_47/lstm_cell_47/add_5AddV2lstm_47/lstm_cell_47/mul_2:z:0lstm_47/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @У
%lstm_47/lstm_cell_47/ReadVariableOp_3ReadVariableOp,lstm_47_lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0{
*lstm_47/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   }
,lstm_47/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_47/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▐
$lstm_47/lstm_cell_47/strided_slice_3StridedSlice-lstm_47/lstm_cell_47/ReadVariableOp_3:value:03lstm_47/lstm_cell_47/strided_slice_3/stack:output:05lstm_47/lstm_cell_47/strided_slice_3/stack_1:output:05lstm_47/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskа
lstm_47/lstm_cell_47/MatMul_7MatMullstm_47/zeros:output:0-lstm_47/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @з
lstm_47/lstm_cell_47/add_6AddV2'lstm_47/lstm_cell_47/BiasAdd_3:output:0'lstm_47/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @a
lstm_47/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>a
lstm_47/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
lstm_47/lstm_cell_47/Mul_4Mullstm_47/lstm_cell_47/add_6:z:0%lstm_47/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ь
lstm_47/lstm_cell_47/Add_7AddV2lstm_47/lstm_cell_47/Mul_4:z:0%lstm_47/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @s
.lstm_47/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
,lstm_47/lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_47/lstm_cell_47/Add_7:z:07lstm_47/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @k
&lstm_47/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ─
$lstm_47/lstm_cell_47/clip_by_value_2Maximum0lstm_47/lstm_cell_47/clip_by_value_2/Minimum:z:0/lstm_47/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @u
lstm_47/lstm_cell_47/Relu_1Relulstm_47/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @и
lstm_47/lstm_cell_47/mul_5Mul(lstm_47/lstm_cell_47/clip_by_value_2:z:0)lstm_47/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @v
%lstm_47/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╨
lstm_47/TensorArrayV2_1TensorListReserve.lstm_47/TensorArrayV2_1/element_shape:output:0 lstm_47/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
lstm_47/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_47/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_47/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ш
lstm_47/whileWhile#lstm_47/while/loop_counter:output:0)lstm_47/while/maximum_iterations:output:0lstm_47/time:output:0 lstm_47/TensorArrayV2_1:handle:0lstm_47/zeros:output:0lstm_47/zeros_1:output:0 lstm_47/strided_slice_1:output:0?lstm_47/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_47_lstm_cell_47_split_readvariableop_resource4lstm_47_lstm_cell_47_split_1_readvariableop_resource,lstm_47_lstm_cell_47_readvariableop_resource*
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
lstm_47_while_body_379933*%
condR
lstm_47_while_cond_379932*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations Й
8lstm_47/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┌
*lstm_47/TensorArrayV2Stack/TensorListStackTensorListStacklstm_47/while:output:3Alstm_47/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:
         @*
element_dtype0p
lstm_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
lstm_47/strided_slice_3StridedSlice3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_47/strided_slice_3/stack:output:0(lstm_47/strided_slice_3/stack_1:output:0(lstm_47/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskm
lstm_47/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
lstm_47/transpose_1	Transpose3lstm_47/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_47/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
@Ж
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Х
dense_23/MatMulMatMul lstm_47/strided_slice_3:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp$^lstm_46/lstm_cell_46/ReadVariableOp&^lstm_46/lstm_cell_46/ReadVariableOp_1&^lstm_46/lstm_cell_46/ReadVariableOp_2&^lstm_46/lstm_cell_46/ReadVariableOp_3*^lstm_46/lstm_cell_46/split/ReadVariableOp,^lstm_46/lstm_cell_46/split_1/ReadVariableOp^lstm_46/while$^lstm_47/lstm_cell_47/ReadVariableOp&^lstm_47/lstm_cell_47/ReadVariableOp_1&^lstm_47/lstm_cell_47/ReadVariableOp_2&^lstm_47/lstm_cell_47/ReadVariableOp_3*^lstm_47/lstm_cell_47/split/ReadVariableOp,^lstm_47/lstm_cell_47/split_1/ReadVariableOp^lstm_47/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2N
%lstm_46/lstm_cell_46/ReadVariableOp_1%lstm_46/lstm_cell_46/ReadVariableOp_12N
%lstm_46/lstm_cell_46/ReadVariableOp_2%lstm_46/lstm_cell_46/ReadVariableOp_22N
%lstm_46/lstm_cell_46/ReadVariableOp_3%lstm_46/lstm_cell_46/ReadVariableOp_32J
#lstm_46/lstm_cell_46/ReadVariableOp#lstm_46/lstm_cell_46/ReadVariableOp2V
)lstm_46/lstm_cell_46/split/ReadVariableOp)lstm_46/lstm_cell_46/split/ReadVariableOp2Z
+lstm_46/lstm_cell_46/split_1/ReadVariableOp+lstm_46/lstm_cell_46/split_1/ReadVariableOp2
lstm_46/whilelstm_46/while2N
%lstm_47/lstm_cell_47/ReadVariableOp_1%lstm_47/lstm_cell_47/ReadVariableOp_12N
%lstm_47/lstm_cell_47/ReadVariableOp_2%lstm_47/lstm_cell_47/ReadVariableOp_22N
%lstm_47/lstm_cell_47/ReadVariableOp_3%lstm_47/lstm_cell_47/ReadVariableOp_32J
#lstm_47/lstm_cell_47/ReadVariableOp#lstm_47/lstm_cell_47/ReadVariableOp2V
)lstm_47/lstm_cell_47/split/ReadVariableOp)lstm_47/lstm_cell_47/split/ReadVariableOp2Z
+lstm_47/lstm_cell_47/split_1/ReadVariableOp+lstm_47/lstm_cell_47/split_1/ReadVariableOp2
lstm_47/whilelstm_47/while:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ё	
╨
.__inference_sequential_23_layer_call_fn_379448
lstm_46_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_46_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_23_layer_call_and_return_conditional_losses_379408o
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
':         
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_46_input
ы}
ж	
while_body_382333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_47_split_readvariableop_resource_0:
ААC
4while_lstm_cell_47_split_1_readvariableop_resource_0:	А?
,while_lstm_cell_47_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_47_split_readvariableop_resource:
ААA
2while_lstm_cell_47_split_1_readvariableop_resource:	А=
*while_lstm_cell_47_readvariableop_resource:	@АИв!while/lstm_cell_47/ReadVariableOpв#while/lstm_cell_47/ReadVariableOp_1в#while/lstm_cell_47/ReadVariableOp_2в#while/lstm_cell_47/ReadVariableOp_3в'while/lstm_cell_47/split/ReadVariableOpв)while/lstm_cell_47/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0d
"while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'while/lstm_cell_47/split/ReadVariableOpReadVariableOp2while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0█
while/lstm_cell_47/splitSplit+while/lstm_cell_47/split/split_dim:output:0/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitк
while/lstm_cell_47/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @м
while/lstm_cell_47/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @f
$while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0═
while/lstm_cell_47/split_1Split-while/lstm_cell_47/split_1/split_dim:output:01while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitб
while/lstm_cell_47/BiasAddBiasAdd#while/lstm_cell_47/MatMul:product:0#while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_1BiasAdd%while/lstm_cell_47/MatMul_1:product:0#while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_2BiasAdd%while/lstm_cell_47/MatMul_2:product:0#while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @е
while/lstm_cell_47/BiasAdd_3BiasAdd%while/lstm_cell_47/MatMul_3:product:0#while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @П
!while/lstm_cell_47/ReadVariableOpReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0w
&while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_47/strided_sliceStridedSlice)while/lstm_cell_47/ReadVariableOp:value:0/while/lstm_cell_47/strided_slice/stack:output:01while/lstm_cell_47/strided_slice/stack_1:output:01while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЧ
while/lstm_cell_47/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Э
while/lstm_cell_47/addAddV2#while/lstm_cell_47/BiasAdd:output:0%while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @]
while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?О
while/lstm_cell_47/MulMulwhile/lstm_cell_47/add:z:0!while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @Ф
while/lstm_cell_47/Add_1AddV2while/lstm_cell_47/Mul:z:0#while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @o
*while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
(while/lstm_cell_47/clip_by_value/MinimumMinimumwhile/lstm_cell_47/Add_1:z:03while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @g
"while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╕
 while/lstm_cell_47/clip_by_valueMaximum,while/lstm_cell_47/clip_by_value/Minimum:z:0+while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_1ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_1StridedSlice+while/lstm_cell_47/ReadVariableOp_1:value:01while/lstm_cell_47/strided_slice_1/stack:output:03while/lstm_cell_47/strided_slice_1/stack_1:output:03while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_2AddV2%while/lstm_cell_47/BiasAdd_1:output:0%while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_1Mulwhile/lstm_cell_47/add_2:z:0#while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_3AddV2while/lstm_cell_47/Mul_1:z:0#while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_1/MinimumMinimumwhile/lstm_cell_47/Add_3:z:05while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_1Maximum.while/lstm_cell_47/clip_by_value_1/Minimum:z:0-while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @О
while/lstm_cell_47/mul_2Mul&while/lstm_cell_47/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_2ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_2StridedSlice+while/lstm_cell_47/ReadVariableOp_2:value:01while/lstm_cell_47/strided_slice_2/stack:output:03while/lstm_cell_47/strided_slice_2/stack_1:output:03while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_4AddV2%while/lstm_cell_47/BiasAdd_2:output:0%while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @o
while/lstm_cell_47/ReluReluwhile/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_47/mul_3Mul$while/lstm_cell_47/clip_by_value:z:0%while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @П
while/lstm_cell_47/add_5AddV2while/lstm_cell_47/mul_2:z:0while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @С
#while/lstm_cell_47/ReadVariableOp_3ReadVariableOp,while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   {
*while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
"while/lstm_cell_47/strided_slice_3StridedSlice+while/lstm_cell_47/ReadVariableOp_3:value:01while/lstm_cell_47/strided_slice_3/stack:output:03while/lstm_cell_47/strided_slice_3/stack_1:output:03while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell_47/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @б
while/lstm_cell_47/add_6AddV2%while/lstm_cell_47/BiasAdd_3:output:0%while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @_
while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
while/lstm_cell_47/Mul_4Mulwhile/lstm_cell_47/add_6:z:0#while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Ц
while/lstm_cell_47/Add_7AddV2while/lstm_cell_47/Mul_4:z:0#while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @q
,while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
*while/lstm_cell_47/clip_by_value_2/MinimumMinimumwhile/lstm_cell_47/Add_7:z:05while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @i
$while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╛
"while/lstm_cell_47/clip_by_value_2Maximum.while/lstm_cell_47/clip_by_value_2/Minimum:z:0-while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @q
while/lstm_cell_47/Relu_1Reluwhile/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @в
while/lstm_cell_47/mul_5Mul&while/lstm_cell_47/clip_by_value_2:z:0'while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_47/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_47/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @y
while/Identity_5Identitywhile/lstm_cell_47/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:         @╕

while/NoOpNoOp"^while/lstm_cell_47/ReadVariableOp$^while/lstm_cell_47/ReadVariableOp_1$^while/lstm_cell_47/ReadVariableOp_2$^while/lstm_cell_47/ReadVariableOp_3(^while/lstm_cell_47/split/ReadVariableOp*^while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_47_readvariableop_resource,while_lstm_cell_47_readvariableop_resource_0"j
2while_lstm_cell_47_split_1_readvariableop_resource4while_lstm_cell_47_split_1_readvariableop_resource_0"f
0while_lstm_cell_47_split_readvariableop_resource2while_lstm_cell_47_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2J
#while/lstm_cell_47/ReadVariableOp_1#while/lstm_cell_47/ReadVariableOp_12J
#while/lstm_cell_47/ReadVariableOp_2#while/lstm_cell_47/ReadVariableOp_22J
#while/lstm_cell_47/ReadVariableOp_3#while/lstm_cell_47/ReadVariableOp_32F
!while/lstm_cell_47/ReadVariableOp!while/lstm_cell_47/ReadVariableOp2R
'while/lstm_cell_47/split/ReadVariableOp'while/lstm_cell_47/split/ReadVariableOp2V
)while/lstm_cell_47/split_1/ReadVariableOp)while/lstm_cell_47/split_1/ReadVariableOp:
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
МK
к
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_377427

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
ю
ў
-__inference_lstm_cell_47_layer_call_fn_382977

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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_377889o
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
▌И
ш
C__inference_lstm_46_layer_call_and_return_conditional_losses_378489

inputs=
*lstm_cell_46_split_readvariableop_resource:	А;
,lstm_cell_46_split_1_readvariableop_resource:	А8
$lstm_cell_46_readvariableop_resource:
АА
identityИвlstm_cell_46/ReadVariableOpвlstm_cell_46/ReadVariableOp_1вlstm_cell_46/ReadVariableOp_2вlstm_cell_46/ReadVariableOp_3в!lstm_cell_46/split/ReadVariableOpв#lstm_cell_46/split_1/ReadVariableOpвwhileI
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
:
         R
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
lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_46/split/ReadVariableOpReadVariableOp*lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_46/splitSplit%lstm_cell_46/split/split_dim:output:0)lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_46/MatMulMatMulstrided_slice_2:output:0lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_46/split_1/ReadVariableOpReadVariableOp,lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_46/split_1Split'lstm_cell_46/split_1/split_dim:output:0+lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_46/BiasAddBiasAddlstm_cell_46/MatMul:product:0lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_1BiasAddlstm_cell_46/MatMul_1:product:0lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_2BiasAddlstm_cell_46/MatMul_2:product:0lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_3BiasAddlstm_cell_46/MatMul_3:product:0lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_46/ReadVariableOpReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_46/strided_sliceStridedSlice#lstm_cell_46/ReadVariableOp:value:0)lstm_cell_46/strided_slice/stack:output:0+lstm_cell_46/strided_slice/stack_1:output:0+lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_46/MatMul_4MatMulzeros:output:0#lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_46/addAddV2lstm_cell_46/BiasAdd:output:0lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_46/MulMullstm_cell_46/add:z:0lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_46/Add_1AddV2lstm_cell_46/Mul:z:0lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_46/clip_by_value/MinimumMinimumlstm_cell_46/Add_1:z:0-lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_46/clip_by_valueMaximum&lstm_cell_46/clip_by_value/Minimum:z:0%lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_1ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_1StridedSlice%lstm_cell_46/ReadVariableOp_1:value:0+lstm_cell_46/strided_slice_1/stack:output:0-lstm_cell_46/strided_slice_1/stack_1:output:0-lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_5MatMulzeros:output:0%lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_2AddV2lstm_cell_46/BiasAdd_1:output:0lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_1Mullstm_cell_46/add_2:z:0lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_3AddV2lstm_cell_46/Mul_1:z:0lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_cell_46/Add_3:z:0/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_1Maximum(lstm_cell_46/clip_by_value_1/Minimum:z:0'lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_46/mul_2Mul lstm_cell_46/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_2ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_2StridedSlice%lstm_cell_46/ReadVariableOp_2:value:0+lstm_cell_46/strided_slice_2/stack:output:0-lstm_cell_46/strided_slice_2/stack_1:output:0-lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_6MatMulzeros:output:0%lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_4AddV2lstm_cell_46/BiasAdd_2:output:0lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_46/ReluRelulstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_46/mul_3Mullstm_cell_46/clip_by_value:z:0lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_46/add_5AddV2lstm_cell_46/mul_2:z:0lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_3ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_3StridedSlice%lstm_cell_46/ReadVariableOp_3:value:0+lstm_cell_46/strided_slice_3/stack:output:0-lstm_cell_46/strided_slice_3/stack_1:output:0-lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_7MatMulzeros:output:0%lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_6AddV2lstm_cell_46/BiasAdd_3:output:0lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_4Mullstm_cell_46/add_6:z:0lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_7AddV2lstm_cell_46/Mul_4:z:0lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_cell_46/Add_7:z:0/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_2Maximum(lstm_cell_46/clip_by_value_2/Minimum:z:0'lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_46/Relu_1Relulstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_46/mul_5Mul lstm_cell_46/clip_by_value_2:z:0!lstm_cell_46/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_46_split_readvariableop_resource,lstm_cell_46_split_1_readvariableop_resource$lstm_cell_46_readvariableop_resource*
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
while_body_378349*
condR
while_cond_378348*M
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
:
         А*
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
:         
Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         
АЦ
NoOpNoOp^lstm_cell_46/ReadVariableOp^lstm_cell_46/ReadVariableOp_1^lstm_cell_46/ReadVariableOp_2^lstm_cell_46/ReadVariableOp_3"^lstm_cell_46/split/ReadVariableOp$^lstm_cell_46/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2>
lstm_cell_46/ReadVariableOp_1lstm_cell_46/ReadVariableOp_12>
lstm_cell_46/ReadVariableOp_2lstm_cell_46/ReadVariableOp_22>
lstm_cell_46/ReadVariableOp_3lstm_cell_46/ReadVariableOp_32:
lstm_cell_46/ReadVariableOplstm_cell_46/ReadVariableOp2F
!lstm_cell_46/split/ReadVariableOp!lstm_cell_46/split/ReadVariableOp2J
#lstm_cell_46/split_1/ReadVariableOp#lstm_cell_46/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
▌И
ш
C__inference_lstm_46_layer_call_and_return_conditional_losses_379353

inputs=
*lstm_cell_46_split_readvariableop_resource:	А;
,lstm_cell_46_split_1_readvariableop_resource:	А8
$lstm_cell_46_readvariableop_resource:
АА
identityИвlstm_cell_46/ReadVariableOpвlstm_cell_46/ReadVariableOp_1вlstm_cell_46/ReadVariableOp_2вlstm_cell_46/ReadVariableOp_3в!lstm_cell_46/split/ReadVariableOpв#lstm_cell_46/split_1/ReadVariableOpвwhileI
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
:
         R
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
lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_46/split/ReadVariableOpReadVariableOp*lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_46/splitSplit%lstm_cell_46/split/split_dim:output:0)lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_46/MatMulMatMulstrided_slice_2:output:0lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_46/split_1/ReadVariableOpReadVariableOp,lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_46/split_1Split'lstm_cell_46/split_1/split_dim:output:0+lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_46/BiasAddBiasAddlstm_cell_46/MatMul:product:0lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_1BiasAddlstm_cell_46/MatMul_1:product:0lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_2BiasAddlstm_cell_46/MatMul_2:product:0lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_3BiasAddlstm_cell_46/MatMul_3:product:0lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_46/ReadVariableOpReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_46/strided_sliceStridedSlice#lstm_cell_46/ReadVariableOp:value:0)lstm_cell_46/strided_slice/stack:output:0+lstm_cell_46/strided_slice/stack_1:output:0+lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_46/MatMul_4MatMulzeros:output:0#lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_46/addAddV2lstm_cell_46/BiasAdd:output:0lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_46/MulMullstm_cell_46/add:z:0lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_46/Add_1AddV2lstm_cell_46/Mul:z:0lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_46/clip_by_value/MinimumMinimumlstm_cell_46/Add_1:z:0-lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_46/clip_by_valueMaximum&lstm_cell_46/clip_by_value/Minimum:z:0%lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_1ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_1StridedSlice%lstm_cell_46/ReadVariableOp_1:value:0+lstm_cell_46/strided_slice_1/stack:output:0-lstm_cell_46/strided_slice_1/stack_1:output:0-lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_5MatMulzeros:output:0%lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_2AddV2lstm_cell_46/BiasAdd_1:output:0lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_1Mullstm_cell_46/add_2:z:0lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_3AddV2lstm_cell_46/Mul_1:z:0lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_cell_46/Add_3:z:0/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_1Maximum(lstm_cell_46/clip_by_value_1/Minimum:z:0'lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_46/mul_2Mul lstm_cell_46/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_2ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_2StridedSlice%lstm_cell_46/ReadVariableOp_2:value:0+lstm_cell_46/strided_slice_2/stack:output:0-lstm_cell_46/strided_slice_2/stack_1:output:0-lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_6MatMulzeros:output:0%lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_4AddV2lstm_cell_46/BiasAdd_2:output:0lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_46/ReluRelulstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_46/mul_3Mullstm_cell_46/clip_by_value:z:0lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_46/add_5AddV2lstm_cell_46/mul_2:z:0lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_3ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_3StridedSlice%lstm_cell_46/ReadVariableOp_3:value:0+lstm_cell_46/strided_slice_3/stack:output:0-lstm_cell_46/strided_slice_3/stack_1:output:0-lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_7MatMulzeros:output:0%lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_6AddV2lstm_cell_46/BiasAdd_3:output:0lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_4Mullstm_cell_46/add_6:z:0lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_7AddV2lstm_cell_46/Mul_4:z:0lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_cell_46/Add_7:z:0/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_2Maximum(lstm_cell_46/clip_by_value_2/Minimum:z:0'lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_46/Relu_1Relulstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_46/mul_5Mul lstm_cell_46/clip_by_value_2:z:0!lstm_cell_46/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_46_split_readvariableop_resource,lstm_cell_46_split_1_readvariableop_resource$lstm_cell_46_readvariableop_resource*
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
while_body_379213*
condR
while_cond_379212*M
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
:
         А*
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
:         
Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         
АЦ
NoOpNoOp^lstm_cell_46/ReadVariableOp^lstm_cell_46/ReadVariableOp_1^lstm_cell_46/ReadVariableOp_2^lstm_cell_46/ReadVariableOp_3"^lstm_cell_46/split/ReadVariableOp$^lstm_cell_46/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2>
lstm_cell_46/ReadVariableOp_1lstm_cell_46/ReadVariableOp_12>
lstm_cell_46/ReadVariableOp_2lstm_cell_46/ReadVariableOp_22>
lstm_cell_46/ReadVariableOp_3lstm_cell_46/ReadVariableOp_32:
lstm_cell_46/ReadVariableOplstm_cell_46/ReadVariableOp2F
!lstm_cell_46/split/ReadVariableOp!lstm_cell_46/split/ReadVariableOp2J
#lstm_cell_46/split_1/ReadVariableOp#lstm_cell_46/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
▌И
ш
C__inference_lstm_46_layer_call_and_return_conditional_losses_381405

inputs=
*lstm_cell_46_split_readvariableop_resource:	А;
,lstm_cell_46_split_1_readvariableop_resource:	А8
$lstm_cell_46_readvariableop_resource:
АА
identityИвlstm_cell_46/ReadVariableOpвlstm_cell_46/ReadVariableOp_1вlstm_cell_46/ReadVariableOp_2вlstm_cell_46/ReadVariableOp_3в!lstm_cell_46/split/ReadVariableOpв#lstm_cell_46/split_1/ReadVariableOpвwhileI
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
:
         R
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
lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_46/split/ReadVariableOpReadVariableOp*lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_46/splitSplit%lstm_cell_46/split/split_dim:output:0)lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_46/MatMulMatMulstrided_slice_2:output:0lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_46/split_1/ReadVariableOpReadVariableOp,lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_46/split_1Split'lstm_cell_46/split_1/split_dim:output:0+lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_46/BiasAddBiasAddlstm_cell_46/MatMul:product:0lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_1BiasAddlstm_cell_46/MatMul_1:product:0lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_2BiasAddlstm_cell_46/MatMul_2:product:0lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_3BiasAddlstm_cell_46/MatMul_3:product:0lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_46/ReadVariableOpReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_46/strided_sliceStridedSlice#lstm_cell_46/ReadVariableOp:value:0)lstm_cell_46/strided_slice/stack:output:0+lstm_cell_46/strided_slice/stack_1:output:0+lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_46/MatMul_4MatMulzeros:output:0#lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_46/addAddV2lstm_cell_46/BiasAdd:output:0lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_46/MulMullstm_cell_46/add:z:0lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_46/Add_1AddV2lstm_cell_46/Mul:z:0lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_46/clip_by_value/MinimumMinimumlstm_cell_46/Add_1:z:0-lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_46/clip_by_valueMaximum&lstm_cell_46/clip_by_value/Minimum:z:0%lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_1ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_1StridedSlice%lstm_cell_46/ReadVariableOp_1:value:0+lstm_cell_46/strided_slice_1/stack:output:0-lstm_cell_46/strided_slice_1/stack_1:output:0-lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_5MatMulzeros:output:0%lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_2AddV2lstm_cell_46/BiasAdd_1:output:0lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_1Mullstm_cell_46/add_2:z:0lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_3AddV2lstm_cell_46/Mul_1:z:0lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_cell_46/Add_3:z:0/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_1Maximum(lstm_cell_46/clip_by_value_1/Minimum:z:0'lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_46/mul_2Mul lstm_cell_46/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_2ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_2StridedSlice%lstm_cell_46/ReadVariableOp_2:value:0+lstm_cell_46/strided_slice_2/stack:output:0-lstm_cell_46/strided_slice_2/stack_1:output:0-lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_6MatMulzeros:output:0%lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_4AddV2lstm_cell_46/BiasAdd_2:output:0lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_46/ReluRelulstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_46/mul_3Mullstm_cell_46/clip_by_value:z:0lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_46/add_5AddV2lstm_cell_46/mul_2:z:0lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_3ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_3StridedSlice%lstm_cell_46/ReadVariableOp_3:value:0+lstm_cell_46/strided_slice_3/stack:output:0-lstm_cell_46/strided_slice_3/stack_1:output:0-lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_7MatMulzeros:output:0%lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_6AddV2lstm_cell_46/BiasAdd_3:output:0lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_4Mullstm_cell_46/add_6:z:0lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_7AddV2lstm_cell_46/Mul_4:z:0lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_cell_46/Add_7:z:0/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_2Maximum(lstm_cell_46/clip_by_value_2/Minimum:z:0'lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_46/Relu_1Relulstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_46/mul_5Mul lstm_cell_46/clip_by_value_2:z:0!lstm_cell_46/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_46_split_readvariableop_resource,lstm_cell_46_split_1_readvariableop_resource$lstm_cell_46_readvariableop_resource*
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
while_body_381265*
condR
while_cond_381264*M
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
:
         А*
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
:         
Аc
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         
АЦ
NoOpNoOp^lstm_cell_46/ReadVariableOp^lstm_cell_46/ReadVariableOp_1^lstm_cell_46/ReadVariableOp_2^lstm_cell_46/ReadVariableOp_3"^lstm_cell_46/split/ReadVariableOp$^lstm_cell_46/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         
: : : 2>
lstm_cell_46/ReadVariableOp_1lstm_cell_46/ReadVariableOp_12>
lstm_cell_46/ReadVariableOp_2lstm_cell_46/ReadVariableOp_22>
lstm_cell_46/ReadVariableOp_3lstm_cell_46/ReadVariableOp_32:
lstm_cell_46/ReadVariableOplstm_cell_46/ReadVariableOp2F
!lstm_cell_46/split/ReadVariableOp!lstm_cell_46/split/ReadVariableOp2J
#lstm_cell_46/split_1/ReadVariableOp#lstm_cell_46/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
█П
╛
lstm_47_while_body_380447,
(lstm_47_while_lstm_47_while_loop_counter2
.lstm_47_while_lstm_47_while_maximum_iterations
lstm_47_while_placeholder
lstm_47_while_placeholder_1
lstm_47_while_placeholder_2
lstm_47_while_placeholder_3+
'lstm_47_while_lstm_47_strided_slice_1_0g
clstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0N
:lstm_47_while_lstm_cell_47_split_readvariableop_resource_0:
ААK
<lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0:	АG
4lstm_47_while_lstm_cell_47_readvariableop_resource_0:	@А
lstm_47_while_identity
lstm_47_while_identity_1
lstm_47_while_identity_2
lstm_47_while_identity_3
lstm_47_while_identity_4
lstm_47_while_identity_5)
%lstm_47_while_lstm_47_strided_slice_1e
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorL
8lstm_47_while_lstm_cell_47_split_readvariableop_resource:
ААI
:lstm_47_while_lstm_cell_47_split_1_readvariableop_resource:	АE
2lstm_47_while_lstm_cell_47_readvariableop_resource:	@АИв)lstm_47/while/lstm_cell_47/ReadVariableOpв+lstm_47/while/lstm_cell_47/ReadVariableOp_1в+lstm_47/while/lstm_cell_47/ReadVariableOp_2в+lstm_47/while/lstm_cell_47/ReadVariableOp_3в/lstm_47/while/lstm_cell_47/split/ReadVariableOpв1lstm_47/while/lstm_cell_47/split_1/ReadVariableOpР
?lstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    А   ╧
1lstm_47/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0lstm_47_while_placeholderHlstm_47/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         А*
element_dtype0l
*lstm_47/while/lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :м
/lstm_47/while/lstm_cell_47/split/ReadVariableOpReadVariableOp:lstm_47_while_lstm_cell_47_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0є
 lstm_47/while/lstm_cell_47/splitSplit3lstm_47/while/lstm_cell_47/split/split_dim:output:07lstm_47/while/lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split┬
!lstm_47/while/lstm_cell_47/MatMulMatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @─
#lstm_47/while/lstm_cell_47/MatMul_1MatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @─
#lstm_47/while/lstm_cell_47/MatMul_2MatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @─
#lstm_47/while/lstm_cell_47/MatMul_3MatMul8lstm_47/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_47/while/lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @n
,lstm_47/while/lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : л
1lstm_47/while/lstm_cell_47/split_1/ReadVariableOpReadVariableOp<lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0х
"lstm_47/while/lstm_cell_47/split_1Split5lstm_47/while/lstm_cell_47/split_1/split_dim:output:09lstm_47/while/lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split╣
"lstm_47/while/lstm_cell_47/BiasAddBiasAdd+lstm_47/while/lstm_cell_47/MatMul:product:0+lstm_47/while/lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @╜
$lstm_47/while/lstm_cell_47/BiasAdd_1BiasAdd-lstm_47/while/lstm_cell_47/MatMul_1:product:0+lstm_47/while/lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @╜
$lstm_47/while/lstm_cell_47/BiasAdd_2BiasAdd-lstm_47/while/lstm_cell_47/MatMul_2:product:0+lstm_47/while/lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @╜
$lstm_47/while/lstm_cell_47/BiasAdd_3BiasAdd-lstm_47/while/lstm_cell_47/MatMul_3:product:0+lstm_47/while/lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Я
)lstm_47/while/lstm_cell_47/ReadVariableOpReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0
.lstm_47/while/lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0lstm_47/while/lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Б
0lstm_47/while/lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Є
(lstm_47/while/lstm_cell_47/strided_sliceStridedSlice1lstm_47/while/lstm_cell_47/ReadVariableOp:value:07lstm_47/while/lstm_cell_47/strided_slice/stack:output:09lstm_47/while/lstm_cell_47/strided_slice/stack_1:output:09lstm_47/while/lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskп
#lstm_47/while/lstm_cell_47/MatMul_4MatMullstm_47_while_placeholder_21lstm_47/while/lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @╡
lstm_47/while/lstm_cell_47/addAddV2+lstm_47/while/lstm_cell_47/BiasAdd:output:0-lstm_47/while/lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @e
 lstm_47/while/lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_47/while/lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?ж
lstm_47/while/lstm_cell_47/MulMul"lstm_47/while/lstm_cell_47/add:z:0)lstm_47/while/lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @м
 lstm_47/while/lstm_cell_47/Add_1AddV2"lstm_47/while/lstm_cell_47/Mul:z:0+lstm_47/while/lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @w
2lstm_47/while/lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╨
0lstm_47/while/lstm_cell_47/clip_by_value/MinimumMinimum$lstm_47/while/lstm_cell_47/Add_1:z:0;lstm_47/while/lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @o
*lstm_47/while/lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╨
(lstm_47/while/lstm_cell_47/clip_by_valueMaximum4lstm_47/while/lstm_cell_47/clip_by_value/Minimum:z:03lstm_47/while/lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @б
+lstm_47/while/lstm_cell_47/ReadVariableOp_1ReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_47/while/lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   Г
2lstm_47/while/lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_47/while/lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_47/while/lstm_cell_47/strided_slice_1StridedSlice3lstm_47/while/lstm_cell_47/ReadVariableOp_1:value:09lstm_47/while/lstm_cell_47/strided_slice_1/stack:output:0;lstm_47/while/lstm_cell_47/strided_slice_1/stack_1:output:0;lstm_47/while/lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_47/while/lstm_cell_47/MatMul_5MatMullstm_47_while_placeholder_23lstm_47/while/lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @╣
 lstm_47/while/lstm_cell_47/add_2AddV2-lstm_47/while/lstm_cell_47/BiasAdd_1:output:0-lstm_47/while/lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @g
"lstm_47/while/lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_47/while/lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_47/while/lstm_cell_47/Mul_1Mul$lstm_47/while/lstm_cell_47/add_2:z:0+lstm_47/while/lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @о
 lstm_47/while/lstm_cell_47/Add_3AddV2$lstm_47/while/lstm_cell_47/Mul_1:z:0+lstm_47/while/lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @y
4lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_47/while/lstm_cell_47/clip_by_value_1/MinimumMinimum$lstm_47/while/lstm_cell_47/Add_3:z:0=lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_47/while/lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_47/while/lstm_cell_47/clip_by_value_1Maximum6lstm_47/while/lstm_cell_47/clip_by_value_1/Minimum:z:05lstm_47/while/lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @ж
 lstm_47/while/lstm_cell_47/mul_2Mul.lstm_47/while/lstm_cell_47/clip_by_value_1:z:0lstm_47_while_placeholder_3*
T0*'
_output_shapes
:         @б
+lstm_47/while/lstm_cell_47/ReadVariableOp_2ReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_47/while/lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   Г
2lstm_47/while/lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_47/while/lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_47/while/lstm_cell_47/strided_slice_2StridedSlice3lstm_47/while/lstm_cell_47/ReadVariableOp_2:value:09lstm_47/while/lstm_cell_47/strided_slice_2/stack:output:0;lstm_47/while/lstm_cell_47/strided_slice_2/stack_1:output:0;lstm_47/while/lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_47/while/lstm_cell_47/MatMul_6MatMullstm_47_while_placeholder_23lstm_47/while/lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @╣
 lstm_47/while/lstm_cell_47/add_4AddV2-lstm_47/while/lstm_cell_47/BiasAdd_2:output:0-lstm_47/while/lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @
lstm_47/while/lstm_cell_47/ReluRelu$lstm_47/while/lstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @╢
 lstm_47/while/lstm_cell_47/mul_3Mul,lstm_47/while/lstm_cell_47/clip_by_value:z:0-lstm_47/while/lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @з
 lstm_47/while/lstm_cell_47/add_5AddV2$lstm_47/while/lstm_cell_47/mul_2:z:0$lstm_47/while/lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @б
+lstm_47/while/lstm_cell_47/ReadVariableOp_3ReadVariableOp4lstm_47_while_lstm_cell_47_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Б
0lstm_47/while/lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   Г
2lstm_47/while/lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2lstm_47/while/lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      №
*lstm_47/while/lstm_cell_47/strided_slice_3StridedSlice3lstm_47/while/lstm_cell_47/ReadVariableOp_3:value:09lstm_47/while/lstm_cell_47/strided_slice_3/stack:output:0;lstm_47/while/lstm_cell_47/strided_slice_3/stack_1:output:0;lstm_47/while/lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask▒
#lstm_47/while/lstm_cell_47/MatMul_7MatMullstm_47_while_placeholder_23lstm_47/while/lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @╣
 lstm_47/while/lstm_cell_47/add_6AddV2-lstm_47/while/lstm_cell_47/BiasAdd_3:output:0-lstm_47/while/lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @g
"lstm_47/while/lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>g
"lstm_47/while/lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?м
 lstm_47/while/lstm_cell_47/Mul_4Mul$lstm_47/while/lstm_cell_47/add_6:z:0+lstm_47/while/lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @о
 lstm_47/while/lstm_cell_47/Add_7AddV2$lstm_47/while/lstm_cell_47/Mul_4:z:0+lstm_47/while/lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @y
4lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
2lstm_47/while/lstm_cell_47/clip_by_value_2/MinimumMinimum$lstm_47/while/lstm_cell_47/Add_7:z:0=lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @q
,lstm_47/while/lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╓
*lstm_47/while/lstm_cell_47/clip_by_value_2Maximum6lstm_47/while/lstm_cell_47/clip_by_value_2/Minimum:z:05lstm_47/while/lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @Б
!lstm_47/while/lstm_cell_47/Relu_1Relu$lstm_47/while/lstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @║
 lstm_47/while/lstm_cell_47/mul_5Mul.lstm_47/while/lstm_cell_47/clip_by_value_2:z:0/lstm_47/while/lstm_cell_47/Relu_1:activations:0*
T0*'
_output_shapes
:         @х
2lstm_47/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_47_while_placeholder_1lstm_47_while_placeholder$lstm_47/while/lstm_cell_47/mul_5:z:0*
_output_shapes
: *
element_dtype0:щш╥U
lstm_47/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_47/while/addAddV2lstm_47_while_placeholderlstm_47/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_47/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_47/while/add_1AddV2(lstm_47_while_lstm_47_while_loop_counterlstm_47/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_47/while/IdentityIdentitylstm_47/while/add_1:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: К
lstm_47/while/Identity_1Identity.lstm_47_while_lstm_47_while_maximum_iterations^lstm_47/while/NoOp*
T0*
_output_shapes
: q
lstm_47/while/Identity_2Identitylstm_47/while/add:z:0^lstm_47/while/NoOp*
T0*
_output_shapes
: Ю
lstm_47/while/Identity_3IdentityBlstm_47/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_47/while/NoOp*
T0*
_output_shapes
: С
lstm_47/while/Identity_4Identity$lstm_47/while/lstm_cell_47/mul_5:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:         @С
lstm_47/while/Identity_5Identity$lstm_47/while/lstm_cell_47/add_5:z:0^lstm_47/while/NoOp*
T0*'
_output_shapes
:         @Ё
lstm_47/while/NoOpNoOp*^lstm_47/while/lstm_cell_47/ReadVariableOp,^lstm_47/while/lstm_cell_47/ReadVariableOp_1,^lstm_47/while/lstm_cell_47/ReadVariableOp_2,^lstm_47/while/lstm_cell_47/ReadVariableOp_30^lstm_47/while/lstm_cell_47/split/ReadVariableOp2^lstm_47/while/lstm_cell_47/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_47_while_identity_1!lstm_47/while/Identity_1:output:0"=
lstm_47_while_identity_2!lstm_47/while/Identity_2:output:0"=
lstm_47_while_identity_3!lstm_47/while/Identity_3:output:0"=
lstm_47_while_identity_4!lstm_47/while/Identity_4:output:0"=
lstm_47_while_identity_5!lstm_47/while/Identity_5:output:0"9
lstm_47_while_identitylstm_47/while/Identity:output:0"P
%lstm_47_while_lstm_47_strided_slice_1'lstm_47_while_lstm_47_strided_slice_1_0"j
2lstm_47_while_lstm_cell_47_readvariableop_resource4lstm_47_while_lstm_cell_47_readvariableop_resource_0"z
:lstm_47_while_lstm_cell_47_split_1_readvariableop_resource<lstm_47_while_lstm_cell_47_split_1_readvariableop_resource_0"v
8lstm_47_while_lstm_cell_47_split_readvariableop_resource:lstm_47_while_lstm_cell_47_split_readvariableop_resource_0"╚
alstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensorclstm_47_while_tensorarrayv2read_tensorlistgetitem_lstm_47_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2Z
+lstm_47/while/lstm_cell_47/ReadVariableOp_1+lstm_47/while/lstm_cell_47/ReadVariableOp_12Z
+lstm_47/while/lstm_cell_47/ReadVariableOp_2+lstm_47/while/lstm_cell_47/ReadVariableOp_22Z
+lstm_47/while/lstm_cell_47/ReadVariableOp_3+lstm_47/while/lstm_cell_47/ReadVariableOp_32V
)lstm_47/while/lstm_cell_47/ReadVariableOp)lstm_47/while/lstm_cell_47/ReadVariableOp2b
/lstm_47/while/lstm_cell_47/split/ReadVariableOp/lstm_47/while/lstm_cell_47/split/ReadVariableOp2f
1lstm_47/while/lstm_cell_47/split_1/ReadVariableOp1lstm_47/while/lstm_cell_47/split_1/ReadVariableOp:
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
_user_specified_name" lstm_47/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_47/while/loop_counter
б~
ж	
while_body_381265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_46_split_readvariableop_resource_0:	АC
4while_lstm_cell_46_split_1_readvariableop_resource_0:	А@
,while_lstm_cell_46_readvariableop_resource_0:
АА
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_46_split_readvariableop_resource:	АA
2while_lstm_cell_46_split_1_readvariableop_resource:	А>
*while_lstm_cell_46_readvariableop_resource:
ААИв!while/lstm_cell_46/ReadVariableOpв#while/lstm_cell_46/ReadVariableOp_1в#while/lstm_cell_46/ReadVariableOp_2в#while/lstm_cell_46/ReadVariableOp_3в'while/lstm_cell_46/split/ReadVariableOpв)while/lstm_cell_46/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0d
"while/lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
'while/lstm_cell_46/split/ReadVariableOpReadVariableOp2while_lstm_cell_46_split_readvariableop_resource_0*
_output_shapes
:	А*
dtype0█
while/lstm_cell_46/splitSplit+while/lstm_cell_46/split/split_dim:output:0/while/lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitл
while/lstm_cell_46/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         Ан
while/lstm_cell_46/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         Аf
$while/lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ы
)while/lstm_cell_46/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_46_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
while/lstm_cell_46/split_1Split-while/lstm_cell_46/split_1/split_dim:output:01while/lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitв
while/lstm_cell_46/BiasAddBiasAdd#while/lstm_cell_46/MatMul:product:0#while/lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_1BiasAdd%while/lstm_cell_46/MatMul_1:product:0#while/lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_2BiasAdd%while/lstm_cell_46/MatMul_2:product:0#while/lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         Аж
while/lstm_cell_46/BiasAdd_3BiasAdd%while/lstm_cell_46/MatMul_3:product:0#while/lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АР
!while/lstm_cell_46/ReadVariableOpReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0w
&while/lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   y
(while/lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
 while/lstm_cell_46/strided_sliceStridedSlice)while/lstm_cell_46/ReadVariableOp:value:0/while/lstm_cell_46/strided_slice/stack:output:01while/lstm_cell_46/strided_slice/stack_1:output:01while/lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskШ
while/lstm_cell_46/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_46/addAddV2#while/lstm_cell_46/BiasAdd:output:0%while/lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         А]
while/lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?П
while/lstm_cell_46/MulMulwhile/lstm_cell_46/add:z:0!while/lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АХ
while/lstm_cell_46/Add_1AddV2while/lstm_cell_46/Mul:z:0#while/lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аo
*while/lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
(while/lstm_cell_46/clip_by_value/MinimumMinimumwhile/lstm_cell_46/Add_1:z:03while/lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аg
"while/lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ╣
 while/lstm_cell_46/clip_by_valueMaximum,while/lstm_cell_46/clip_by_value/Minimum:z:0+while/lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_1ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   {
*while/lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_1StridedSlice+while/lstm_cell_46/ReadVariableOp_1:value:01while/lstm_cell_46/strided_slice_1/stack:output:03while/lstm_cell_46/strided_slice_1/stack_1:output:03while/lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_2AddV2%while/lstm_cell_46/BiasAdd_1:output:0%while/lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_1Mulwhile/lstm_cell_46/add_2:z:0#while/lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_3AddV2while/lstm_cell_46/Mul_1:z:0#while/lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_1/MinimumMinimumwhile/lstm_cell_46/Add_3:z:05while/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_1Maximum.while/lstm_cell_46/clip_by_value_1/Minimum:z:0-while/lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АП
while/lstm_cell_46/mul_2Mul&while/lstm_cell_46/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_2ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_2StridedSlice+while/lstm_cell_46/ReadVariableOp_2:value:01while/lstm_cell_46/strided_slice_2/stack:output:03while/lstm_cell_46/strided_slice_2/stack_1:output:03while/lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_4AddV2%while/lstm_cell_46/BiasAdd_2:output:0%while/lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аp
while/lstm_cell_46/ReluReluwhile/lstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АЯ
while/lstm_cell_46/mul_3Mul$while/lstm_cell_46/clip_by_value:z:0%while/lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         АР
while/lstm_cell_46/add_5AddV2while/lstm_cell_46/mul_2:z:0while/lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АТ
#while/lstm_cell_46/ReadVariableOp_3ReadVariableOp,while_lstm_cell_46_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0y
(while/lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  {
*while/lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
"while/lstm_cell_46/strided_slice_3StridedSlice+while/lstm_cell_46/ReadVariableOp_3:value:01while/lstm_cell_46/strided_slice_3/stack:output:03while/lstm_cell_46/strided_slice_3/stack_1:output:03while/lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЪ
while/lstm_cell_46/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         Ав
while/lstm_cell_46/add_6AddV2%while/lstm_cell_46/BiasAdd_3:output:0%while/lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         А_
while/lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>_
while/lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
while/lstm_cell_46/Mul_4Mulwhile/lstm_cell_46/add_6:z:0#while/lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЧ
while/lstm_cell_46/Add_7AddV2while/lstm_cell_46/Mul_4:z:0#while/lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аq
,while/lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
*while/lstm_cell_46/clip_by_value_2/MinimumMinimumwhile/lstm_cell_46/Add_7:z:05while/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аi
$while/lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ┐
"while/lstm_cell_46/clip_by_value_2Maximum.while/lstm_cell_46/clip_by_value_2/Minimum:z:0-while/lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аr
while/lstm_cell_46/Relu_1Reluwhile/lstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         Аг
while/lstm_cell_46/mul_5Mul&while/lstm_cell_46/clip_by_value_2:z:0'while/lstm_cell_46/Relu_1:activations:0*
T0*(
_output_shapes
:         А┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_46/mul_5:z:0*
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
while/Identity_4Identitywhile/lstm_cell_46/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Аz
while/Identity_5Identitywhile/lstm_cell_46/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         А╕

while/NoOpNoOp"^while/lstm_cell_46/ReadVariableOp$^while/lstm_cell_46/ReadVariableOp_1$^while/lstm_cell_46/ReadVariableOp_2$^while/lstm_cell_46/ReadVariableOp_3(^while/lstm_cell_46/split/ReadVariableOp*^while/lstm_cell_46/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_46_readvariableop_resource,while_lstm_cell_46_readvariableop_resource_0"j
2while_lstm_cell_46_split_1_readvariableop_resource4while_lstm_cell_46_split_1_readvariableop_resource_0"f
0while_lstm_cell_46_split_readvariableop_resource2while_lstm_cell_46_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         А:         А: : : : : 2J
#while/lstm_cell_46/ReadVariableOp_1#while/lstm_cell_46/ReadVariableOp_12J
#while/lstm_cell_46/ReadVariableOp_2#while/lstm_cell_46/ReadVariableOp_22J
#while/lstm_cell_46/ReadVariableOp_3#while/lstm_cell_46/ReadVariableOp_32F
!while/lstm_cell_46/ReadVariableOp!while/lstm_cell_46/ReadVariableOp2R
'while/lstm_cell_46/split/ReadVariableOp'while/lstm_cell_46/split/ReadVariableOp2V
)while/lstm_cell_46/split_1/ReadVariableOp)while/lstm_cell_46/split_1/ReadVariableOp:
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
$__inference_signature_wrapper_379523
lstm_46_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_46_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_377303o
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
':         
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         

'
_user_specified_namelstm_46_input
ЫЙ
ъ
C__inference_lstm_46_layer_call_and_return_conditional_losses_381149
inputs_0=
*lstm_cell_46_split_readvariableop_resource:	А;
,lstm_cell_46_split_1_readvariableop_resource:	А8
$lstm_cell_46_readvariableop_resource:
АА
identityИвlstm_cell_46/ReadVariableOpвlstm_cell_46/ReadVariableOp_1вlstm_cell_46/ReadVariableOp_2вlstm_cell_46/ReadVariableOp_3в!lstm_cell_46/split/ReadVariableOpв#lstm_cell_46/split_1/ReadVariableOpвwhileK
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
lstm_cell_46/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
!lstm_cell_46/split/ReadVariableOpReadVariableOp*lstm_cell_46_split_readvariableop_resource*
_output_shapes
:	А*
dtype0╔
lstm_cell_46/splitSplit%lstm_cell_46/split/split_dim:output:0)lstm_cell_46/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А:	А:	А:	А*
	num_splitЗ
lstm_cell_46/MatMulMatMulstrided_slice_2:output:0lstm_cell_46/split:output:0*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_46/split:output:1*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_46/split:output:2*
T0*(
_output_shapes
:         АЙ
lstm_cell_46/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_46/split:output:3*
T0*(
_output_shapes
:         А`
lstm_cell_46/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_46/split_1/ReadVariableOpReadVariableOp,lstm_cell_46_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
lstm_cell_46/split_1Split'lstm_cell_46/split_1/split_dim:output:0+lstm_cell_46/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:А:А:А:А*
	num_splitР
lstm_cell_46/BiasAddBiasAddlstm_cell_46/MatMul:product:0lstm_cell_46/split_1:output:0*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_1BiasAddlstm_cell_46/MatMul_1:product:0lstm_cell_46/split_1:output:1*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_2BiasAddlstm_cell_46/MatMul_2:product:0lstm_cell_46/split_1:output:2*
T0*(
_output_shapes
:         АФ
lstm_cell_46/BiasAdd_3BiasAddlstm_cell_46/MatMul_3:product:0lstm_cell_46/split_1:output:3*
T0*(
_output_shapes
:         АВ
lstm_cell_46/ReadVariableOpReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0q
 lstm_cell_46/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_46/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   s
"lstm_cell_46/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      о
lstm_cell_46/strided_sliceStridedSlice#lstm_cell_46/ReadVariableOp:value:0)lstm_cell_46/strided_slice/stack:output:0+lstm_cell_46/strided_slice/stack_1:output:0+lstm_cell_46/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЗ
lstm_cell_46/MatMul_4MatMulzeros:output:0#lstm_cell_46/strided_slice:output:0*
T0*(
_output_shapes
:         АМ
lstm_cell_46/addAddV2lstm_cell_46/BiasAdd:output:0lstm_cell_46/MatMul_4:product:0*
T0*(
_output_shapes
:         АW
lstm_cell_46/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_46/MulMullstm_cell_46/add:z:0lstm_cell_46/Const:output:0*
T0*(
_output_shapes
:         АГ
lstm_cell_46/Add_1AddV2lstm_cell_46/Mul:z:0lstm_cell_46/Const_1:output:0*
T0*(
_output_shapes
:         Аi
$lstm_cell_46/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
"lstm_cell_46/clip_by_value/MinimumMinimumlstm_cell_46/Add_1:z:0-lstm_cell_46/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         Аa
lstm_cell_46/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    з
lstm_cell_46/clip_by_valueMaximum&lstm_cell_46/clip_by_value/Minimum:z:0%lstm_cell_46/clip_by_value/y:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_1ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_46/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_1StridedSlice%lstm_cell_46/ReadVariableOp_1:value:0+lstm_cell_46/strided_slice_1/stack:output:0-lstm_cell_46/strided_slice_1/stack_1:output:0-lstm_cell_46/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_5MatMulzeros:output:0%lstm_cell_46/strided_slice_1:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_2AddV2lstm_cell_46/BiasAdd_1:output:0lstm_cell_46/MatMul_5:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_1Mullstm_cell_46/add_2:z:0lstm_cell_46/Const_2:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_3AddV2lstm_cell_46/Mul_1:z:0lstm_cell_46/Const_3:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_1/MinimumMinimumlstm_cell_46/Add_3:z:0/lstm_cell_46/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_1Maximum(lstm_cell_46/clip_by_value_1/Minimum:z:0'lstm_cell_46/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:         АА
lstm_cell_46/mul_2Mul lstm_cell_46/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_2ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_46/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_2StridedSlice%lstm_cell_46/ReadVariableOp_2:value:0+lstm_cell_46/strided_slice_2/stack:output:0-lstm_cell_46/strided_slice_2/stack_1:output:0-lstm_cell_46/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_6MatMulzeros:output:0%lstm_cell_46/strided_slice_2:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_4AddV2lstm_cell_46/BiasAdd_2:output:0lstm_cell_46/MatMul_6:product:0*
T0*(
_output_shapes
:         Аd
lstm_cell_46/ReluRelulstm_cell_46/add_4:z:0*
T0*(
_output_shapes
:         АН
lstm_cell_46/mul_3Mullstm_cell_46/clip_by_value:z:0lstm_cell_46/Relu:activations:0*
T0*(
_output_shapes
:         А~
lstm_cell_46/add_5AddV2lstm_cell_46/mul_2:z:0lstm_cell_46/mul_3:z:0*
T0*(
_output_shapes
:         АД
lstm_cell_46/ReadVariableOp_3ReadVariableOp$lstm_cell_46_readvariableop_resource* 
_output_shapes
:
АА*
dtype0s
"lstm_cell_46/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    А  u
$lstm_cell_46/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_46/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╕
lstm_cell_46/strided_slice_3StridedSlice%lstm_cell_46/ReadVariableOp_3:value:0+lstm_cell_46/strided_slice_3/stack:output:0-lstm_cell_46/strided_slice_3/stack_1:output:0-lstm_cell_46/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
АА*

begin_mask*
end_maskЙ
lstm_cell_46/MatMul_7MatMulzeros:output:0%lstm_cell_46/strided_slice_3:output:0*
T0*(
_output_shapes
:         АР
lstm_cell_46/add_6AddV2lstm_cell_46/BiasAdd_3:output:0lstm_cell_46/MatMul_7:product:0*
T0*(
_output_shapes
:         АY
lstm_cell_46/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_46/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Г
lstm_cell_46/Mul_4Mullstm_cell_46/add_6:z:0lstm_cell_46/Const_4:output:0*
T0*(
_output_shapes
:         АЕ
lstm_cell_46/Add_7AddV2lstm_cell_46/Mul_4:z:0lstm_cell_46/Const_5:output:0*
T0*(
_output_shapes
:         Аk
&lstm_cell_46/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
$lstm_cell_46/clip_by_value_2/MinimumMinimumlstm_cell_46/Add_7:z:0/lstm_cell_46/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:         Аc
lstm_cell_46/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    н
lstm_cell_46/clip_by_value_2Maximum(lstm_cell_46/clip_by_value_2/Minimum:z:0'lstm_cell_46/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:         Аf
lstm_cell_46/Relu_1Relulstm_cell_46/add_5:z:0*
T0*(
_output_shapes
:         АС
lstm_cell_46/mul_5Mul lstm_cell_46/clip_by_value_2:z:0!lstm_cell_46/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_46_split_readvariableop_resource,lstm_cell_46_split_1_readvariableop_resource$lstm_cell_46_readvariableop_resource*
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
while_body_381009*
condR
while_cond_381008*M
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
NoOpNoOp^lstm_cell_46/ReadVariableOp^lstm_cell_46/ReadVariableOp_1^lstm_cell_46/ReadVariableOp_2^lstm_cell_46/ReadVariableOp_3"^lstm_cell_46/split/ReadVariableOp$^lstm_cell_46/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2>
lstm_cell_46/ReadVariableOp_1lstm_cell_46/ReadVariableOp_12>
lstm_cell_46/ReadVariableOp_2lstm_cell_46/ReadVariableOp_22>
lstm_cell_46/ReadVariableOp_3lstm_cell_46/ReadVariableOp_32:
lstm_cell_46/ReadVariableOplstm_cell_46/ReadVariableOp2F
!lstm_cell_46/split/ReadVariableOp!lstm_cell_46/split/ReadVariableOp2J
#lstm_cell_46/split_1/ReadVariableOp#lstm_cell_46/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
гЙ
·
"__inference__traced_restore_383484
file_prefix2
 assignvariableop_dense_23_kernel:@.
 assignvariableop_1_dense_23_bias:A
.assignvariableop_2_lstm_46_lstm_cell_46_kernel:	АL
8assignvariableop_3_lstm_46_lstm_cell_46_recurrent_kernel:
АА;
,assignvariableop_4_lstm_46_lstm_cell_46_bias:	АB
.assignvariableop_5_lstm_47_lstm_cell_47_kernel:
ААK
8assignvariableop_6_lstm_47_lstm_cell_47_recurrent_kernel:	@А;
,assignvariableop_7_lstm_47_lstm_cell_47_bias:	А#
assignvariableop_8_beta_1: #
assignvariableop_9_beta_2: #
assignvariableop_10_decay: +
!assignvariableop_11_learning_rate: '
assignvariableop_12_adam_iter:	 #
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_23_kernel_m:@6
(assignvariableop_16_adam_dense_23_bias_m:I
6assignvariableop_17_adam_lstm_46_lstm_cell_46_kernel_m:	АT
@assignvariableop_18_adam_lstm_46_lstm_cell_46_recurrent_kernel_m:
ААC
4assignvariableop_19_adam_lstm_46_lstm_cell_46_bias_m:	АJ
6assignvariableop_20_adam_lstm_47_lstm_cell_47_kernel_m:
ААS
@assignvariableop_21_adam_lstm_47_lstm_cell_47_recurrent_kernel_m:	@АC
4assignvariableop_22_adam_lstm_47_lstm_cell_47_bias_m:	А<
*assignvariableop_23_adam_dense_23_kernel_v:@6
(assignvariableop_24_adam_dense_23_bias_v:I
6assignvariableop_25_adam_lstm_46_lstm_cell_46_kernel_v:	АT
@assignvariableop_26_adam_lstm_46_lstm_cell_46_recurrent_kernel_v:
ААC
4assignvariableop_27_adam_lstm_46_lstm_cell_46_bias_v:	АJ
6assignvariableop_28_adam_lstm_47_lstm_cell_47_kernel_v:
ААS
@assignvariableop_29_adam_lstm_47_lstm_cell_47_recurrent_kernel_v:	@АC
4assignvariableop_30_adam_lstm_47_lstm_cell_47_bias_v:	А
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
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_46_lstm_cell_46_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_46_lstm_cell_46_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_46_lstm_cell_46_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_47_lstm_cell_47_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_47_lstm_cell_47_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_47_lstm_cell_47_biasIdentity_7:output:0"/device:CPU:0*&
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_23_kernel_mIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_23_bias_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_46_lstm_cell_46_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_46_lstm_cell_46_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_46_lstm_cell_46_bias_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_47_lstm_cell_47_kernel_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_47_lstm_cell_47_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_47_lstm_cell_47_bias_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_23_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_23_bias_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_46_lstm_cell_46_kernel_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_46_lstm_cell_46_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_46_lstm_cell_46_bias_vIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_47_lstm_cell_47_kernel_vIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_47_lstm_cell_47_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_47_lstm_cell_47_bias_vIdentity_30:output:0"/device:CPU:0*&
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
Ы	
├
while_cond_377687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_377687___redundant_placeholder04
0while_while_cond_377687___redundant_placeholder14
0while_while_cond_377687___redundant_placeholder24
0while_while_cond_377687___redundant_placeholder3
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
иИ
ш
C__inference_lstm_47_layer_call_and_return_conditional_losses_379075

inputs>
*lstm_cell_47_split_readvariableop_resource:
АА;
,lstm_cell_47_split_1_readvariableop_resource:	А7
$lstm_cell_47_readvariableop_resource:	@А
identityИвlstm_cell_47/ReadVariableOpвlstm_cell_47/ReadVariableOp_1вlstm_cell_47/ReadVariableOp_2вlstm_cell_47/ReadVariableOp_3в!lstm_cell_47/split/ReadVariableOpв#lstm_cell_47/split_1/ReadVariableOpвwhileI
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
:
         АR
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
lstm_cell_47/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :О
!lstm_cell_47/split/ReadVariableOpReadVariableOp*lstm_cell_47_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╔
lstm_cell_47/splitSplit%lstm_cell_47/split/split_dim:output:0)lstm_cell_47/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitЖ
lstm_cell_47/MatMulMatMulstrided_slice_2:output:0lstm_cell_47/split:output:0*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_47/split:output:1*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_47/split:output:2*
T0*'
_output_shapes
:         @И
lstm_cell_47/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_47/split:output:3*
T0*'
_output_shapes
:         @`
lstm_cell_47/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Н
#lstm_cell_47/split_1/ReadVariableOpReadVariableOp,lstm_cell_47_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
lstm_cell_47/split_1Split'lstm_cell_47/split_1/split_dim:output:0+lstm_cell_47/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitП
lstm_cell_47/BiasAddBiasAddlstm_cell_47/MatMul:product:0lstm_cell_47/split_1:output:0*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_1BiasAddlstm_cell_47/MatMul_1:product:0lstm_cell_47/split_1:output:1*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_2BiasAddlstm_cell_47/MatMul_2:product:0lstm_cell_47/split_1:output:2*
T0*'
_output_shapes
:         @У
lstm_cell_47/BiasAdd_3BiasAddlstm_cell_47/MatMul_3:product:0lstm_cell_47/split_1:output:3*
T0*'
_output_shapes
:         @Б
lstm_cell_47/ReadVariableOpReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0q
 lstm_cell_47/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_47/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   s
"lstm_cell_47/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
lstm_cell_47/strided_sliceStridedSlice#lstm_cell_47/ReadVariableOp:value:0)lstm_cell_47/strided_slice/stack:output:0+lstm_cell_47/strided_slice/stack_1:output:0+lstm_cell_47/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЖ
lstm_cell_47/MatMul_4MatMulzeros:output:0#lstm_cell_47/strided_slice:output:0*
T0*'
_output_shapes
:         @Л
lstm_cell_47/addAddV2lstm_cell_47/BiasAdd:output:0lstm_cell_47/MatMul_4:product:0*
T0*'
_output_shapes
:         @W
lstm_cell_47/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_47/MulMullstm_cell_47/add:z:0lstm_cell_47/Const:output:0*
T0*'
_output_shapes
:         @В
lstm_cell_47/Add_1AddV2lstm_cell_47/Mul:z:0lstm_cell_47/Const_1:output:0*
T0*'
_output_shapes
:         @i
$lstm_cell_47/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
"lstm_cell_47/clip_by_value/MinimumMinimumlstm_cell_47/Add_1:z:0-lstm_cell_47/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         @a
lstm_cell_47/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ж
lstm_cell_47/clip_by_valueMaximum&lstm_cell_47/clip_by_value/Minimum:z:0%lstm_cell_47/clip_by_value/y:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_1ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm_cell_47/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_1StridedSlice%lstm_cell_47/ReadVariableOp_1:value:0+lstm_cell_47/strided_slice_1/stack:output:0-lstm_cell_47/strided_slice_1/stack_1:output:0-lstm_cell_47/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_5MatMulzeros:output:0%lstm_cell_47/strided_slice_1:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_2AddV2lstm_cell_47/BiasAdd_1:output:0lstm_cell_47/MatMul_5:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_1Mullstm_cell_47/add_2:z:0lstm_cell_47/Const_2:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_3AddV2lstm_cell_47/Mul_1:z:0lstm_cell_47/Const_3:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_1/MinimumMinimumlstm_cell_47/Add_3:z:0/lstm_cell_47/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_1Maximum(lstm_cell_47/clip_by_value_1/Minimum:z:0'lstm_cell_47/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         @
lstm_cell_47/mul_2Mul lstm_cell_47/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_2ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   u
$lstm_cell_47/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_2StridedSlice%lstm_cell_47/ReadVariableOp_2:value:0+lstm_cell_47/strided_slice_2/stack:output:0-lstm_cell_47/strided_slice_2/stack_1:output:0-lstm_cell_47/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_6MatMulzeros:output:0%lstm_cell_47/strided_slice_2:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_4AddV2lstm_cell_47/BiasAdd_2:output:0lstm_cell_47/MatMul_6:product:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/ReluRelulstm_cell_47/add_4:z:0*
T0*'
_output_shapes
:         @М
lstm_cell_47/mul_3Mullstm_cell_47/clip_by_value:z:0lstm_cell_47/Relu:activations:0*
T0*'
_output_shapes
:         @}
lstm_cell_47/add_5AddV2lstm_cell_47/mul_2:z:0lstm_cell_47/mul_3:z:0*
T0*'
_output_shapes
:         @Г
lstm_cell_47/ReadVariableOp_3ReadVariableOp$lstm_cell_47_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm_cell_47/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    └   u
$lstm_cell_47/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_47/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╢
lstm_cell_47/strided_slice_3StridedSlice%lstm_cell_47/ReadVariableOp_3:value:0+lstm_cell_47/strided_slice_3/stack:output:0-lstm_cell_47/strided_slice_3/stack_1:output:0-lstm_cell_47/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskИ
lstm_cell_47/MatMul_7MatMulzeros:output:0%lstm_cell_47/strided_slice_3:output:0*
T0*'
_output_shapes
:         @П
lstm_cell_47/add_6AddV2lstm_cell_47/BiasAdd_3:output:0lstm_cell_47/MatMul_7:product:0*
T0*'
_output_shapes
:         @Y
lstm_cell_47/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Y
lstm_cell_47/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?В
lstm_cell_47/Mul_4Mullstm_cell_47/add_6:z:0lstm_cell_47/Const_4:output:0*
T0*'
_output_shapes
:         @Д
lstm_cell_47/Add_7AddV2lstm_cell_47/Mul_4:z:0lstm_cell_47/Const_5:output:0*
T0*'
_output_shapes
:         @k
&lstm_cell_47/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
$lstm_cell_47/clip_by_value_2/MinimumMinimumlstm_cell_47/Add_7:z:0/lstm_cell_47/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:         @c
lstm_cell_47/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    м
lstm_cell_47/clip_by_value_2Maximum(lstm_cell_47/clip_by_value_2/Minimum:z:0'lstm_cell_47/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:         @e
lstm_cell_47/Relu_1Relulstm_cell_47/add_5:z:0*
T0*'
_output_shapes
:         @Р
lstm_cell_47/mul_5Mul lstm_cell_47/clip_by_value_2:z:0!lstm_cell_47/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_47_split_readvariableop_resource,lstm_cell_47_split_1_readvariableop_resource$lstm_cell_47_readvariableop_resource*
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
while_body_378935*
condR
while_cond_378934*K
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
:
         @*
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
:         
@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @Ц
NoOpNoOp^lstm_cell_47/ReadVariableOp^lstm_cell_47/ReadVariableOp_1^lstm_cell_47/ReadVariableOp_2^lstm_cell_47/ReadVariableOp_3"^lstm_cell_47/split/ReadVariableOp$^lstm_cell_47/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         
А: : : 2>
lstm_cell_47/ReadVariableOp_1lstm_cell_47/ReadVariableOp_12>
lstm_cell_47/ReadVariableOp_2lstm_cell_47/ReadVariableOp_22>
lstm_cell_47/ReadVariableOp_3lstm_cell_47/ReadVariableOp_32:
lstm_cell_47/ReadVariableOplstm_cell_47/ReadVariableOp2F
!lstm_cell_47/split/ReadVariableOp!lstm_cell_47/split/ReadVariableOp2J
#lstm_cell_47/split_1/ReadVariableOp#lstm_cell_47/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         
А
 
_user_specified_nameinputs
Ы	
├
while_cond_381520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_381520___redundant_placeholder04
0while_while_cond_381520___redundant_placeholder14
0while_while_cond_381520___redundant_placeholder24
0while_while_cond_381520___redundant_placeholder3
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
╥7
Ж
C__inference_lstm_47_layer_call_and_return_conditional_losses_377971

inputs'
lstm_cell_47_377890:
АА"
lstm_cell_47_377892:	А&
lstm_cell_47_377894:	@А
identityИв$lstm_cell_47/StatefulPartitionedCallвwhileI
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
$lstm_cell_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_47_377890lstm_cell_47_377892lstm_cell_47_377894*
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_377889n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_47_377890lstm_cell_47_377892lstm_cell_47_377894*
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
while_body_377903*
condR
while_cond_377902*K
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
NoOpNoOp%^lstm_cell_47/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  А: : : 2L
$lstm_cell_47/StatefulPartitionedCall$lstm_cell_47/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
м
╕
(__inference_lstm_46_layer_call_fn_380615
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_377756}
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
inputs_0"є
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
lstm_46_input:
serving_default_lstm_46_input:0         
<
dense_230
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
!__inference__wrapped_model_377303╞
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
lstm_46_input         
z0trace_0
╧
1trace_0
2trace_1
3trace_2
4trace_32ф
I__inference_sequential_23_layer_call_and_return_conditional_losses_380079
I__inference_sequential_23_layer_call_and_return_conditional_losses_380593
I__inference_sequential_23_layer_call_and_return_conditional_losses_379471
I__inference_sequential_23_layer_call_and_return_conditional_losses_379494╡
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
.__inference_sequential_23_layer_call_fn_378796
.__inference_sequential_23_layer_call_fn_379544
.__inference_sequential_23_layer_call_fn_379565
.__inference_sequential_23_layer_call_fn_379448╡
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_380893
C__inference_lstm_46_layer_call_and_return_conditional_losses_381149
C__inference_lstm_46_layer_call_and_return_conditional_losses_381405
C__inference_lstm_46_layer_call_and_return_conditional_losses_381661╩
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
(__inference_lstm_46_layer_call_fn_380604
(__inference_lstm_46_layer_call_fn_380615
(__inference_lstm_46_layer_call_fn_380626
(__inference_lstm_46_layer_call_fn_380637╩
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_381961
C__inference_lstm_47_layer_call_and_return_conditional_losses_382217
C__inference_lstm_47_layer_call_and_return_conditional_losses_382473
C__inference_lstm_47_layer_call_and_return_conditional_losses_382729╩
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
(__inference_lstm_47_layer_call_fn_381672
(__inference_lstm_47_layer_call_fn_381683
(__inference_lstm_47_layer_call_fn_381694
(__inference_lstm_47_layer_call_fn_381705╩
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
D__inference_dense_23_layer_call_and_return_conditional_losses_382748Ш
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
)__inference_dense_23_layer_call_fn_382738Ш
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
!:@2dense_23/kernel
:2dense_23/bias
.:,	А2lstm_46/lstm_cell_46/kernel
9:7
АА2%lstm_46/lstm_cell_46/recurrent_kernel
(:&А2lstm_46/lstm_cell_46/bias
/:-
АА2lstm_47/lstm_cell_47/kernel
8:6	@А2%lstm_47/lstm_cell_47/recurrent_kernel
(:&А2lstm_47/lstm_cell_47/bias
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
!__inference__wrapped_model_377303lstm_46_input"╞
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
lstm_46_input         

РBН
I__inference_sequential_23_layer_call_and_return_conditional_losses_380079inputs"╡
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
I__inference_sequential_23_layer_call_and_return_conditional_losses_380593inputs"╡
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
I__inference_sequential_23_layer_call_and_return_conditional_losses_379471lstm_46_input"╡
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
I__inference_sequential_23_layer_call_and_return_conditional_losses_379494lstm_46_input"╡
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
.__inference_sequential_23_layer_call_fn_378796lstm_46_input"╡
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
.__inference_sequential_23_layer_call_fn_379544inputs"╡
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
.__inference_sequential_23_layer_call_fn_379565inputs"╡
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
.__inference_sequential_23_layer_call_fn_379448lstm_46_input"╡
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
$__inference_signature_wrapper_379523lstm_46_input"Ф
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_380893inputs_0"╩
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_381149inputs_0"╩
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_381405inputs"╩
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_381661inputs"╩
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
(__inference_lstm_46_layer_call_fn_380604inputs_0"╩
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
(__inference_lstm_46_layer_call_fn_380615inputs_0"╩
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
(__inference_lstm_46_layer_call_fn_380626inputs"╩
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
(__inference_lstm_46_layer_call_fn_380637inputs"╩
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382871
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382960│
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
-__inference_lstm_cell_46_layer_call_fn_382765
-__inference_lstm_cell_46_layer_call_fn_382782│
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_381961inputs_0"╩
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_382217inputs_0"╩
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_382473inputs"╩
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_382729inputs"╩
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
(__inference_lstm_47_layer_call_fn_381672inputs_0"╩
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
(__inference_lstm_47_layer_call_fn_381683inputs_0"╩
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
(__inference_lstm_47_layer_call_fn_381694inputs"╩
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
(__inference_lstm_47_layer_call_fn_381705inputs"╩
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383083
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383172│
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
-__inference_lstm_cell_47_layer_call_fn_382977
-__inference_lstm_cell_47_layer_call_fn_382994│
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
D__inference_dense_23_layer_call_and_return_conditional_losses_382748inputs"Ш
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
)__inference_dense_23_layer_call_fn_382738inputs"Ш
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382871inputsstates_0states_1"│
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382960inputsstates_0states_1"│
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
-__inference_lstm_cell_46_layer_call_fn_382765inputsstates_0states_1"│
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
-__inference_lstm_cell_46_layer_call_fn_382782inputsstates_0states_1"│
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383083inputsstates_0states_1"│
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383172inputsstates_0states_1"│
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
-__inference_lstm_cell_47_layer_call_fn_382977inputsstates_0states_1"│
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
-__inference_lstm_cell_47_layer_call_fn_382994inputsstates_0states_1"│
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
&:$@2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
3:1	А2"Adam/lstm_46/lstm_cell_46/kernel/m
>:<
АА2,Adam/lstm_46/lstm_cell_46/recurrent_kernel/m
-:+А2 Adam/lstm_46/lstm_cell_46/bias/m
4:2
АА2"Adam/lstm_47/lstm_cell_47/kernel/m
=:;	@А2,Adam/lstm_47/lstm_cell_47/recurrent_kernel/m
-:+А2 Adam/lstm_47/lstm_cell_47/bias/m
&:$@2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v
3:1	А2"Adam/lstm_46/lstm_cell_46/kernel/v
>:<
АА2,Adam/lstm_46/lstm_cell_46/recurrent_kernel/v
-:+А2 Adam/lstm_46/lstm_cell_46/bias/v
4:2
АА2"Adam/lstm_47/lstm_cell_47/kernel/v
=:;	@А2,Adam/lstm_47/lstm_cell_47/recurrent_kernel/v
-:+А2 Adam/lstm_47/lstm_cell_47/bias/vа
!__inference__wrapped_model_377303{%'&(*)#$:в7
0в-
+К(
lstm_46_input         

к "3к0
.
dense_23"К
dense_23         л
D__inference_dense_23_layer_call_and_return_conditional_losses_382748c#$/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         
Ъ Е
)__inference_dense_23_layer_call_fn_382738X#$/в,
%в"
 К
inputs         @
к "!К
unknown         ┌
C__inference_lstm_46_layer_call_and_return_conditional_losses_380893Т%'&OвL
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_381149Т%'&OвL
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
C__inference_lstm_46_layer_call_and_return_conditional_losses_381405y%'&?в<
5в2
$К!
inputs         


 
p 

 
к "1в.
'К$
tensor_0         
А
Ъ └
C__inference_lstm_46_layer_call_and_return_conditional_losses_381661y%'&?в<
5в2
$К!
inputs         


 
p

 
к "1в.
'К$
tensor_0         
А
Ъ ┤
(__inference_lstm_46_layer_call_fn_380604З%'&OвL
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
(__inference_lstm_46_layer_call_fn_380615З%'&OвL
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
(__inference_lstm_46_layer_call_fn_380626n%'&?в<
5в2
$К!
inputs         


 
p 

 
к "&К#
unknown         
АЪ
(__inference_lstm_46_layer_call_fn_380637n%'&?в<
5в2
$К!
inputs         


 
p

 
к "&К#
unknown         
А═
C__inference_lstm_47_layer_call_and_return_conditional_losses_381961Е(*)PвM
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_382217Е(*)PвM
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
C__inference_lstm_47_layer_call_and_return_conditional_losses_382473u(*)@в=
6в3
%К"
inputs         
А

 
p 

 
к ",в)
"К
tensor_0         @
Ъ ╝
C__inference_lstm_47_layer_call_and_return_conditional_losses_382729u(*)@в=
6в3
%К"
inputs         
А

 
p

 
к ",в)
"К
tensor_0         @
Ъ ж
(__inference_lstm_47_layer_call_fn_381672z(*)PвM
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
(__inference_lstm_47_layer_call_fn_381683z(*)PвM
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
(__inference_lstm_47_layer_call_fn_381694j(*)@в=
6в3
%К"
inputs         
А

 
p 

 
к "!К
unknown         @Ц
(__inference_lstm_47_layer_call_fn_381705j(*)@в=
6в3
%К"
inputs         
А

 
p

 
к "!К
unknown         @ч
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382871Ъ%'&Вв
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
H__inference_lstm_cell_46_layer_call_and_return_conditional_losses_382960Ъ%'&Вв
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
-__inference_lstm_cell_46_layer_call_fn_382765З%'&Вв
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
-__inference_lstm_cell_46_layer_call_fn_382782З%'&Вв
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383083Х(*)Бв~
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
H__inference_lstm_cell_47_layer_call_and_return_conditional_losses_383172Х(*)Бв~
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
-__inference_lstm_cell_47_layer_call_fn_382977Г(*)Бв~
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
-__inference_lstm_cell_47_layer_call_fn_382994Г(*)Бв~
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
I__inference_sequential_23_layer_call_and_return_conditional_losses_379471|%'&(*)#$Bв?
8в5
+К(
lstm_46_input         

p 

 
к ",в)
"К
tensor_0         
Ъ ╔
I__inference_sequential_23_layer_call_and_return_conditional_losses_379494|%'&(*)#$Bв?
8в5
+К(
lstm_46_input         

p

 
к ",в)
"К
tensor_0         
Ъ ┬
I__inference_sequential_23_layer_call_and_return_conditional_losses_380079u%'&(*)#$;в8
1в.
$К!
inputs         

p 

 
к ",в)
"К
tensor_0         
Ъ ┬
I__inference_sequential_23_layer_call_and_return_conditional_losses_380593u%'&(*)#$;в8
1в.
$К!
inputs         

p

 
к ",в)
"К
tensor_0         
Ъ г
.__inference_sequential_23_layer_call_fn_378796q%'&(*)#$Bв?
8в5
+К(
lstm_46_input         

p 

 
к "!К
unknown         г
.__inference_sequential_23_layer_call_fn_379448q%'&(*)#$Bв?
8в5
+К(
lstm_46_input         

p

 
к "!К
unknown         Ь
.__inference_sequential_23_layer_call_fn_379544j%'&(*)#$;в8
1в.
$К!
inputs         

p 

 
к "!К
unknown         Ь
.__inference_sequential_23_layer_call_fn_379565j%'&(*)#$;в8
1в.
$К!
inputs         

p

 
к "!К
unknown         ╡
$__inference_signature_wrapper_379523М%'&(*)#$KвH
в 
Aк>
<
lstm_46_input+К(
lstm_46_input         
"3к0
.
dense_23"К
dense_23         