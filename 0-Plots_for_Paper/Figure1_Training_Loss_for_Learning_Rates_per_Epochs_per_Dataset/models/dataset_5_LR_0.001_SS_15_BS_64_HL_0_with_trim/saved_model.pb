ë¸
Ţ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

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
resource
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
output"out_typeíout_type"	
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
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
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint˙˙˙˙˙˙˙˙˙
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Š

 Adam/lstm_10/lstm_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_10/lstm_cell_10/bias/v

4Adam/lstm_10/lstm_cell_10/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_10/lstm_cell_10/bias/v*
_output_shapes	
:*
dtype0
ś
,Adam/lstm_10/lstm_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v
Ż
@Adam/lstm_10/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
Ą
"Adam/lstm_10/lstm_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_10/lstm_cell_10/kernel/v

6Adam/lstm_10/lstm_cell_10/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_10/lstm_cell_10/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0

Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_10/kernel/v

*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	*
dtype0

 Adam/lstm_10/lstm_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_10/lstm_cell_10/bias/m

4Adam/lstm_10/lstm_cell_10/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_10/lstm_cell_10/bias/m*
_output_shapes	
:*
dtype0
ś
,Adam/lstm_10/lstm_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m
Ż
@Adam/lstm_10/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
Ą
"Adam/lstm_10/lstm_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/lstm_10/lstm_cell_10/kernel/m

6Adam/lstm_10/lstm_cell_10/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_10/lstm_cell_10/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0

Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_10/kernel/m

*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	*
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

lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_10/lstm_cell_10/bias

-lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/bias*
_output_shapes	
:*
dtype0
¨
%lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_10/lstm_cell_10/recurrent_kernel
Ą
9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_10/lstm_cell_10/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_namelstm_10/lstm_cell_10/kernel

/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/kernel*
_output_shapes
:	*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	*
dtype0

serving_default_lstm_10_inputPlaceholder*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0* 
shape:˙˙˙˙˙˙˙˙˙
Ŕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_10_inputlstm_10/lstm_cell_10/kernellstm_10/lstm_cell_10/bias%lstm_10/lstm_cell_10/recurrent_kerneldense_10/kerneldense_10/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_91460

NoOpNoOp
ń+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ź+
value˘+B+ B+

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__
	_default_save_signature

	optimizer

signatures*
Ş
trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__
cell

state_spec*
Ś
trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__

kernel
bias*
'
0
1
2
3
4*
* 
'
0
1
2
3
4*
°
layer_regularization_losses
trainable_variables
 layer_metrics
!metrics

"layers
regularization_losses
#non_trainable_variables
	variables
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
$trace_0
%trace_1
&trace_2
'trace_3* 
6
(trace_0
)trace_1
*trace_2
+trace_3* 

,trace_0* 


-beta_1

.beta_2
	/decay
0learning_rate
1iterm]m^m_m`mavbvcvdvevf*

2serving_default* 

0
1
2*
* 

0
1
2*

3layer_regularization_losses
trainable_variables
4non_trainable_variables
5layer_metrics
6metrics

7layers
regularization_losses

8states
	variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
9trace_0
:trace_1
;trace_2
<trace_3* 
6
=trace_0
>trace_1
?trace_2
@trace_3* 
Ě
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
*E&call_and_return_all_conditional_losses
F__call__
G
state_size

kernel
recurrent_kernel
bias*
* 

0
1*
* 

0
1*

Hlayer_regularization_losses
trainable_variables
Ilayer_metrics
Jmetrics

Klayers
regularization_losses
Lnon_trainable_variables
	variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUElstm_10/lstm_cell_10/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE%lstm_10/lstm_cell_10/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUElstm_10/lstm_cell_10/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

O0*

0
1*
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
* 
* 

0*
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
0
1
2*
* 

0
1
2*

Player_regularization_losses
Atrainable_variables
Qlayer_metrics
Rmetrics

Slayers
Bregularization_losses
Tnon_trainable_variables
C	variables
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Utrace_0
Vtrace_1* 

Wtrace_0
Xtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
8
Y	variables
Z	keras_api
	[total
	\count*
* 
* 
* 
* 
* 
* 
* 
* 
* 

[0
\1*

Y	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/lstm_10/lstm_cell_10/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_10/lstm_cell_10/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/lstm_10/lstm_cell_10/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/lstm_10/lstm_cell_10/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_10/lstm_cell_10/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/lstm_10/lstm_cell_10/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
˛
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biaslstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense_10/kernel/mAdam/dense_10/bias/m"Adam/lstm_10/lstm_cell_10/kernel/m,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m Adam/lstm_10/lstm_cell_10/bias/mAdam/dense_10/kernel/vAdam/dense_10/bias/v"Adam/lstm_10/lstm_cell_10/kernel/v,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v Adam/lstm_10/lstm_cell_10/bias/vConst*#
Tin
2*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_93468
­
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biaslstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense_10/kernel/mAdam/dense_10/bias/m"Adam/lstm_10/lstm_cell_10/kernel/m,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m Adam/lstm_10/lstm_cell_10/bias/mAdam/dense_10/kernel/vAdam/dense_10/bias/v"Adam/lstm_10/lstm_cell_10/kernel/v,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v Adam/lstm_10/lstm_cell_10/bias/v*"
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_93544
Ţ
×
H__inference_sequential_10_layer_call_and_return_conditional_losses_91437
lstm_10_input 
lstm_10_91424:	
lstm_10_91426:	!
lstm_10_91428:
!
dense_10_91431:	
dense_10_91433:
identity˘ dense_10/StatefulPartitionedCall˘lstm_10/StatefulPartitionedCall
lstm_10/StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputlstm_10_91424lstm_10_91426lstm_10_91428*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_91335
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_10_91431dense_10_91433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_91036x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall:Z V
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namelstm_10_input
É
Đ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91043

inputs 
lstm_10_91019:	
lstm_10_91021:	!
lstm_10_91023:
!
dense_10_91037:	
dense_10_91039:
identity˘ dense_10/StatefulPartitionedCall˘lstm_10/StatefulPartitionedCallű
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_91019lstm_10_91021lstm_10_91023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_91018
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_10_91037dense_10_91039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_91036x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô
ö
,__inference_lstm_cell_10_layer_call_fn_93135

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90620p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ř
ľ
'__inference_lstm_10_layer_call_fn_92058

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_91335p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
ž
while_cond_91194
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_91194___redundant_placeholder03
/while_while_cond_91194___redundant_placeholder13
/while_while_cond_91194___redundant_placeholder23
/while_while_cond_91194___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
K
Š
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90418

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
ž
while_cond_90678
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_90678___redundant_placeholder03
/while_while_cond_90678___redundant_placeholder13
/while_while_cond_90678___redundant_placeholder23
/while_while_cond_90678___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
Ę	
ő
C__inference_dense_10_layer_call_and_return_conditional_losses_91036

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ţ
lstm_10_while_cond_91605,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1C
?lstm_10_while_lstm_10_while_cond_91605___redundant_placeholder0C
?lstm_10_while_lstm_10_while_cond_91605___redundant_placeholder1C
?lstm_10_while_lstm_10_while_cond_91605___redundant_placeholder2C
?lstm_10_while_lstm_10_while_cond_91605___redundant_placeholder3
lstm_10_while_identity

lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: [
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
_user_specified_name" lstm_10/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_10/while/loop_counter
ß
ç
B__inference_lstm_10_layer_call_and_return_conditional_losses_92826

inputs=
*lstm_cell_10_split_readvariableop_resource:	;
,lstm_cell_10_split_1_readvariableop_resource:	8
$lstm_cell_10_readvariableop_resource:

identity˘lstm_cell_10/ReadVariableOp˘lstm_cell_10/ReadVariableOp_1˘lstm_cell_10/ReadVariableOp_2˘lstm_cell_10/ReadVariableOp_3˘!lstm_cell_10/split/ReadVariableOp˘#lstm_cell_10/split_1/ReadVariableOp˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0É
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ż
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_4MatMulzeros:output:0#lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_10/MulMullstm_cell_10/add:z:0lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_1AddV2lstm_cell_10/Mul:z:0lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
"lstm_cell_10/clip_by_value/MinimumMinimumlstm_cell_10/Add_1:z:0-lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    §
lstm_cell_10/clip_by_valueMaximum&lstm_cell_10/clip_by_value/Minimum:z:0%lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_5MatMulzeros:output:0%lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_1Mullstm_cell_10/add_2:z:0lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_3AddV2lstm_cell_10/Mul_1:z:0lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_cell_10/Add_3:z:0/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_1Maximum(lstm_cell_10/clip_by_value_1/Minimum:z:0'lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_2Mul lstm_cell_10/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_6MatMulzeros:output:0%lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_10/ReluRelulstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_3Mullstm_cell_10/clip_by_value:z:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_10/add_5AddV2lstm_cell_10/mul_2:z:0lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_7MatMulzeros:output:0%lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_6AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_4Mullstm_cell_10/add_6:z:0lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_7AddV2lstm_cell_10/Mul_4:z:0lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_cell_10/Add_7:z:0/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_2Maximum(lstm_cell_10/clip_by_value_2/Minimum:z:0'lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_10/Relu_1Relulstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_5Mul lstm_cell_10/clip_by_value_2:z:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_92686*
condR
while_cond_92685*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : 2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô
ö
,__inference_lstm_cell_10_layer_call_fn_93118

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90418p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ß
ç
B__inference_lstm_10_layer_call_and_return_conditional_losses_91018

inputs=
*lstm_cell_10_split_readvariableop_resource:	;
,lstm_cell_10_split_1_readvariableop_resource:	8
$lstm_cell_10_readvariableop_resource:

identity˘lstm_cell_10/ReadVariableOp˘lstm_cell_10/ReadVariableOp_1˘lstm_cell_10/ReadVariableOp_2˘lstm_cell_10/ReadVariableOp_3˘!lstm_cell_10/split/ReadVariableOp˘#lstm_cell_10/split_1/ReadVariableOp˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0É
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ż
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_4MatMulzeros:output:0#lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_10/MulMullstm_cell_10/add:z:0lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_1AddV2lstm_cell_10/Mul:z:0lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
"lstm_cell_10/clip_by_value/MinimumMinimumlstm_cell_10/Add_1:z:0-lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    §
lstm_cell_10/clip_by_valueMaximum&lstm_cell_10/clip_by_value/Minimum:z:0%lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_5MatMulzeros:output:0%lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_1Mullstm_cell_10/add_2:z:0lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_3AddV2lstm_cell_10/Mul_1:z:0lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_cell_10/Add_3:z:0/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_1Maximum(lstm_cell_10/clip_by_value_1/Minimum:z:0'lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_2Mul lstm_cell_10/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_6MatMulzeros:output:0%lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_10/ReluRelulstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_3Mullstm_cell_10/clip_by_value:z:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_10/add_5AddV2lstm_cell_10/mul_2:z:0lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_7MatMulzeros:output:0%lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_6AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_4Mullstm_cell_10/add_6:z:0lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_7AddV2lstm_cell_10/Mul_4:z:0lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_cell_10/Add_7:z:0/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_2Maximum(lstm_cell_10/clip_by_value_2/Minimum:z:0'lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_10/Relu_1Relulstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_5Mul lstm_cell_10/clip_by_value_2:z:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_90878*
condR
while_cond_90877*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : 2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ˇ
'__inference_lstm_10_layer_call_fn_92025
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_90500p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0
Ę	
ő
C__inference_dense_10_layer_call_and_return_conditional_losses_93101

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Š
ß
__inference__traced_save_93468
file_prefix9
&read_disablecopyonread_dense_10_kernel:	4
&read_1_disablecopyonread_dense_10_bias:G
4read_2_disablecopyonread_lstm_10_lstm_cell_10_kernel:	R
>read_3_disablecopyonread_lstm_10_lstm_cell_10_recurrent_kernel:
A
2read_4_disablecopyonread_lstm_10_lstm_cell_10_bias:	)
read_5_disablecopyonread_beta_1: )
read_6_disablecopyonread_beta_2: (
read_7_disablecopyonread_decay: 0
&read_8_disablecopyonread_learning_rate: ,
"read_9_disablecopyonread_adam_iter:	 )
read_10_disablecopyonread_total: )
read_11_disablecopyonread_count: C
0read_12_disablecopyonread_adam_dense_10_kernel_m:	<
.read_13_disablecopyonread_adam_dense_10_bias_m:O
<read_14_disablecopyonread_adam_lstm_10_lstm_cell_10_kernel_m:	Z
Fread_15_disablecopyonread_adam_lstm_10_lstm_cell_10_recurrent_kernel_m:
I
:read_16_disablecopyonread_adam_lstm_10_lstm_cell_10_bias_m:	C
0read_17_disablecopyonread_adam_dense_10_kernel_v:	<
.read_18_disablecopyonread_adam_dense_10_bias_v:O
<read_19_disablecopyonread_adam_lstm_10_lstm_cell_10_kernel_v:	Z
Fread_20_disablecopyonread_adam_lstm_10_lstm_cell_10_recurrent_kernel_v:
I
:read_21_disablecopyonread_adam_lstm_10_lstm_cell_10_bias_v:	
savev2_const
identity_45˘MergeV2Checkpoints˘Read/DisableCopyOnRead˘Read/ReadVariableOp˘Read_1/DisableCopyOnRead˘Read_1/ReadVariableOp˘Read_10/DisableCopyOnRead˘Read_10/ReadVariableOp˘Read_11/DisableCopyOnRead˘Read_11/ReadVariableOp˘Read_12/DisableCopyOnRead˘Read_12/ReadVariableOp˘Read_13/DisableCopyOnRead˘Read_13/ReadVariableOp˘Read_14/DisableCopyOnRead˘Read_14/ReadVariableOp˘Read_15/DisableCopyOnRead˘Read_15/ReadVariableOp˘Read_16/DisableCopyOnRead˘Read_16/ReadVariableOp˘Read_17/DisableCopyOnRead˘Read_17/ReadVariableOp˘Read_18/DisableCopyOnRead˘Read_18/ReadVariableOp˘Read_19/DisableCopyOnRead˘Read_19/ReadVariableOp˘Read_2/DisableCopyOnRead˘Read_2/ReadVariableOp˘Read_20/DisableCopyOnRead˘Read_20/ReadVariableOp˘Read_21/DisableCopyOnRead˘Read_21/ReadVariableOp˘Read_3/DisableCopyOnRead˘Read_3/ReadVariableOp˘Read_4/DisableCopyOnRead˘Read_4/ReadVariableOp˘Read_5/DisableCopyOnRead˘Read_5/ReadVariableOp˘Read_6/DisableCopyOnRead˘Read_6/ReadVariableOp˘Read_7/DisableCopyOnRead˘Read_7/ReadVariableOp˘Read_8/DisableCopyOnRead˘Read_8/ReadVariableOp˘Read_9/DisableCopyOnRead˘Read_9/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 Ł
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_10_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 ˘
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_10_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_lstm_10_lstm_cell_10_kernel"/device:CPU:0*
_output_shapes
 ľ
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_lstm_10_lstm_cell_10_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_3/DisableCopyOnReadDisableCopyOnRead>read_3_disablecopyonread_lstm_10_lstm_cell_10_recurrent_kernel"/device:CPU:0*
_output_shapes
 Ŕ
Read_3/ReadVariableOpReadVariableOp>read_3_disablecopyonread_lstm_10_lstm_cell_10_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
e

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_4/DisableCopyOnReadDisableCopyOnRead2read_4_disablecopyonread_lstm_10_lstm_cell_10_bias"/device:CPU:0*
_output_shapes
 Ż
Read_4/ReadVariableOpReadVariableOp2read_4_disablecopyonread_lstm_10_lstm_cell_10_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:s
Read_5/DisableCopyOnReadDisableCopyOnReadread_5_disablecopyonread_beta_1"/device:CPU:0*
_output_shapes
 
Read_5/ReadVariableOpReadVariableOpread_5_disablecopyonread_beta_1^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: s
Read_6/DisableCopyOnReadDisableCopyOnReadread_6_disablecopyonread_beta_2"/device:CPU:0*
_output_shapes
 
Read_6/ReadVariableOpReadVariableOpread_6_disablecopyonread_beta_2^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_7/DisableCopyOnReadDisableCopyOnReadread_7_disablecopyonread_decay"/device:CPU:0*
_output_shapes
 
Read_7/ReadVariableOpReadVariableOpread_7_disablecopyonread_decay^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_learning_rate^Read_8/DisableCopyOnRead"/device:CPU:0*
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
: v
Read_9/DisableCopyOnReadDisableCopyOnRead"read_9_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp"read_9_disablecopyonread_adam_iter^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0	*
_output_shapes
: t
Read_10/DisableCopyOnReadDisableCopyOnReadread_10_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOpread_10_disablecopyonread_total^Read_10/DisableCopyOnRead"/device:CPU:0*
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
: t
Read_11/DisableCopyOnReadDisableCopyOnReadread_11_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_11/ReadVariableOpReadVariableOpread_11_disablecopyonread_count^Read_11/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_adam_dense_10_kernel_m"/device:CPU:0*
_output_shapes
 ł
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_adam_dense_10_kernel_m^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_13/DisableCopyOnReadDisableCopyOnRead.read_13_disablecopyonread_adam_dense_10_bias_m"/device:CPU:0*
_output_shapes
 Ź
Read_13/ReadVariableOpReadVariableOp.read_13_disablecopyonread_adam_dense_10_bias_m^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_14/DisableCopyOnReadDisableCopyOnRead<read_14_disablecopyonread_adam_lstm_10_lstm_cell_10_kernel_m"/device:CPU:0*
_output_shapes
 ż
Read_14/ReadVariableOpReadVariableOp<read_14_disablecopyonread_adam_lstm_10_lstm_cell_10_kernel_m^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_15/DisableCopyOnReadDisableCopyOnReadFread_15_disablecopyonread_adam_lstm_10_lstm_cell_10_recurrent_kernel_m"/device:CPU:0*
_output_shapes
 Ę
Read_15/ReadVariableOpReadVariableOpFread_15_disablecopyonread_adam_lstm_10_lstm_cell_10_recurrent_kernel_m^Read_15/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_16/DisableCopyOnReadDisableCopyOnRead:read_16_disablecopyonread_adam_lstm_10_lstm_cell_10_bias_m"/device:CPU:0*
_output_shapes
 š
Read_16/ReadVariableOpReadVariableOp:read_16_disablecopyonread_adam_lstm_10_lstm_cell_10_bias_m^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_dense_10_kernel_v"/device:CPU:0*
_output_shapes
 ł
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_dense_10_kernel_v^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_dense_10_bias_v"/device:CPU:0*
_output_shapes
 Ź
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_dense_10_bias_v^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_19/DisableCopyOnReadDisableCopyOnRead<read_19_disablecopyonread_adam_lstm_10_lstm_cell_10_kernel_v"/device:CPU:0*
_output_shapes
 ż
Read_19/ReadVariableOpReadVariableOp<read_19_disablecopyonread_adam_lstm_10_lstm_cell_10_kernel_v^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_20/DisableCopyOnReadDisableCopyOnReadFread_20_disablecopyonread_adam_lstm_10_lstm_cell_10_recurrent_kernel_v"/device:CPU:0*
_output_shapes
 Ę
Read_20/ReadVariableOpReadVariableOpFread_20_disablecopyonread_adam_lstm_10_lstm_cell_10_recurrent_kernel_v^Read_20/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_21/DisableCopyOnReadDisableCopyOnRead:read_21_disablecopyonread_adam_lstm_10_lstm_cell_10_bias_v"/device:CPU:0*
_output_shapes
 š
Read_21/ReadVariableOpReadVariableOp:read_21_disablecopyonread_adam_lstm_10_lstm_cell_10_bias_v^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:˙
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¨
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B Í
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_44Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_45IdentityIdentity_44:output:0^NoOp*
T0*
_output_shapes
: Ń	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_45Identity_45:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_21/ReadVariableOpRead_21/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
K
Š
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90620

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
 ~
Ľ	
while_body_92942
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	C
4while_lstm_cell_10_split_1_readvariableop_resource_0:	@
,while_lstm_cell_10_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	A
2while_lstm_cell_10_split_1_readvariableop_resource:	>
*while_lstm_cell_10_readvariableop_resource:
˘!while/lstm_cell_10/ReadVariableOp˘#while/lstm_cell_10/ReadVariableOp_1˘#while/lstm_cell_10/ReadVariableOp_2˘#while/lstm_cell_10/ReadVariableOp_3˘'while/lstm_cell_10/split/ReadVariableOp˘)while/lstm_cell_10/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ű
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitŤ
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ń
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/MulMulwhile/lstm_cell_10/add:z:0!while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_1AddV2while/lstm_cell_10/Mul:z:0#while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?š
(while/lstm_cell_10/clip_by_value/MinimumMinimumwhile/lstm_cell_10/Add_1:z:03while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    š
 while/lstm_cell_10/clip_by_valueMaximum,while/lstm_cell_10/clip_by_value/Minimum:z:0+while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_1Mulwhile/lstm_cell_10/add_2:z:0#while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_3AddV2while/lstm_cell_10/Mul_1:z:0#while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_1/MinimumMinimumwhile/lstm_cell_10/Add_3:z:05while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_1Maximum.while/lstm_cell_10/clip_by_value_1/Minimum:z:0-while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_2Mul&while/lstm_cell_10/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_3Mul$while/lstm_cell_10/clip_by_value:z:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/add_5AddV2while/lstm_cell_10/mul_2:z:0while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_6AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_4Mulwhile/lstm_cell_10/add_6:z:0#while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_7AddV2while/lstm_cell_10/Mul_4:z:0#while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_2/MinimumMinimumwhile/lstm_cell_10/Add_7:z:05while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_2Maximum.while/lstm_cell_10/clip_by_value_2/Minimum:z:0-while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_10/mul_5Mul&while/lstm_cell_10/clip_by_value_2:z:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_10/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/Identity_5Identitywhile/lstm_cell_10/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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

˝
lstm_10_while_body_91606,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0:	K
<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0:	H
4lstm_10_while_lstm_cell_10_readvariableop_resource_0:

lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorK
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:	I
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource:	F
2lstm_10_while_lstm_cell_10_readvariableop_resource:
˘)lstm_10/while/lstm_cell_10/ReadVariableOp˘+lstm_10/while/lstm_cell_10/ReadVariableOp_1˘+lstm_10/while/lstm_cell_10/ReadVariableOp_2˘+lstm_10/while/lstm_cell_10/ReadVariableOp_3˘/lstm_10/while/lstm_cell_10/split/ReadVariableOp˘1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Î
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0l
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ť
/lstm_10/while/lstm_cell_10/split/ReadVariableOpReadVariableOp:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0ó
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:07lstm_10/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitĂ
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#lstm_10/while/lstm_cell_10/MatMul_1MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#lstm_10/while/lstm_cell_10/MatMul_2MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#lstm_10/while/lstm_cell_10/MatMul_3MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
,lstm_10/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ť
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0é
"lstm_10/while/lstm_cell_10/split_1Split5lstm_10/while/lstm_cell_10/split_1/split_dim:output:09lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitş
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd+lstm_10/while/lstm_cell_10/MatMul:product:0+lstm_10/while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
$lstm_10/while/lstm_cell_10/BiasAdd_1BiasAdd-lstm_10/while/lstm_cell_10/MatMul_1:product:0+lstm_10/while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
$lstm_10/while/lstm_cell_10/BiasAdd_2BiasAdd-lstm_10/while/lstm_cell_10/MatMul_2:product:0+lstm_10/while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
$lstm_10/while/lstm_cell_10/BiasAdd_3BiasAdd-lstm_10/while/lstm_cell_10/MatMul_3:product:0+lstm_10/while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
)lstm_10/while/lstm_cell_10/ReadVariableOpReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_10/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0lstm_10/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_10/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_10/while/lstm_cell_10/strided_sliceStridedSlice1lstm_10/while/lstm_cell_10/ReadVariableOp:value:07lstm_10/while/lstm_cell_10/strided_slice/stack:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_1:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask°
#lstm_10/while/lstm_cell_10/MatMul_4MatMullstm_10_while_placeholder_21lstm_10/while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ś
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/BiasAdd:output:0-lstm_10/while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
 lstm_10/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>g
"lstm_10/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?§
lstm_10/while/lstm_cell_10/MulMul"lstm_10/while/lstm_cell_10/add:z:0)lstm_10/while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
 lstm_10/while/lstm_cell_10/Add_1AddV2"lstm_10/while/lstm_cell_10/Mul:z:0+lstm_10/while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
2lstm_10/while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ń
0lstm_10/while/lstm_cell_10/clip_by_value/MinimumMinimum$lstm_10/while/lstm_cell_10/Add_1:z:0;lstm_10/while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*lstm_10/while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ń
(lstm_10/while/lstm_cell_10/clip_by_valueMaximum4lstm_10/while/lstm_cell_10/clip_by_value/Minimum:z:03lstm_10/while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+lstm_10/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0lstm_10/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ţ
*lstm_10/while/lstm_cell_10/strided_slice_1StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_1:value:09lstm_10/while/lstm_cell_10/strided_slice_1/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask˛
#lstm_10/while/lstm_cell_10/MatMul_5MatMullstm_10_while_placeholder_23lstm_10/while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
 lstm_10/while/lstm_cell_10/add_2AddV2-lstm_10/while/lstm_cell_10/BiasAdd_1:output:0-lstm_10/while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_10/while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>g
"lstm_10/while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?­
 lstm_10/while/lstm_cell_10/Mul_1Mul$lstm_10/while/lstm_cell_10/add_2:z:0+lstm_10/while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
 lstm_10/while/lstm_cell_10/Add_3AddV2$lstm_10/while/lstm_cell_10/Mul_1:z:0+lstm_10/while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
4lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ő
2lstm_10/while/lstm_cell_10/clip_by_value_1/MinimumMinimum$lstm_10/while/lstm_cell_10/Add_3:z:0=lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,lstm_10/while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ×
*lstm_10/while/lstm_cell_10/clip_by_value_1Maximum6lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum:z:05lstm_10/while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙§
 lstm_10/while/lstm_cell_10/mul_2Mul.lstm_10/while/lstm_cell_10/clip_by_value_1:z:0lstm_10_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+lstm_10/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0lstm_10/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ţ
*lstm_10/while/lstm_cell_10/strided_slice_2StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_2:value:09lstm_10/while/lstm_cell_10/strided_slice_2/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask˛
#lstm_10/while/lstm_cell_10/MatMul_6MatMullstm_10_while_placeholder_23lstm_10/while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
 lstm_10/while/lstm_cell_10/add_4AddV2-lstm_10/while/lstm_cell_10/BiasAdd_2:output:0-lstm_10/while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/while/lstm_cell_10/ReluRelu$lstm_10/while/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
 lstm_10/while/lstm_cell_10/mul_3Mul,lstm_10/while/lstm_cell_10/clip_by_value:z:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
 lstm_10/while/lstm_cell_10/add_5AddV2$lstm_10/while/lstm_cell_10/mul_2:z:0$lstm_10/while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+lstm_10/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0lstm_10/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ţ
*lstm_10/while/lstm_cell_10/strided_slice_3StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_3:value:09lstm_10/while/lstm_cell_10/strided_slice_3/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask˛
#lstm_10/while/lstm_cell_10/MatMul_7MatMullstm_10_while_placeholder_23lstm_10/while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
 lstm_10/while/lstm_cell_10/add_6AddV2-lstm_10/while/lstm_cell_10/BiasAdd_3:output:0-lstm_10/while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_10/while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>g
"lstm_10/while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?­
 lstm_10/while/lstm_cell_10/Mul_4Mul$lstm_10/while/lstm_cell_10/add_6:z:0+lstm_10/while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
 lstm_10/while/lstm_cell_10/Add_7AddV2$lstm_10/while/lstm_cell_10/Mul_4:z:0+lstm_10/while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
4lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ő
2lstm_10/while/lstm_cell_10/clip_by_value_2/MinimumMinimum$lstm_10/while/lstm_cell_10/Add_7:z:0=lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,lstm_10/while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ×
*lstm_10/while/lstm_cell_10/clip_by_value_2Maximum6lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum:z:05lstm_10/while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
 lstm_10/while/lstm_cell_10/mul_5Mul.lstm_10/while/lstm_cell_10/clip_by_value_2:z:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ĺ
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1lstm_10_while_placeholder$lstm_10/while/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇU
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: q
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_5:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_5:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
lstm_10/while/NoOpNoOp*^lstm_10/while/lstm_cell_10/ReadVariableOp,^lstm_10/while/lstm_cell_10/ReadVariableOp_1,^lstm_10/while/lstm_cell_10/ReadVariableOp_2,^lstm_10/while/lstm_cell_10/ReadVariableOp_30^lstm_10/while/lstm_cell_10/split/ReadVariableOp2^lstm_10/while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"9
lstm_10_while_identitylstm_10/while/Identity:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"j
2lstm_10_while_lstm_cell_10_readvariableop_resource4lstm_10_while_lstm_cell_10_readvariableop_resource_0"z
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0"v
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0"Č
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_1+lstm_10/while/lstm_cell_10/ReadVariableOp_12Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_2+lstm_10/while/lstm_cell_10/ReadVariableOp_22Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_3+lstm_10/while/lstm_cell_10/ReadVariableOp_32V
)lstm_10/while/lstm_cell_10/ReadVariableOp)lstm_10/while/lstm_cell_10/ReadVariableOp2b
/lstm_10/while/lstm_cell_10/split/ReadVariableOp/lstm_10/while/lstm_cell_10/split/ReadVariableOp2f
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
_user_specified_name" lstm_10/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_10/while/loop_counter
ß
ç
B__inference_lstm_10_layer_call_and_return_conditional_losses_93082

inputs=
*lstm_cell_10_split_readvariableop_resource:	;
,lstm_cell_10_split_1_readvariableop_resource:	8
$lstm_cell_10_readvariableop_resource:

identity˘lstm_cell_10/ReadVariableOp˘lstm_cell_10/ReadVariableOp_1˘lstm_cell_10/ReadVariableOp_2˘lstm_cell_10/ReadVariableOp_3˘!lstm_cell_10/split/ReadVariableOp˘#lstm_cell_10/split_1/ReadVariableOp˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0É
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ż
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_4MatMulzeros:output:0#lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_10/MulMullstm_cell_10/add:z:0lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_1AddV2lstm_cell_10/Mul:z:0lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
"lstm_cell_10/clip_by_value/MinimumMinimumlstm_cell_10/Add_1:z:0-lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    §
lstm_cell_10/clip_by_valueMaximum&lstm_cell_10/clip_by_value/Minimum:z:0%lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_5MatMulzeros:output:0%lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_1Mullstm_cell_10/add_2:z:0lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_3AddV2lstm_cell_10/Mul_1:z:0lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_cell_10/Add_3:z:0/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_1Maximum(lstm_cell_10/clip_by_value_1/Minimum:z:0'lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_2Mul lstm_cell_10/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_6MatMulzeros:output:0%lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_10/ReluRelulstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_3Mullstm_cell_10/clip_by_value:z:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_10/add_5AddV2lstm_cell_10/mul_2:z:0lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_7MatMulzeros:output:0%lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_6AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_4Mullstm_cell_10/add_6:z:0lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_7AddV2lstm_cell_10/Mul_4:z:0lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_cell_10/Add_7:z:0/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_2Maximum(lstm_cell_10/clip_by_value_2/Minimum:z:0'lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_10/Relu_1Relulstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_5Mul lstm_cell_10/clip_by_value_2:z:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_92942*
condR
while_cond_92941*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : 2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
K
Ť
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93313

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ť
ö
&sequential_10_lstm_10_while_cond_90147H
Dsequential_10_lstm_10_while_sequential_10_lstm_10_while_loop_counterN
Jsequential_10_lstm_10_while_sequential_10_lstm_10_while_maximum_iterations+
'sequential_10_lstm_10_while_placeholder-
)sequential_10_lstm_10_while_placeholder_1-
)sequential_10_lstm_10_while_placeholder_2-
)sequential_10_lstm_10_while_placeholder_3J
Fsequential_10_lstm_10_while_less_sequential_10_lstm_10_strided_slice_1_
[sequential_10_lstm_10_while_sequential_10_lstm_10_while_cond_90147___redundant_placeholder0_
[sequential_10_lstm_10_while_sequential_10_lstm_10_while_cond_90147___redundant_placeholder1_
[sequential_10_lstm_10_while_sequential_10_lstm_10_while_cond_90147___redundant_placeholder2_
[sequential_10_lstm_10_while_sequential_10_lstm_10_while_cond_90147___redundant_placeholder3(
$sequential_10_lstm_10_while_identity
ş
 sequential_10/lstm_10/while/LessLess'sequential_10_lstm_10_while_placeholderFsequential_10_lstm_10_while_less_sequential_10_lstm_10_strided_slice_1*
T0*
_output_shapes
: w
$sequential_10/lstm_10/while/IdentityIdentity$sequential_10/lstm_10/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_10_lstm_10_while_identity-sequential_10/lstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
_user_specified_name0.sequential_10/lstm_10/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_10/lstm_10/while/loop_counter
×
ň
-__inference_sequential_10_layer_call_fn_91475

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91043o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
ž
while_cond_90877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_90877___redundant_placeholder03
/while_while_cond_90877___redundant_placeholder13
/while_while_cond_90877___redundant_placeholder23
/while_while_cond_90877___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
	
ž
while_cond_92941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92941___redundant_placeholder03
/while_while_cond_92941___redundant_placeholder13
/while_while_cond_92941___redundant_placeholder23
/while_while_cond_92941___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
Đ7

B__inference_lstm_10_layer_call_and_return_conditional_losses_90747

inputs%
lstm_cell_10_90666:	!
lstm_cell_10_90668:	&
lstm_cell_10_90670:

identity˘$lstm_cell_10/StatefulPartitionedCall˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskô
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_90666lstm_cell_10_90668lstm_cell_10_90670*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90620n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_90666lstm_cell_10_90668lstm_cell_10_90670*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_90679*
condR
while_cond_90678*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ěĂ

 __inference__wrapped_model_90294
lstm_10_inputS
@sequential_10_lstm_10_lstm_cell_10_split_readvariableop_resource:	Q
Bsequential_10_lstm_10_lstm_cell_10_split_1_readvariableop_resource:	N
:sequential_10_lstm_10_lstm_cell_10_readvariableop_resource:
H
5sequential_10_dense_10_matmul_readvariableop_resource:	D
6sequential_10_dense_10_biasadd_readvariableop_resource:
identity˘-sequential_10/dense_10/BiasAdd/ReadVariableOp˘,sequential_10/dense_10/MatMul/ReadVariableOp˘1sequential_10/lstm_10/lstm_cell_10/ReadVariableOp˘3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_1˘3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_2˘3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_3˘7sequential_10/lstm_10/lstm_cell_10/split/ReadVariableOp˘9sequential_10/lstm_10/lstm_cell_10/split_1/ReadVariableOp˘sequential_10/lstm_10/whilef
sequential_10/lstm_10/ShapeShapelstm_10_input*
T0*
_output_shapes
::íĎs
)sequential_10/lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_10/lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_10/lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#sequential_10/lstm_10/strided_sliceStridedSlice$sequential_10/lstm_10/Shape:output:02sequential_10/lstm_10/strided_slice/stack:output:04sequential_10/lstm_10/strided_slice/stack_1:output:04sequential_10/lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_10/lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ľ
"sequential_10/lstm_10/zeros/packedPack,sequential_10/lstm_10/strided_slice:output:0-sequential_10/lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_10/lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ż
sequential_10/lstm_10/zerosFill+sequential_10/lstm_10/zeros/packed:output:0*sequential_10/lstm_10/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
&sequential_10/lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :š
$sequential_10/lstm_10/zeros_1/packedPack,sequential_10/lstm_10/strided_slice:output:0/sequential_10/lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_10/lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ľ
sequential_10/lstm_10/zeros_1Fill-sequential_10/lstm_10/zeros_1/packed:output:0,sequential_10/lstm_10/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
$sequential_10/lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
sequential_10/lstm_10/transpose	Transposelstm_10_input-sequential_10/lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙~
sequential_10/lstm_10/Shape_1Shape#sequential_10/lstm_10/transpose:y:0*
T0*
_output_shapes
::íĎu
+sequential_10/lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_10/lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_10/lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%sequential_10/lstm_10/strided_slice_1StridedSlice&sequential_10/lstm_10/Shape_1:output:04sequential_10/lstm_10/strided_slice_1/stack:output:06sequential_10/lstm_10/strided_slice_1/stack_1:output:06sequential_10/lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_10/lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ö
#sequential_10/lstm_10/TensorArrayV2TensorListReserve:sequential_10/lstm_10/TensorArrayV2/element_shape:output:0.sequential_10/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
Ksequential_10/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ˘
=sequential_10/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_10/transpose:y:0Tsequential_10/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇu
+sequential_10/lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_10/lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_10/lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%sequential_10/lstm_10/strided_slice_2StridedSlice#sequential_10/lstm_10/transpose:y:04sequential_10/lstm_10/strided_slice_2/stack:output:06sequential_10/lstm_10/strided_slice_2/stack_1:output:06sequential_10/lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskt
2sequential_10/lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :š
7sequential_10/lstm_10/lstm_cell_10/split/ReadVariableOpReadVariableOp@sequential_10_lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0
(sequential_10/lstm_10/lstm_cell_10/splitSplit;sequential_10/lstm_10/lstm_cell_10/split/split_dim:output:0?sequential_10/lstm_10/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitÉ
)sequential_10/lstm_10/lstm_cell_10/MatMulMatMul.sequential_10/lstm_10/strided_slice_2:output:01sequential_10/lstm_10/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ë
+sequential_10/lstm_10/lstm_cell_10/MatMul_1MatMul.sequential_10/lstm_10/strided_slice_2:output:01sequential_10/lstm_10/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ë
+sequential_10/lstm_10/lstm_cell_10/MatMul_2MatMul.sequential_10/lstm_10/strided_slice_2:output:01sequential_10/lstm_10/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ë
+sequential_10/lstm_10/lstm_cell_10/MatMul_3MatMul.sequential_10/lstm_10/strided_slice_2:output:01sequential_10/lstm_10/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
4sequential_10/lstm_10/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : š
9sequential_10/lstm_10/lstm_cell_10/split_1/ReadVariableOpReadVariableOpBsequential_10_lstm_10_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0
*sequential_10/lstm_10/lstm_cell_10/split_1Split=sequential_10/lstm_10/lstm_cell_10/split_1/split_dim:output:0Asequential_10/lstm_10/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitŇ
*sequential_10/lstm_10/lstm_cell_10/BiasAddBiasAdd3sequential_10/lstm_10/lstm_cell_10/MatMul:product:03sequential_10/lstm_10/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ö
,sequential_10/lstm_10/lstm_cell_10/BiasAdd_1BiasAdd5sequential_10/lstm_10/lstm_cell_10/MatMul_1:product:03sequential_10/lstm_10/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ö
,sequential_10/lstm_10/lstm_cell_10/BiasAdd_2BiasAdd5sequential_10/lstm_10/lstm_cell_10/MatMul_2:product:03sequential_10/lstm_10/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ö
,sequential_10/lstm_10/lstm_cell_10/BiasAdd_3BiasAdd5sequential_10/lstm_10/lstm_cell_10/MatMul_3:product:03sequential_10/lstm_10/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
1sequential_10/lstm_10/lstm_cell_10/ReadVariableOpReadVariableOp:sequential_10_lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0
6sequential_10/lstm_10/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
8sequential_10/lstm_10/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
8sequential_10/lstm_10/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
0sequential_10/lstm_10/lstm_cell_10/strided_sliceStridedSlice9sequential_10/lstm_10/lstm_cell_10/ReadVariableOp:value:0?sequential_10/lstm_10/lstm_cell_10/strided_slice/stack:output:0Asequential_10/lstm_10/lstm_cell_10/strided_slice/stack_1:output:0Asequential_10/lstm_10/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÉ
+sequential_10/lstm_10/lstm_cell_10/MatMul_4MatMul$sequential_10/lstm_10/zeros:output:09sequential_10/lstm_10/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Î
&sequential_10/lstm_10/lstm_cell_10/addAddV23sequential_10/lstm_10/lstm_cell_10/BiasAdd:output:05sequential_10/lstm_10/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(sequential_10/lstm_10/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>o
*sequential_10/lstm_10/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?ż
&sequential_10/lstm_10/lstm_cell_10/MulMul*sequential_10/lstm_10/lstm_cell_10/add:z:01sequential_10/lstm_10/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
(sequential_10/lstm_10/lstm_cell_10/Add_1AddV2*sequential_10/lstm_10/lstm_cell_10/Mul:z:03sequential_10/lstm_10/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_10/lstm_10/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?é
8sequential_10/lstm_10/lstm_cell_10/clip_by_value/MinimumMinimum,sequential_10/lstm_10/lstm_cell_10/Add_1:z:0Csequential_10/lstm_10/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
2sequential_10/lstm_10/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    é
0sequential_10/lstm_10/lstm_cell_10/clip_by_valueMaximum<sequential_10/lstm_10/lstm_cell_10/clip_by_value/Minimum:z:0;sequential_10/lstm_10/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_1ReadVariableOp:sequential_10_lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0
8sequential_10/lstm_10/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
:sequential_10/lstm_10/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
:sequential_10/lstm_10/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ś
2sequential_10/lstm_10/lstm_cell_10/strided_slice_1StridedSlice;sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_1:value:0Asequential_10/lstm_10/lstm_cell_10/strided_slice_1/stack:output:0Csequential_10/lstm_10/lstm_cell_10/strided_slice_1/stack_1:output:0Csequential_10/lstm_10/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskË
+sequential_10/lstm_10/lstm_cell_10/MatMul_5MatMul$sequential_10/lstm_10/zeros:output:0;sequential_10/lstm_10/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ň
(sequential_10/lstm_10/lstm_cell_10/add_2AddV25sequential_10/lstm_10/lstm_cell_10/BiasAdd_1:output:05sequential_10/lstm_10/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*sequential_10/lstm_10/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>o
*sequential_10/lstm_10/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ĺ
(sequential_10/lstm_10/lstm_cell_10/Mul_1Mul,sequential_10/lstm_10/lstm_cell_10/add_2:z:03sequential_10/lstm_10/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ç
(sequential_10/lstm_10/lstm_cell_10/Add_3AddV2,sequential_10/lstm_10/lstm_cell_10/Mul_1:z:03sequential_10/lstm_10/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
<sequential_10/lstm_10/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?í
:sequential_10/lstm_10/lstm_cell_10/clip_by_value_1/MinimumMinimum,sequential_10/lstm_10/lstm_cell_10/Add_3:z:0Esequential_10/lstm_10/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
4sequential_10/lstm_10/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ď
2sequential_10/lstm_10/lstm_cell_10/clip_by_value_1Maximum>sequential_10/lstm_10/lstm_cell_10/clip_by_value_1/Minimum:z:0=sequential_10/lstm_10/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Â
(sequential_10/lstm_10/lstm_cell_10/mul_2Mul6sequential_10/lstm_10/lstm_cell_10/clip_by_value_1:z:0&sequential_10/lstm_10/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_2ReadVariableOp:sequential_10_lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0
8sequential_10/lstm_10/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
:sequential_10/lstm_10/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
:sequential_10/lstm_10/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ś
2sequential_10/lstm_10/lstm_cell_10/strided_slice_2StridedSlice;sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_2:value:0Asequential_10/lstm_10/lstm_cell_10/strided_slice_2/stack:output:0Csequential_10/lstm_10/lstm_cell_10/strided_slice_2/stack_1:output:0Csequential_10/lstm_10/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskË
+sequential_10/lstm_10/lstm_cell_10/MatMul_6MatMul$sequential_10/lstm_10/zeros:output:0;sequential_10/lstm_10/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ň
(sequential_10/lstm_10/lstm_cell_10/add_4AddV25sequential_10/lstm_10/lstm_cell_10/BiasAdd_2:output:05sequential_10/lstm_10/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'sequential_10/lstm_10/lstm_cell_10/ReluRelu,sequential_10/lstm_10/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ď
(sequential_10/lstm_10/lstm_cell_10/mul_3Mul4sequential_10/lstm_10/lstm_cell_10/clip_by_value:z:05sequential_10/lstm_10/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
(sequential_10/lstm_10/lstm_cell_10/add_5AddV2,sequential_10/lstm_10/lstm_cell_10/mul_2:z:0,sequential_10/lstm_10/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_3ReadVariableOp:sequential_10_lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0
8sequential_10/lstm_10/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
:sequential_10/lstm_10/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:sequential_10/lstm_10/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ś
2sequential_10/lstm_10/lstm_cell_10/strided_slice_3StridedSlice;sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_3:value:0Asequential_10/lstm_10/lstm_cell_10/strided_slice_3/stack:output:0Csequential_10/lstm_10/lstm_cell_10/strided_slice_3/stack_1:output:0Csequential_10/lstm_10/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskË
+sequential_10/lstm_10/lstm_cell_10/MatMul_7MatMul$sequential_10/lstm_10/zeros:output:0;sequential_10/lstm_10/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ň
(sequential_10/lstm_10/lstm_cell_10/add_6AddV25sequential_10/lstm_10/lstm_cell_10/BiasAdd_3:output:05sequential_10/lstm_10/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*sequential_10/lstm_10/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>o
*sequential_10/lstm_10/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ĺ
(sequential_10/lstm_10/lstm_cell_10/Mul_4Mul,sequential_10/lstm_10/lstm_cell_10/add_6:z:03sequential_10/lstm_10/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ç
(sequential_10/lstm_10/lstm_cell_10/Add_7AddV2,sequential_10/lstm_10/lstm_cell_10/Mul_4:z:03sequential_10/lstm_10/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
<sequential_10/lstm_10/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?í
:sequential_10/lstm_10/lstm_cell_10/clip_by_value_2/MinimumMinimum,sequential_10/lstm_10/lstm_cell_10/Add_7:z:0Esequential_10/lstm_10/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
4sequential_10/lstm_10/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ď
2sequential_10/lstm_10/lstm_cell_10/clip_by_value_2Maximum>sequential_10/lstm_10/lstm_cell_10/clip_by_value_2/Minimum:z:0=sequential_10/lstm_10/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential_10/lstm_10/lstm_cell_10/Relu_1Relu,sequential_10/lstm_10/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ó
(sequential_10/lstm_10/lstm_cell_10/mul_5Mul6sequential_10/lstm_10/lstm_cell_10/clip_by_value_2:z:07sequential_10/lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
3sequential_10/lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ú
%sequential_10/lstm_10/TensorArrayV2_1TensorListReserve<sequential_10/lstm_10/TensorArrayV2_1/element_shape:output:0.sequential_10/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ\
sequential_10/lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_10/lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙j
(sequential_10/lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ž
sequential_10/lstm_10/whileWhile1sequential_10/lstm_10/while/loop_counter:output:07sequential_10/lstm_10/while/maximum_iterations:output:0#sequential_10/lstm_10/time:output:0.sequential_10/lstm_10/TensorArrayV2_1:handle:0$sequential_10/lstm_10/zeros:output:0&sequential_10/lstm_10/zeros_1:output:0.sequential_10/lstm_10/strided_slice_1:output:0Msequential_10/lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_10_lstm_10_lstm_cell_10_split_readvariableop_resourceBsequential_10_lstm_10_lstm_cell_10_split_1_readvariableop_resource:sequential_10_lstm_10_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_10_lstm_10_while_body_90148*2
cond*R(
&sequential_10_lstm_10_while_cond_90147*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
Fsequential_10/lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
8sequential_10/lstm_10/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_10/while:output:3Osequential_10/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0~
+sequential_10/lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙w
-sequential_10/lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_10/lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
%sequential_10/lstm_10/strided_slice_3StridedSliceAsequential_10/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_10/strided_slice_3/stack:output:06sequential_10/lstm_10/strided_slice_3/stack_1:output:06sequential_10/lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask{
&sequential_10/lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ů
!sequential_10/lstm_10/transpose_1	TransposeAsequential_10/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
,sequential_10/dense_10/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_10_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ż
sequential_10/dense_10/MatMulMatMul.sequential_10/lstm_10/strided_slice_3:output:04sequential_10/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
-sequential_10/dense_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ť
sequential_10/dense_10/BiasAddBiasAdd'sequential_10/dense_10/MatMul:product:05sequential_10/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
IdentityIdentity'sequential_10/dense_10/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp.^sequential_10/dense_10/BiasAdd/ReadVariableOp-^sequential_10/dense_10/MatMul/ReadVariableOp2^sequential_10/lstm_10/lstm_cell_10/ReadVariableOp4^sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_14^sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_24^sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_38^sequential_10/lstm_10/lstm_cell_10/split/ReadVariableOp:^sequential_10/lstm_10/lstm_cell_10/split_1/ReadVariableOp^sequential_10/lstm_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2^
-sequential_10/dense_10/BiasAdd/ReadVariableOp-sequential_10/dense_10/BiasAdd/ReadVariableOp2\
,sequential_10/dense_10/MatMul/ReadVariableOp,sequential_10/dense_10/MatMul/ReadVariableOp2j
3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_13sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_12j
3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_23sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_22j
3sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_33sequential_10/lstm_10/lstm_cell_10/ReadVariableOp_32f
1sequential_10/lstm_10/lstm_cell_10/ReadVariableOp1sequential_10/lstm_10/lstm_cell_10/ReadVariableOp2r
7sequential_10/lstm_10/lstm_cell_10/split/ReadVariableOp7sequential_10/lstm_10/lstm_cell_10/split/ReadVariableOp2v
9sequential_10/lstm_10/lstm_cell_10/split_1/ReadVariableOp9sequential_10/lstm_10/lstm_cell_10/split_1/ReadVariableOp2:
sequential_10/lstm_10/whilesequential_10/lstm_10/while:Z V
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namelstm_10_input
	
ž
while_cond_92685
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92685___redundant_placeholder03
/while_while_cond_92685___redundant_placeholder13
/while_while_cond_92685___redundant_placeholder23
/while_while_cond_92685___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
ě
ů
-__inference_sequential_10_layer_call_fn_91056
lstm_10_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91043o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namelstm_10_input

˝
lstm_10_while_body_91868,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0M
:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0:	K
<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0:	H
4lstm_10_while_lstm_cell_10_readvariableop_resource_0:

lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorK
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:	I
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource:	F
2lstm_10_while_lstm_cell_10_readvariableop_resource:
˘)lstm_10/while/lstm_cell_10/ReadVariableOp˘+lstm_10/while/lstm_cell_10/ReadVariableOp_1˘+lstm_10/while/lstm_cell_10/ReadVariableOp_2˘+lstm_10/while/lstm_cell_10/ReadVariableOp_3˘/lstm_10/while/lstm_cell_10/split/ReadVariableOp˘1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Î
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0l
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ť
/lstm_10/while/lstm_cell_10/split/ReadVariableOpReadVariableOp:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0ó
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:07lstm_10/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitĂ
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#lstm_10/while/lstm_cell_10/MatMul_1MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#lstm_10/while/lstm_cell_10/MatMul_2MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
#lstm_10/while/lstm_cell_10/MatMul_3MatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm_10/while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
,lstm_10/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ť
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0é
"lstm_10/while/lstm_cell_10/split_1Split5lstm_10/while/lstm_cell_10/split_1/split_dim:output:09lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitş
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd+lstm_10/while/lstm_cell_10/MatMul:product:0+lstm_10/while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
$lstm_10/while/lstm_cell_10/BiasAdd_1BiasAdd-lstm_10/while/lstm_cell_10/MatMul_1:product:0+lstm_10/while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
$lstm_10/while/lstm_cell_10/BiasAdd_2BiasAdd-lstm_10/while/lstm_cell_10/MatMul_2:product:0+lstm_10/while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
$lstm_10/while/lstm_cell_10/BiasAdd_3BiasAdd-lstm_10/while/lstm_cell_10/MatMul_3:product:0+lstm_10/while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
)lstm_10/while/lstm_cell_10/ReadVariableOpReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.lstm_10/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0lstm_10/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_10/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_10/while/lstm_cell_10/strided_sliceStridedSlice1lstm_10/while/lstm_cell_10/ReadVariableOp:value:07lstm_10/while/lstm_cell_10/strided_slice/stack:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_1:output:09lstm_10/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask°
#lstm_10/while/lstm_cell_10/MatMul_4MatMullstm_10_while_placeholder_21lstm_10/while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ś
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/BiasAdd:output:0-lstm_10/while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
 lstm_10/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>g
"lstm_10/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?§
lstm_10/while/lstm_cell_10/MulMul"lstm_10/while/lstm_cell_10/add:z:0)lstm_10/while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
 lstm_10/while/lstm_cell_10/Add_1AddV2"lstm_10/while/lstm_cell_10/Mul:z:0+lstm_10/while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
2lstm_10/while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ń
0lstm_10/while/lstm_cell_10/clip_by_value/MinimumMinimum$lstm_10/while/lstm_cell_10/Add_1:z:0;lstm_10/while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*lstm_10/while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ń
(lstm_10/while/lstm_cell_10/clip_by_valueMaximum4lstm_10/while/lstm_cell_10/clip_by_value/Minimum:z:03lstm_10/while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+lstm_10/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0lstm_10/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2lstm_10/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ţ
*lstm_10/while/lstm_cell_10/strided_slice_1StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_1:value:09lstm_10/while/lstm_cell_10/strided_slice_1/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask˛
#lstm_10/while/lstm_cell_10/MatMul_5MatMullstm_10_while_placeholder_23lstm_10/while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
 lstm_10/while/lstm_cell_10/add_2AddV2-lstm_10/while/lstm_cell_10/BiasAdd_1:output:0-lstm_10/while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_10/while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>g
"lstm_10/while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?­
 lstm_10/while/lstm_cell_10/Mul_1Mul$lstm_10/while/lstm_cell_10/add_2:z:0+lstm_10/while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
 lstm_10/while/lstm_cell_10/Add_3AddV2$lstm_10/while/lstm_cell_10/Mul_1:z:0+lstm_10/while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
4lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ő
2lstm_10/while/lstm_cell_10/clip_by_value_1/MinimumMinimum$lstm_10/while/lstm_cell_10/Add_3:z:0=lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,lstm_10/while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ×
*lstm_10/while/lstm_cell_10/clip_by_value_1Maximum6lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum:z:05lstm_10/while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙§
 lstm_10/while/lstm_cell_10/mul_2Mul.lstm_10/while/lstm_cell_10/clip_by_value_1:z:0lstm_10_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+lstm_10/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0lstm_10/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
2lstm_10/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ţ
*lstm_10/while/lstm_cell_10/strided_slice_2StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_2:value:09lstm_10/while/lstm_cell_10/strided_slice_2/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask˛
#lstm_10/while/lstm_cell_10/MatMul_6MatMullstm_10_while_placeholder_23lstm_10/while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
 lstm_10/while/lstm_cell_10/add_4AddV2-lstm_10/while/lstm_cell_10/BiasAdd_2:output:0-lstm_10/while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/while/lstm_cell_10/ReluRelu$lstm_10/while/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
 lstm_10/while/lstm_cell_10/mul_3Mul,lstm_10/while/lstm_cell_10/clip_by_value:z:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
 lstm_10/while/lstm_cell_10/add_5AddV2$lstm_10/while/lstm_cell_10/mul_2:z:0$lstm_10/while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
+lstm_10/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp4lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0lstm_10/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
2lstm_10/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ţ
*lstm_10/while/lstm_cell_10/strided_slice_3StridedSlice3lstm_10/while/lstm_cell_10/ReadVariableOp_3:value:09lstm_10/while/lstm_cell_10/strided_slice_3/stack:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_1:output:0;lstm_10/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask˛
#lstm_10/while/lstm_cell_10/MatMul_7MatMullstm_10_while_placeholder_23lstm_10/while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
 lstm_10/while/lstm_cell_10/add_6AddV2-lstm_10/while/lstm_cell_10/BiasAdd_3:output:0-lstm_10/while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"lstm_10/while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>g
"lstm_10/while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?­
 lstm_10/while/lstm_cell_10/Mul_4Mul$lstm_10/while/lstm_cell_10/add_6:z:0+lstm_10/while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
 lstm_10/while/lstm_cell_10/Add_7AddV2$lstm_10/while/lstm_cell_10/Mul_4:z:0+lstm_10/while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
4lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ő
2lstm_10/while/lstm_cell_10/clip_by_value_2/MinimumMinimum$lstm_10/while/lstm_cell_10/Add_7:z:0=lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,lstm_10/while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ×
*lstm_10/while/lstm_cell_10/clip_by_value_2Maximum6lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum:z:05lstm_10/while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
 lstm_10/while/lstm_cell_10/mul_5Mul.lstm_10/while/lstm_cell_10/clip_by_value_2:z:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ĺ
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1lstm_10_while_placeholder$lstm_10/while/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇU
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: q
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_5:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_5:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
lstm_10/while/NoOpNoOp*^lstm_10/while/lstm_cell_10/ReadVariableOp,^lstm_10/while/lstm_cell_10/ReadVariableOp_1,^lstm_10/while/lstm_cell_10/ReadVariableOp_2,^lstm_10/while/lstm_cell_10/ReadVariableOp_30^lstm_10/while/lstm_cell_10/split/ReadVariableOp2^lstm_10/while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"9
lstm_10_while_identitylstm_10/while/Identity:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"j
2lstm_10_while_lstm_cell_10_readvariableop_resource4lstm_10_while_lstm_cell_10_readvariableop_resource_0"z
:lstm_10_while_lstm_cell_10_split_1_readvariableop_resource<lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0"v
8lstm_10_while_lstm_cell_10_split_readvariableop_resource:lstm_10_while_lstm_cell_10_split_readvariableop_resource_0"Č
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_1+lstm_10/while/lstm_cell_10/ReadVariableOp_12Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_2+lstm_10/while/lstm_cell_10/ReadVariableOp_22Z
+lstm_10/while/lstm_cell_10/ReadVariableOp_3+lstm_10/while/lstm_cell_10/ReadVariableOp_32V
)lstm_10/while/lstm_cell_10/ReadVariableOp)lstm_10/while/lstm_cell_10/ReadVariableOp2b
/lstm_10/while/lstm_cell_10/split/ReadVariableOp/lstm_10/while/lstm_cell_10/split/ReadVariableOp2f
1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp1lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
_user_specified_name" lstm_10/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_10/while/loop_counter
ě
ů
-__inference_sequential_10_layer_call_fn_91405
lstm_10_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91377o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namelstm_10_input

ˇ
'__inference_lstm_10_layer_call_fn_92036
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_90747p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0
˘
ô
H__inference_sequential_10_layer_call_and_return_conditional_losses_91752

inputsE
2lstm_10_lstm_cell_10_split_readvariableop_resource:	C
4lstm_10_lstm_cell_10_split_1_readvariableop_resource:	@
,lstm_10_lstm_cell_10_readvariableop_resource:
:
'dense_10_matmul_readvariableop_resource:	6
(dense_10_biasadd_readvariableop_resource:
identity˘dense_10/BiasAdd/ReadVariableOp˘dense_10/MatMul/ReadVariableOp˘#lstm_10/lstm_cell_10/ReadVariableOp˘%lstm_10/lstm_cell_10/ReadVariableOp_1˘%lstm_10/lstm_cell_10/ReadVariableOp_2˘%lstm_10/lstm_cell_10/ReadVariableOp_3˘)lstm_10/lstm_cell_10/split/ReadVariableOp˘+lstm_10/lstm_cell_10/split_1/ReadVariableOp˘lstm_10/whileQ
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
::íĎe
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ů
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
::íĎg
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ř
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇg
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskf
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm_10/lstm_cell_10/split/ReadVariableOpReadVariableOp2lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0á
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:01lstm_10/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_10/lstm_cell_10/MatMul_1MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_10/lstm_cell_10/MatMul_2MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_10/lstm_cell_10/MatMul_3MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
&lstm_10/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm_10/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0×
lstm_10/lstm_cell_10/split_1Split/lstm_10/lstm_cell_10/split_1/split_dim:output:03lstm_10/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¨
lstm_10/lstm_cell_10/BiasAddBiasAdd%lstm_10/lstm_cell_10/MatMul:product:0%lstm_10/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
lstm_10/lstm_cell_10/BiasAdd_1BiasAdd'lstm_10/lstm_cell_10/MatMul_1:product:0%lstm_10/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
lstm_10/lstm_cell_10/BiasAdd_2BiasAdd'lstm_10/lstm_cell_10/MatMul_2:product:0%lstm_10/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
lstm_10/lstm_cell_10/BiasAdd_3BiasAdd'lstm_10/lstm_cell_10/MatMul_3:product:0%lstm_10/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_10/lstm_cell_10/ReadVariableOpReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_10/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_10/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_10/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_10/lstm_cell_10/strided_sliceStridedSlice+lstm_10/lstm_cell_10/ReadVariableOp:value:01lstm_10/lstm_cell_10/strided_slice/stack:output:03lstm_10/lstm_cell_10/strided_slice/stack_1:output:03lstm_10/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_10/lstm_cell_10/MatMul_4MatMullstm_10/zeros:output:0+lstm_10/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/BiasAdd:output:0'lstm_10/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_10/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>a
lstm_10/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_10/lstm_cell_10/MulMullstm_10/lstm_cell_10/add:z:0#lstm_10/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/Add_1AddV2lstm_10/lstm_cell_10/Mul:z:0%lstm_10/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,lstm_10/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ż
*lstm_10/lstm_cell_10/clip_by_value/MinimumMinimumlstm_10/lstm_cell_10/Add_1:z:05lstm_10/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_10/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"lstm_10/lstm_cell_10/clip_by_valueMaximum.lstm_10/lstm_cell_10/clip_by_value/Minimum:z:0-lstm_10/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%lstm_10/lstm_cell_10/ReadVariableOp_1ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*lstm_10/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_10/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_10/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$lstm_10/lstm_cell_10/strided_slice_1StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_1:value:03lstm_10/lstm_cell_10/strided_slice_1/stack:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĄ
lstm_10/lstm_cell_10/MatMul_5MatMullstm_10/zeros:output:0-lstm_10/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
lstm_10/lstm_cell_10/add_2AddV2'lstm_10/lstm_cell_10/BiasAdd_1:output:0'lstm_10/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_10/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>a
lstm_10/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_10/lstm_cell_10/Mul_1Mullstm_10/lstm_cell_10/add_2:z:0%lstm_10/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/Add_3AddV2lstm_10/lstm_cell_10/Mul_1:z:0%lstm_10/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
.lstm_10/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
,lstm_10/lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_10/lstm_cell_10/Add_3:z:07lstm_10/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_10/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ĺ
$lstm_10/lstm_cell_10/clip_by_value_1Maximum0lstm_10/lstm_cell_10/clip_by_value_1/Minimum:z:0/lstm_10/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/mul_2Mul(lstm_10/lstm_cell_10/clip_by_value_1:z:0lstm_10/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%lstm_10/lstm_cell_10/ReadVariableOp_2ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*lstm_10/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_10/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm_10/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$lstm_10/lstm_cell_10/strided_slice_2StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_2:value:03lstm_10/lstm_cell_10/strided_slice_2/stack:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĄ
lstm_10/lstm_cell_10/MatMul_6MatMullstm_10/zeros:output:0-lstm_10/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
lstm_10/lstm_cell_10/add_4AddV2'lstm_10/lstm_cell_10/BiasAdd_2:output:0'lstm_10/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_10/lstm_cell_10/ReluRelulstm_10/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
lstm_10/lstm_cell_10/mul_3Mul&lstm_10/lstm_cell_10/clip_by_value:z:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/add_5AddV2lstm_10/lstm_cell_10/mul_2:z:0lstm_10/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%lstm_10/lstm_cell_10/ReadVariableOp_3ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*lstm_10/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm_10/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_10/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$lstm_10/lstm_cell_10/strided_slice_3StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_3:value:03lstm_10/lstm_cell_10/strided_slice_3/stack:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĄ
lstm_10/lstm_cell_10/MatMul_7MatMullstm_10/zeros:output:0-lstm_10/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
lstm_10/lstm_cell_10/add_6AddV2'lstm_10/lstm_cell_10/BiasAdd_3:output:0'lstm_10/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_10/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>a
lstm_10/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_10/lstm_cell_10/Mul_4Mullstm_10/lstm_cell_10/add_6:z:0%lstm_10/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/Add_7AddV2lstm_10/lstm_cell_10/Mul_4:z:0%lstm_10/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
.lstm_10/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
,lstm_10/lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_10/lstm_cell_10/Add_7:z:07lstm_10/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_10/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ĺ
$lstm_10/lstm_cell_10/clip_by_value_2Maximum0lstm_10/lstm_cell_10/clip_by_value_2/Minimum:z:0/lstm_10/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
lstm_10/lstm_cell_10/mul_5Mul(lstm_10/lstm_cell_10/clip_by_value_2:z:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Đ
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇN
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙\
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ę
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_10_lstm_cell_10_split_readvariableop_resource4lstm_10_lstm_cell_10_split_1_readvariableop_resource,lstm_10_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_10_while_body_91606*$
condR
lstm_10_while_cond_91605*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ű
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0p
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙i
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskm
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ż
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_10/MatMulMatMul lstm_10/strided_slice_3:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
IdentityIdentitydense_10/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp$^lstm_10/lstm_cell_10/ReadVariableOp&^lstm_10/lstm_cell_10/ReadVariableOp_1&^lstm_10/lstm_cell_10/ReadVariableOp_2&^lstm_10/lstm_cell_10/ReadVariableOp_3*^lstm_10/lstm_cell_10/split/ReadVariableOp,^lstm_10/lstm_cell_10/split_1/ReadVariableOp^lstm_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2N
%lstm_10/lstm_cell_10/ReadVariableOp_1%lstm_10/lstm_cell_10/ReadVariableOp_12N
%lstm_10/lstm_cell_10/ReadVariableOp_2%lstm_10/lstm_cell_10/ReadVariableOp_22N
%lstm_10/lstm_cell_10/ReadVariableOp_3%lstm_10/lstm_cell_10/ReadVariableOp_32J
#lstm_10/lstm_cell_10/ReadVariableOp#lstm_10/lstm_cell_10/ReadVariableOp2V
)lstm_10/lstm_cell_10/split/ReadVariableOp)lstm_10/lstm_cell_10/split/ReadVariableOp2Z
+lstm_10/lstm_cell_10/split_1/ReadVariableOp+lstm_10/lstm_cell_10/split_1/ReadVariableOp2
lstm_10/whilelstm_10/while:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ţ
lstm_10_while_cond_91867,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1C
?lstm_10_while_lstm_10_while_cond_91867___redundant_placeholder0C
?lstm_10_while_lstm_10_while_cond_91867___redundant_placeholder1C
?lstm_10_while_lstm_10_while_cond_91867___redundant_placeholder2C
?lstm_10_while_lstm_10_while_cond_91867___redundant_placeholder3
lstm_10_while_identity

lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: [
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
_user_specified_name" lstm_10/while/maximum_iterations:R N

_output_shapes
: 
4
_user_specified_namelstm_10/while/loop_counter

é
B__inference_lstm_10_layer_call_and_return_conditional_losses_92570
inputs_0=
*lstm_cell_10_split_readvariableop_resource:	;
,lstm_cell_10_split_1_readvariableop_resource:	8
$lstm_cell_10_readvariableop_resource:

identity˘lstm_cell_10/ReadVariableOp˘lstm_cell_10/ReadVariableOp_1˘lstm_cell_10/ReadVariableOp_2˘lstm_cell_10/ReadVariableOp_3˘!lstm_cell_10/split/ReadVariableOp˘#lstm_cell_10/split_1/ReadVariableOp˘whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0É
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ż
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_4MatMulzeros:output:0#lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_10/MulMullstm_cell_10/add:z:0lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_1AddV2lstm_cell_10/Mul:z:0lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
"lstm_cell_10/clip_by_value/MinimumMinimumlstm_cell_10/Add_1:z:0-lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    §
lstm_cell_10/clip_by_valueMaximum&lstm_cell_10/clip_by_value/Minimum:z:0%lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_5MatMulzeros:output:0%lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_1Mullstm_cell_10/add_2:z:0lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_3AddV2lstm_cell_10/Mul_1:z:0lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_cell_10/Add_3:z:0/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_1Maximum(lstm_cell_10/clip_by_value_1/Minimum:z:0'lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_2Mul lstm_cell_10/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_6MatMulzeros:output:0%lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_10/ReluRelulstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_3Mullstm_cell_10/clip_by_value:z:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_10/add_5AddV2lstm_cell_10/mul_2:z:0lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_7MatMulzeros:output:0%lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_6AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_4Mullstm_cell_10/add_6:z:0lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_7AddV2lstm_cell_10/Mul_4:z:0lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_cell_10/Add_7:z:0/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_2Maximum(lstm_cell_10/clip_by_value_2/Minimum:z:0'lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_10/Relu_1Relulstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_5Mul lstm_cell_10/clip_by_value_2:z:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_92430*
condR
while_cond_92429*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0
ř
ľ
'__inference_lstm_10_layer_call_fn_92047

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_91018p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
 ~
Ľ	
while_body_91195
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	C
4while_lstm_cell_10_split_1_readvariableop_resource_0:	@
,while_lstm_cell_10_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	A
2while_lstm_cell_10_split_1_readvariableop_resource:	>
*while_lstm_cell_10_readvariableop_resource:
˘!while/lstm_cell_10/ReadVariableOp˘#while/lstm_cell_10/ReadVariableOp_1˘#while/lstm_cell_10/ReadVariableOp_2˘#while/lstm_cell_10/ReadVariableOp_3˘'while/lstm_cell_10/split/ReadVariableOp˘)while/lstm_cell_10/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ű
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitŤ
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ń
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/MulMulwhile/lstm_cell_10/add:z:0!while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_1AddV2while/lstm_cell_10/Mul:z:0#while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?š
(while/lstm_cell_10/clip_by_value/MinimumMinimumwhile/lstm_cell_10/Add_1:z:03while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    š
 while/lstm_cell_10/clip_by_valueMaximum,while/lstm_cell_10/clip_by_value/Minimum:z:0+while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_1Mulwhile/lstm_cell_10/add_2:z:0#while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_3AddV2while/lstm_cell_10/Mul_1:z:0#while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_1/MinimumMinimumwhile/lstm_cell_10/Add_3:z:05while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_1Maximum.while/lstm_cell_10/clip_by_value_1/Minimum:z:0-while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_2Mul&while/lstm_cell_10/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_3Mul$while/lstm_cell_10/clip_by_value:z:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/add_5AddV2while/lstm_cell_10/mul_2:z:0while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_6AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_4Mulwhile/lstm_cell_10/add_6:z:0#while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_7AddV2while/lstm_cell_10/Mul_4:z:0#while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_2/MinimumMinimumwhile/lstm_cell_10/Add_7:z:05while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_2Maximum.while/lstm_cell_10/clip_by_value_2/Minimum:z:0-while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_10/mul_5Mul&while/lstm_cell_10/clip_by_value_2:z:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_10/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/Identity_5Identitywhile/lstm_cell_10/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
Ňb

!__inference__traced_restore_93544
file_prefix3
 assignvariableop_dense_10_kernel:	.
 assignvariableop_1_dense_10_bias:A
.assignvariableop_2_lstm_10_lstm_cell_10_kernel:	L
8assignvariableop_3_lstm_10_lstm_cell_10_recurrent_kernel:
;
,assignvariableop_4_lstm_10_lstm_cell_10_bias:	#
assignvariableop_5_beta_1: #
assignvariableop_6_beta_2: "
assignvariableop_7_decay: *
 assignvariableop_8_learning_rate: &
assignvariableop_9_adam_iter:	 #
assignvariableop_10_total: #
assignvariableop_11_count: =
*assignvariableop_12_adam_dense_10_kernel_m:	6
(assignvariableop_13_adam_dense_10_bias_m:I
6assignvariableop_14_adam_lstm_10_lstm_cell_10_kernel_m:	T
@assignvariableop_15_adam_lstm_10_lstm_cell_10_recurrent_kernel_m:
C
4assignvariableop_16_adam_lstm_10_lstm_cell_10_bias_m:	=
*assignvariableop_17_adam_dense_10_kernel_v:	6
(assignvariableop_18_adam_dense_10_bias_v:I
6assignvariableop_19_adam_lstm_10_lstm_cell_10_kernel_v:	T
@assignvariableop_20_adam_lstm_10_lstm_cell_10_recurrent_kernel_v:
C
4assignvariableop_21_adam_lstm_10_lstm_cell_10_bias_v:	
identity_23˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¨
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ł
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ĺ
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_10_lstm_cell_10_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_10_lstm_cell_10_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_10_lstm_cell_10_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_5AssignVariableOpassignvariableop_5_beta_1Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_6AssignVariableOpassignvariableop_6_beta_2Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_7AssignVariableOpassignvariableop_7_decayIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_8AssignVariableOp assignvariableop_8_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:ł
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:˛
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_dense_10_kernel_mIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_dense_10_bias_mIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_lstm_10_lstm_cell_10_kernel_mIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ů
AssignVariableOp_15AssignVariableOp@assignvariableop_15_adam_lstm_10_lstm_cell_10_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_lstm_10_lstm_cell_10_bias_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ă
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_10_kernel_vIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_10_bias_vIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ď
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_10_lstm_cell_10_kernel_vIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ů
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_10_lstm_cell_10_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_10_lstm_cell_10_bias_vIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ł
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_2AssignVariableOp_22(
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
	
ž
while_cond_90431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_90431___redundant_placeholder03
/while_while_cond_90431___redundant_placeholder13
/while_while_cond_90431___redundant_placeholder23
/while_while_cond_90431___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
 ~
Ľ	
while_body_92686
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	C
4while_lstm_cell_10_split_1_readvariableop_resource_0:	@
,while_lstm_cell_10_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	A
2while_lstm_cell_10_split_1_readvariableop_resource:	>
*while_lstm_cell_10_readvariableop_resource:
˘!while/lstm_cell_10/ReadVariableOp˘#while/lstm_cell_10/ReadVariableOp_1˘#while/lstm_cell_10/ReadVariableOp_2˘#while/lstm_cell_10/ReadVariableOp_3˘'while/lstm_cell_10/split/ReadVariableOp˘)while/lstm_cell_10/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ű
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitŤ
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ń
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/MulMulwhile/lstm_cell_10/add:z:0!while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_1AddV2while/lstm_cell_10/Mul:z:0#while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?š
(while/lstm_cell_10/clip_by_value/MinimumMinimumwhile/lstm_cell_10/Add_1:z:03while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    š
 while/lstm_cell_10/clip_by_valueMaximum,while/lstm_cell_10/clip_by_value/Minimum:z:0+while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_1Mulwhile/lstm_cell_10/add_2:z:0#while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_3AddV2while/lstm_cell_10/Mul_1:z:0#while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_1/MinimumMinimumwhile/lstm_cell_10/Add_3:z:05while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_1Maximum.while/lstm_cell_10/clip_by_value_1/Minimum:z:0-while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_2Mul&while/lstm_cell_10/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_3Mul$while/lstm_cell_10/clip_by_value:z:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/add_5AddV2while/lstm_cell_10/mul_2:z:0while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_6AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_4Mulwhile/lstm_cell_10/add_6:z:0#while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_7AddV2while/lstm_cell_10/Mul_4:z:0#while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_2/MinimumMinimumwhile/lstm_cell_10/Add_7:z:05while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_2Maximum.while/lstm_cell_10/clip_by_value_2/Minimum:z:0-while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_10/mul_5Mul&while/lstm_cell_10/clip_by_value_2:z:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_10/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/Identity_5Identitywhile/lstm_cell_10/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
	
ž
while_cond_92429
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92429___redundant_placeholder03
/while_while_cond_92429___redundant_placeholder13
/while_while_cond_92429___redundant_placeholder23
/while_while_cond_92429___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
 ~
Ľ	
while_body_92430
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	C
4while_lstm_cell_10_split_1_readvariableop_resource_0:	@
,while_lstm_cell_10_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	A
2while_lstm_cell_10_split_1_readvariableop_resource:	>
*while_lstm_cell_10_readvariableop_resource:
˘!while/lstm_cell_10/ReadVariableOp˘#while/lstm_cell_10/ReadVariableOp_1˘#while/lstm_cell_10/ReadVariableOp_2˘#while/lstm_cell_10/ReadVariableOp_3˘'while/lstm_cell_10/split/ReadVariableOp˘)while/lstm_cell_10/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ű
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitŤ
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ń
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/MulMulwhile/lstm_cell_10/add:z:0!while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_1AddV2while/lstm_cell_10/Mul:z:0#while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?š
(while/lstm_cell_10/clip_by_value/MinimumMinimumwhile/lstm_cell_10/Add_1:z:03while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    š
 while/lstm_cell_10/clip_by_valueMaximum,while/lstm_cell_10/clip_by_value/Minimum:z:0+while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_1Mulwhile/lstm_cell_10/add_2:z:0#while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_3AddV2while/lstm_cell_10/Mul_1:z:0#while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_1/MinimumMinimumwhile/lstm_cell_10/Add_3:z:05while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_1Maximum.while/lstm_cell_10/clip_by_value_1/Minimum:z:0-while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_2Mul&while/lstm_cell_10/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_3Mul$while/lstm_cell_10/clip_by_value:z:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/add_5AddV2while/lstm_cell_10/mul_2:z:0while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_6AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_4Mulwhile/lstm_cell_10/add_6:z:0#while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_7AddV2while/lstm_cell_10/Mul_4:z:0#while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_2/MinimumMinimumwhile/lstm_cell_10/Add_7:z:05while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_2Maximum.while/lstm_cell_10/clip_by_value_2/Minimum:z:0-while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_10/mul_5Mul&while/lstm_cell_10/clip_by_value_2:z:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_10/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/Identity_5Identitywhile/lstm_cell_10/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
 ~
Ľ	
while_body_92174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	C
4while_lstm_cell_10_split_1_readvariableop_resource_0:	@
,while_lstm_cell_10_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	A
2while_lstm_cell_10_split_1_readvariableop_resource:	>
*while_lstm_cell_10_readvariableop_resource:
˘!while/lstm_cell_10/ReadVariableOp˘#while/lstm_cell_10/ReadVariableOp_1˘#while/lstm_cell_10/ReadVariableOp_2˘#while/lstm_cell_10/ReadVariableOp_3˘'while/lstm_cell_10/split/ReadVariableOp˘)while/lstm_cell_10/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ű
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitŤ
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ń
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/MulMulwhile/lstm_cell_10/add:z:0!while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_1AddV2while/lstm_cell_10/Mul:z:0#while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?š
(while/lstm_cell_10/clip_by_value/MinimumMinimumwhile/lstm_cell_10/Add_1:z:03while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    š
 while/lstm_cell_10/clip_by_valueMaximum,while/lstm_cell_10/clip_by_value/Minimum:z:0+while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_1Mulwhile/lstm_cell_10/add_2:z:0#while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_3AddV2while/lstm_cell_10/Mul_1:z:0#while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_1/MinimumMinimumwhile/lstm_cell_10/Add_3:z:05while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_1Maximum.while/lstm_cell_10/clip_by_value_1/Minimum:z:0-while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_2Mul&while/lstm_cell_10/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_3Mul$while/lstm_cell_10/clip_by_value:z:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/add_5AddV2while/lstm_cell_10/mul_2:z:0while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_6AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_4Mulwhile/lstm_cell_10/add_6:z:0#while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_7AddV2while/lstm_cell_10/Mul_4:z:0#while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_2/MinimumMinimumwhile/lstm_cell_10/Add_7:z:05while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_2Maximum.while/lstm_cell_10/clip_by_value_2/Minimum:z:0-while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_10/mul_5Mul&while/lstm_cell_10/clip_by_value_2:z:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_10/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/Identity_5Identitywhile/lstm_cell_10/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
Ţ
×
H__inference_sequential_10_layer_call_and_return_conditional_losses_91421
lstm_10_input 
lstm_10_91408:	
lstm_10_91410:	!
lstm_10_91412:
!
dense_10_91415:	
dense_10_91417:
identity˘ dense_10/StatefulPartitionedCall˘lstm_10/StatefulPartitionedCall
lstm_10/StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputlstm_10_91408lstm_10_91410lstm_10_91412*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_91018
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_10_91415dense_10_91417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_91036x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall:Z V
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namelstm_10_input
K
Ť
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93224

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0˘
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
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
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
ReluRelu	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Relu_1Relu	add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
ž
while_cond_92173
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_92173___redundant_placeholder03
/while_while_cond_92173___redundant_placeholder13
/while_while_cond_92173___redundant_placeholder23
/while_while_cond_92173___redundant_placeholder3
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
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
É
Đ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91377

inputs 
lstm_10_91364:	
lstm_10_91366:	!
lstm_10_91368:
!
dense_10_91371:	
dense_10_91373:
identity˘ dense_10/StatefulPartitionedCall˘lstm_10/StatefulPartitionedCallű
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_91364lstm_10_91366lstm_10_91368*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_10_layer_call_and_return_conditional_losses_91335
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_10_91371dense_10_91373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_91036x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Â#
Ţ
while_body_90679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_10_90703_0:	)
while_lstm_cell_10_90705_0:	.
while_lstm_cell_10_90707_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_10_90703:	'
while_lstm_cell_10_90705:	,
while_lstm_cell_10_90707:
˘*while/lstm_cell_10/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0˛
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_90703_0while_lstm_cell_10_90705_0while_lstm_cell_10_90707_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90620Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"6
while_lstm_cell_10_90703while_lstm_cell_10_90703_0"6
while_lstm_cell_10_90705while_lstm_cell_10_90705_0"6
while_lstm_cell_10_90707while_lstm_cell_10_90707_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
 ~
Ľ	
while_body_90878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_10_split_readvariableop_resource_0:	C
4while_lstm_cell_10_split_1_readvariableop_resource_0:	@
,while_lstm_cell_10_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_10_split_readvariableop_resource:	A
2while_lstm_cell_10_split_1_readvariableop_resource:	>
*while_lstm_cell_10_readvariableop_resource:
˘!while/lstm_cell_10/ReadVariableOp˘#while/lstm_cell_10/ReadVariableOp_1˘#while/lstm_cell_10/ReadVariableOp_2˘#while/lstm_cell_10/ReadVariableOp_3˘'while/lstm_cell_10/split/ReadVariableOp˘)while/lstm_cell_10/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0d
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ű
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitŤ
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
while/lstm_cell_10/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ń
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split˘
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0w
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ě
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_4MatMulwhile_placeholder_2)while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/MulMulwhile/lstm_cell_10/add:z:0!while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_1AddV2while/lstm_cell_10/Mul:z:0#while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
*while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?š
(while/lstm_cell_10/clip_by_value/MinimumMinimumwhile/lstm_cell_10/Add_1:z:03while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙g
"while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    š
 while/lstm_cell_10/clip_by_valueMaximum,while/lstm_cell_10/clip_by_value/Minimum:z:0+while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_5MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_1Mulwhile/lstm_cell_10/add_2:z:0#while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_3AddV2while/lstm_cell_10/Mul_1:z:0#while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_1/MinimumMinimumwhile/lstm_cell_10/Add_3:z:05while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_1Maximum.while/lstm_cell_10/clip_by_value_1/Minimum:z:0-while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_2Mul&while/lstm_cell_10/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_6MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/lstm_cell_10/ReluReluwhile/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/mul_3Mul$while/lstm_cell_10/clip_by_value:z:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/add_5AddV2while/lstm_cell_10/mul_2:z:0while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell_10/MatMul_7MatMulwhile_placeholder_2+while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
while/lstm_cell_10/add_6AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>_
while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell_10/Mul_4Mulwhile/lstm_cell_10/add_6:z:0#while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell_10/Add_7AddV2while/lstm_cell_10/Mul_4:z:0#while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˝
*while/lstm_cell_10/clip_by_value_2/MinimumMinimumwhile/lstm_cell_10/Add_7:z:05while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"while/lstm_cell_10/clip_by_value_2Maximum.while/lstm_cell_10/clip_by_value_2/Minimum:z:0-while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
while/lstm_cell_10/mul_5Mul&while/lstm_cell_10/clip_by_value_2:z:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_10/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/Identity_5Identitywhile/lstm_cell_10/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸

while/NoOpNoOp"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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

é
B__inference_lstm_10_layer_call_and_return_conditional_losses_92314
inputs_0=
*lstm_cell_10_split_readvariableop_resource:	;
,lstm_cell_10_split_1_readvariableop_resource:	8
$lstm_cell_10_readvariableop_resource:

identity˘lstm_cell_10/ReadVariableOp˘lstm_cell_10/ReadVariableOp_1˘lstm_cell_10/ReadVariableOp_2˘lstm_cell_10/ReadVariableOp_3˘!lstm_cell_10/split/ReadVariableOp˘#lstm_cell_10/split_1/ReadVariableOp˘whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0É
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ż
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_4MatMulzeros:output:0#lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_10/MulMullstm_cell_10/add:z:0lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_1AddV2lstm_cell_10/Mul:z:0lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
"lstm_cell_10/clip_by_value/MinimumMinimumlstm_cell_10/Add_1:z:0-lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    §
lstm_cell_10/clip_by_valueMaximum&lstm_cell_10/clip_by_value/Minimum:z:0%lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_5MatMulzeros:output:0%lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_1Mullstm_cell_10/add_2:z:0lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_3AddV2lstm_cell_10/Mul_1:z:0lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_cell_10/Add_3:z:0/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_1Maximum(lstm_cell_10/clip_by_value_1/Minimum:z:0'lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_2Mul lstm_cell_10/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_6MatMulzeros:output:0%lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_10/ReluRelulstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_3Mullstm_cell_10/clip_by_value:z:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_10/add_5AddV2lstm_cell_10/mul_2:z:0lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_7MatMulzeros:output:0%lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_6AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_4Mullstm_cell_10/add_6:z:0lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_7AddV2lstm_cell_10/Mul_4:z:0lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_cell_10/Add_7:z:0/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_2Maximum(lstm_cell_10/clip_by_value_2/Minimum:z:0'lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_10/Relu_1Relulstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_5Mul lstm_cell_10/clip_by_value_2:z:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_92174*
condR
while_cond_92173*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0
Â#
Ţ
while_body_90432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_10_90456_0:	)
while_lstm_cell_10_90458_0:	.
while_lstm_cell_10_90460_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_10_90456:	'
while_lstm_cell_10_90458:	,
while_lstm_cell_10_90460:
˘*while/lstm_cell_10/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0˛
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_90456_0while_lstm_cell_10_90458_0while_lstm_cell_10_90460_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90418Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"6
while_lstm_cell_10_90456while_lstm_cell_10_90456_0"6
while_lstm_cell_10_90458while_lstm_cell_10_90458_0"6
while_lstm_cell_10_90460while_lstm_cell_10_90460_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
Đ7

B__inference_lstm_10_layer_call_and_return_conditional_losses_90500

inputs%
lstm_cell_10_90419:	!
lstm_cell_10_90421:	&
lstm_cell_10_90423:

identity˘$lstm_cell_10/StatefulPartitionedCall˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskô
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_90419lstm_cell_10_90421lstm_cell_10_90423*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_90418n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_90419lstm_cell_10_90421lstm_cell_10_90423*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_90432*
condR
while_cond_90431*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ß
ç
B__inference_lstm_10_layer_call_and_return_conditional_losses_91335

inputs=
*lstm_cell_10_split_readvariableop_resource:	;
,lstm_cell_10_split_1_readvariableop_resource:	8
$lstm_cell_10_readvariableop_resource:

identity˘lstm_cell_10/ReadVariableOp˘lstm_cell_10/ReadVariableOp_1˘lstm_cell_10/ReadVariableOp_2˘lstm_cell_10/ReadVariableOp_3˘!lstm_cell_10/split/ReadVariableOp˘#lstm_cell_10/split_1/ReadVariableOp˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
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
valueB:Ń
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
B :s
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
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
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
valueB:Ű
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
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0É
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ż
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0q
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ž
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_4MatMulzeros:output:0#lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?}
lstm_cell_10/MulMullstm_cell_10/add:z:0lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_1AddV2lstm_cell_10/Mul:z:0lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
"lstm_cell_10/clip_by_value/MinimumMinimumlstm_cell_10/Add_1:z:0-lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    §
lstm_cell_10/clip_by_valueMaximum&lstm_cell_10/clip_by_value/Minimum:z:0%lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_5MatMulzeros:output:0%lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_1Mullstm_cell_10/add_2:z:0lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_3AddV2lstm_cell_10/Mul_1:z:0lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_cell_10/Add_3:z:0/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_1Maximum(lstm_cell_10/clip_by_value_1/Minimum:z:0'lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_2Mul lstm_cell_10/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_6MatMulzeros:output:0%lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell_10/ReluRelulstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_3Mullstm_cell_10/clip_by_value:z:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell_10/add_5AddV2lstm_cell_10/mul_2:z:0lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      u
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell_10/MatMul_7MatMulzeros:output:0%lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/add_6AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Y
lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_10/Mul_4Mullstm_cell_10/add_6:z:0lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/Add_7AddV2lstm_cell_10/Mul_4:z:0lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ť
$lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_cell_10/Add_7:z:0/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
lstm_cell_10/clip_by_value_2Maximum(lstm_cell_10/clip_by_value_2/Minimum:z:0'lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell_10/Relu_1Relulstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell_10/mul_5Mul lstm_cell_10/clip_by_value_2:z:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
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
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_91195*
condR
while_cond_91194*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ă
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : 2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ş
ď
#__inference_signature_wrapper_91460
lstm_10_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
identity˘StatefulPartitionedCallă
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_90294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namelstm_10_input
˘
ô
H__inference_sequential_10_layer_call_and_return_conditional_losses_92014

inputsE
2lstm_10_lstm_cell_10_split_readvariableop_resource:	C
4lstm_10_lstm_cell_10_split_1_readvariableop_resource:	@
,lstm_10_lstm_cell_10_readvariableop_resource:
:
'dense_10_matmul_readvariableop_resource:	6
(dense_10_biasadd_readvariableop_resource:
identity˘dense_10/BiasAdd/ReadVariableOp˘dense_10/MatMul/ReadVariableOp˘#lstm_10/lstm_cell_10/ReadVariableOp˘%lstm_10/lstm_cell_10/ReadVariableOp_1˘%lstm_10/lstm_cell_10/ReadVariableOp_2˘%lstm_10/lstm_cell_10/ReadVariableOp_3˘)lstm_10/lstm_cell_10/split/ReadVariableOp˘+lstm_10/lstm_cell_10/split_1/ReadVariableOp˘lstm_10/whileQ
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
::íĎe
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ů
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
::íĎg
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ř
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇg
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskf
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm_10/lstm_cell_10/split/ReadVariableOpReadVariableOp2lstm_10_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	*
dtype0á
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:01lstm_10/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_10/lstm_cell_10/MatMul_1MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_10/lstm_cell_10/MatMul_2MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
lstm_10/lstm_cell_10/MatMul_3MatMul lstm_10/strided_slice_2:output:0#lstm_10/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
&lstm_10/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm_10/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0×
lstm_10/lstm_cell_10/split_1Split/lstm_10/lstm_cell_10/split_1/split_dim:output:03lstm_10/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¨
lstm_10/lstm_cell_10/BiasAddBiasAdd%lstm_10/lstm_cell_10/MatMul:product:0%lstm_10/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
lstm_10/lstm_cell_10/BiasAdd_1BiasAdd'lstm_10/lstm_cell_10/MatMul_1:product:0%lstm_10/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
lstm_10/lstm_cell_10/BiasAdd_2BiasAdd'lstm_10/lstm_cell_10/MatMul_2:product:0%lstm_10/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
lstm_10/lstm_cell_10/BiasAdd_3BiasAdd'lstm_10/lstm_cell_10/MatMul_3:product:0%lstm_10/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#lstm_10/lstm_cell_10/ReadVariableOpReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(lstm_10/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_10/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_10/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_10/lstm_cell_10/strided_sliceStridedSlice+lstm_10/lstm_cell_10/ReadVariableOp:value:01lstm_10/lstm_cell_10/strided_slice/stack:output:03lstm_10/lstm_cell_10/strided_slice/stack_1:output:03lstm_10/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_10/lstm_cell_10/MatMul_4MatMullstm_10/zeros:output:0+lstm_10/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/BiasAdd:output:0'lstm_10/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
lstm_10/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>a
lstm_10/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_10/lstm_cell_10/MulMullstm_10/lstm_cell_10/add:z:0#lstm_10/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/Add_1AddV2lstm_10/lstm_cell_10/Mul:z:0%lstm_10/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
,lstm_10/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ż
*lstm_10/lstm_cell_10/clip_by_value/MinimumMinimumlstm_10/lstm_cell_10/Add_1:z:05lstm_10/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
$lstm_10/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ż
"lstm_10/lstm_cell_10/clip_by_valueMaximum.lstm_10/lstm_cell_10/clip_by_value/Minimum:z:0-lstm_10/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%lstm_10/lstm_cell_10/ReadVariableOp_1ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*lstm_10/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_10/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_10/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$lstm_10/lstm_cell_10/strided_slice_1StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_1:value:03lstm_10/lstm_cell_10/strided_slice_1/stack:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĄ
lstm_10/lstm_cell_10/MatMul_5MatMullstm_10/zeros:output:0-lstm_10/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
lstm_10/lstm_cell_10/add_2AddV2'lstm_10/lstm_cell_10/BiasAdd_1:output:0'lstm_10/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_10/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>a
lstm_10/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_10/lstm_cell_10/Mul_1Mullstm_10/lstm_cell_10/add_2:z:0%lstm_10/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/Add_3AddV2lstm_10/lstm_cell_10/Mul_1:z:0%lstm_10/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
.lstm_10/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
,lstm_10/lstm_cell_10/clip_by_value_1/MinimumMinimumlstm_10/lstm_cell_10/Add_3:z:07lstm_10/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_10/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ĺ
$lstm_10/lstm_cell_10/clip_by_value_1Maximum0lstm_10/lstm_cell_10/clip_by_value_1/Minimum:z:0/lstm_10/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/mul_2Mul(lstm_10/lstm_cell_10/clip_by_value_1:z:0lstm_10/zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%lstm_10/lstm_cell_10/ReadVariableOp_2ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*lstm_10/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_10/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm_10/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$lstm_10/lstm_cell_10/strided_slice_2StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_2:value:03lstm_10/lstm_cell_10/strided_slice_2/stack:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĄ
lstm_10/lstm_cell_10/MatMul_6MatMullstm_10/zeros:output:0-lstm_10/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
lstm_10/lstm_cell_10/add_4AddV2'lstm_10/lstm_cell_10/BiasAdd_2:output:0'lstm_10/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_10/lstm_cell_10/ReluRelulstm_10/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
lstm_10/lstm_cell_10/mul_3Mul&lstm_10/lstm_cell_10/clip_by_value:z:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/add_5AddV2lstm_10/lstm_cell_10/mul_2:z:0lstm_10/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%lstm_10/lstm_cell_10/ReadVariableOp_3ReadVariableOp,lstm_10_lstm_cell_10_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*lstm_10/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm_10/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_10/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ŕ
$lstm_10/lstm_cell_10/strided_slice_3StridedSlice-lstm_10/lstm_cell_10/ReadVariableOp_3:value:03lstm_10/lstm_cell_10/strided_slice_3/stack:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_1:output:05lstm_10/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĄ
lstm_10/lstm_cell_10/MatMul_7MatMullstm_10/zeros:output:0-lstm_10/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
lstm_10/lstm_cell_10/add_6AddV2'lstm_10/lstm_cell_10/BiasAdd_3:output:0'lstm_10/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
lstm_10/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>a
lstm_10/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_10/lstm_cell_10/Mul_4Mullstm_10/lstm_cell_10/add_6:z:0%lstm_10/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_10/lstm_cell_10/Add_7AddV2lstm_10/lstm_cell_10/Mul_4:z:0%lstm_10/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
.lstm_10/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ă
,lstm_10/lstm_cell_10/clip_by_value_2/MinimumMinimumlstm_10/lstm_cell_10/Add_7:z:07lstm_10/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
&lstm_10/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ĺ
$lstm_10/lstm_cell_10/clip_by_value_2Maximum0lstm_10/lstm_cell_10/clip_by_value_2/Minimum:z:0/lstm_10/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
lstm_10/lstm_cell_10/mul_5Mul(lstm_10/lstm_cell_10/clip_by_value_2:z:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙v
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Đ
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇN
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙\
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ę
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_10_lstm_cell_10_split_readvariableop_resource4lstm_10_lstm_cell_10_split_1_readvariableop_resource,lstm_10_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_10_while_body_91868*$
condR
lstm_10_while_cond_91867*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ű
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0p
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙i
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskm
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ż
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_10/MatMulMatMul lstm_10/strided_slice_3:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
IdentityIdentitydense_10/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp$^lstm_10/lstm_cell_10/ReadVariableOp&^lstm_10/lstm_cell_10/ReadVariableOp_1&^lstm_10/lstm_cell_10/ReadVariableOp_2&^lstm_10/lstm_cell_10/ReadVariableOp_3*^lstm_10/lstm_cell_10/split/ReadVariableOp,^lstm_10/lstm_cell_10/split_1/ReadVariableOp^lstm_10/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2N
%lstm_10/lstm_cell_10/ReadVariableOp_1%lstm_10/lstm_cell_10/ReadVariableOp_12N
%lstm_10/lstm_cell_10/ReadVariableOp_2%lstm_10/lstm_cell_10/ReadVariableOp_22N
%lstm_10/lstm_cell_10/ReadVariableOp_3%lstm_10/lstm_cell_10/ReadVariableOp_32J
#lstm_10/lstm_cell_10/ReadVariableOp#lstm_10/lstm_cell_10/ReadVariableOp2V
)lstm_10/lstm_cell_10/split/ReadVariableOp)lstm_10/lstm_cell_10/split/ReadVariableOp2Z
+lstm_10/lstm_cell_10/split_1/ReadVariableOp+lstm_10/lstm_cell_10/split_1/ReadVariableOp2
lstm_10/whilelstm_10/while:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
§Ż
Š
&sequential_10_lstm_10_while_body_90148H
Dsequential_10_lstm_10_while_sequential_10_lstm_10_while_loop_counterN
Jsequential_10_lstm_10_while_sequential_10_lstm_10_while_maximum_iterations+
'sequential_10_lstm_10_while_placeholder-
)sequential_10_lstm_10_while_placeholder_1-
)sequential_10_lstm_10_while_placeholder_2-
)sequential_10_lstm_10_while_placeholder_3G
Csequential_10_lstm_10_while_sequential_10_lstm_10_strided_slice_1_0
sequential_10_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_10_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_10_lstm_10_while_lstm_cell_10_split_readvariableop_resource_0:	Y
Jsequential_10_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0:	V
Bsequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource_0:
(
$sequential_10_lstm_10_while_identity*
&sequential_10_lstm_10_while_identity_1*
&sequential_10_lstm_10_while_identity_2*
&sequential_10_lstm_10_while_identity_3*
&sequential_10_lstm_10_while_identity_4*
&sequential_10_lstm_10_while_identity_5E
Asequential_10_lstm_10_while_sequential_10_lstm_10_strided_slice_1
}sequential_10_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_10_tensorarrayunstack_tensorlistfromtensorY
Fsequential_10_lstm_10_while_lstm_cell_10_split_readvariableop_resource:	W
Hsequential_10_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource:	T
@sequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource:
˘7sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp˘9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_1˘9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_2˘9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_3˘=sequential_10/lstm_10/while/lstm_cell_10/split/ReadVariableOp˘?sequential_10/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp
Msequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
?sequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_10_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_10_while_placeholderVsequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0z
8sequential_10/lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ç
=sequential_10/lstm_10/while/lstm_cell_10/split/ReadVariableOpReadVariableOpHsequential_10_lstm_10_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0
.sequential_10/lstm_10/while/lstm_cell_10/splitSplitAsequential_10/lstm_10/while/lstm_cell_10/split/split_dim:output:0Esequential_10/lstm_10/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splití
/sequential_10/lstm_10/while/lstm_cell_10/MatMulMatMulFsequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_10/lstm_10/while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ď
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_1MatMulFsequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_10/lstm_10/while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ď
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_2MatMulFsequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_10/lstm_10/while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ď
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_3MatMulFsequential_10/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:07sequential_10/lstm_10/while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
:sequential_10/lstm_10/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ç
?sequential_10/lstm_10/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOpJsequential_10_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0
0sequential_10/lstm_10/while/lstm_cell_10/split_1SplitCsequential_10/lstm_10/while/lstm_cell_10/split_1/split_dim:output:0Gsequential_10/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitä
0sequential_10/lstm_10/while/lstm_cell_10/BiasAddBiasAdd9sequential_10/lstm_10/while/lstm_cell_10/MatMul:product:09sequential_10/lstm_10/while/lstm_cell_10/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙č
2sequential_10/lstm_10/while/lstm_cell_10/BiasAdd_1BiasAdd;sequential_10/lstm_10/while/lstm_cell_10/MatMul_1:product:09sequential_10/lstm_10/while/lstm_cell_10/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙č
2sequential_10/lstm_10/while/lstm_cell_10/BiasAdd_2BiasAdd;sequential_10/lstm_10/while/lstm_cell_10/MatMul_2:product:09sequential_10/lstm_10/while/lstm_cell_10/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙č
2sequential_10/lstm_10/while/lstm_cell_10/BiasAdd_3BiasAdd;sequential_10/lstm_10/while/lstm_cell_10/MatMul_3:product:09sequential_10/lstm_10/while/lstm_cell_10/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ź
7sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOpReadVariableOpBsequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
<sequential_10/lstm_10/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
>sequential_10/lstm_10/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
>sequential_10/lstm_10/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ş
6sequential_10/lstm_10/while/lstm_cell_10/strided_sliceStridedSlice?sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp:value:0Esequential_10/lstm_10/while/lstm_cell_10/strided_slice/stack:output:0Gsequential_10/lstm_10/while/lstm_cell_10/strided_slice/stack_1:output:0Gsequential_10/lstm_10/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÚ
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_4MatMul)sequential_10_lstm_10_while_placeholder_2?sequential_10/lstm_10/while/lstm_cell_10/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
,sequential_10/lstm_10/while/lstm_cell_10/addAddV29sequential_10/lstm_10/while/lstm_cell_10/BiasAdd:output:0;sequential_10/lstm_10/while/lstm_cell_10/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
.sequential_10/lstm_10/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>u
0sequential_10/lstm_10/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ń
,sequential_10/lstm_10/while/lstm_cell_10/MulMul0sequential_10/lstm_10/while/lstm_cell_10/add:z:07sequential_10/lstm_10/while/lstm_cell_10/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙×
.sequential_10/lstm_10/while/lstm_cell_10/Add_1AddV20sequential_10/lstm_10/while/lstm_cell_10/Mul:z:09sequential_10/lstm_10/while/lstm_cell_10/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
@sequential_10/lstm_10/while/lstm_cell_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ű
>sequential_10/lstm_10/while/lstm_cell_10/clip_by_value/MinimumMinimum2sequential_10/lstm_10/while/lstm_cell_10/Add_1:z:0Isequential_10/lstm_10/while/lstm_cell_10/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙}
8sequential_10/lstm_10/while/lstm_cell_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ű
6sequential_10/lstm_10/while/lstm_cell_10/clip_by_valueMaximumBsequential_10/lstm_10/while/lstm_cell_10/clip_by_value/Minimum:z:0Asequential_10/lstm_10/while/lstm_cell_10/clip_by_value/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_1ReadVariableOpBsequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
>sequential_10/lstm_10/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential_10/lstm_10/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
@sequential_10/lstm_10/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
8sequential_10/lstm_10/while/lstm_cell_10/strided_slice_1StridedSliceAsequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_1:value:0Gsequential_10/lstm_10/while/lstm_cell_10/strided_slice_1/stack:output:0Isequential_10/lstm_10/while/lstm_cell_10/strided_slice_1/stack_1:output:0Isequential_10/lstm_10/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_5MatMul)sequential_10_lstm_10_while_placeholder_2Asequential_10/lstm_10/while/lstm_cell_10/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
.sequential_10/lstm_10/while/lstm_cell_10/add_2AddV2;sequential_10/lstm_10/while/lstm_cell_10/BiasAdd_1:output:0;sequential_10/lstm_10/while/lstm_cell_10/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
0sequential_10/lstm_10/while/lstm_cell_10/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>u
0sequential_10/lstm_10/while/lstm_cell_10/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?×
.sequential_10/lstm_10/while/lstm_cell_10/Mul_1Mul2sequential_10/lstm_10/while/lstm_cell_10/add_2:z:09sequential_10/lstm_10/while/lstm_cell_10/Const_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ů
.sequential_10/lstm_10/while/lstm_cell_10/Add_3AddV22sequential_10/lstm_10/while/lstm_cell_10/Mul_1:z:09sequential_10/lstm_10/while/lstm_cell_10/Const_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Bsequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˙
@sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1/MinimumMinimum2sequential_10/lstm_10/while/lstm_cell_10/Add_3:z:0Ksequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
8sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1MaximumDsequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1/Minimum:z:0Csequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ń
.sequential_10/lstm_10/while/lstm_cell_10/mul_2Mul<sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_1:z:0)sequential_10_lstm_10_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_2ReadVariableOpBsequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
>sequential_10/lstm_10/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential_10/lstm_10/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
@sequential_10/lstm_10/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
8sequential_10/lstm_10/while/lstm_cell_10/strided_slice_2StridedSliceAsequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_2:value:0Gsequential_10/lstm_10/while/lstm_cell_10/strided_slice_2/stack:output:0Isequential_10/lstm_10/while/lstm_cell_10/strided_slice_2/stack_1:output:0Isequential_10/lstm_10/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_6MatMul)sequential_10_lstm_10_while_placeholder_2Asequential_10/lstm_10/while/lstm_cell_10/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
.sequential_10/lstm_10/while/lstm_cell_10/add_4AddV2;sequential_10/lstm_10/while/lstm_cell_10/BiasAdd_2:output:0;sequential_10/lstm_10/while/lstm_cell_10/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_10/lstm_10/while/lstm_cell_10/ReluRelu2sequential_10/lstm_10/while/lstm_cell_10/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙á
.sequential_10/lstm_10/while/lstm_cell_10/mul_3Mul:sequential_10/lstm_10/while/lstm_cell_10/clip_by_value:z:0;sequential_10/lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ň
.sequential_10/lstm_10/while/lstm_cell_10/add_5AddV22sequential_10/lstm_10/while/lstm_cell_10/mul_2:z:02sequential_10/lstm_10/while/lstm_cell_10/mul_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ž
9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_3ReadVariableOpBsequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
>sequential_10/lstm_10/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
@sequential_10/lstm_10/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
@sequential_10/lstm_10/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
8sequential_10/lstm_10/while/lstm_cell_10/strided_slice_3StridedSliceAsequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_3:value:0Gsequential_10/lstm_10/while/lstm_cell_10/strided_slice_3/stack:output:0Isequential_10/lstm_10/while/lstm_cell_10/strided_slice_3/stack_1:output:0Isequential_10/lstm_10/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÜ
1sequential_10/lstm_10/while/lstm_cell_10/MatMul_7MatMul)sequential_10_lstm_10_while_placeholder_2Asequential_10/lstm_10/while/lstm_cell_10/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ä
.sequential_10/lstm_10/while/lstm_cell_10/add_6AddV2;sequential_10/lstm_10/while/lstm_cell_10/BiasAdd_3:output:0;sequential_10/lstm_10/while/lstm_cell_10/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
0sequential_10/lstm_10/while/lstm_cell_10/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>u
0sequential_10/lstm_10/while/lstm_cell_10/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?×
.sequential_10/lstm_10/while/lstm_cell_10/Mul_4Mul2sequential_10/lstm_10/while/lstm_cell_10/add_6:z:09sequential_10/lstm_10/while/lstm_cell_10/Const_4:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ů
.sequential_10/lstm_10/while/lstm_cell_10/Add_7AddV22sequential_10/lstm_10/while/lstm_cell_10/Mul_4:z:09sequential_10/lstm_10/while/lstm_cell_10/Const_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Bsequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?˙
@sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2/MinimumMinimum2sequential_10/lstm_10/while/lstm_cell_10/Add_7:z:0Ksequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
8sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2MaximumDsequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2/Minimum:z:0Csequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
/sequential_10/lstm_10/while/lstm_cell_10/Relu_1Relu2sequential_10/lstm_10/while/lstm_cell_10/add_5:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ĺ
.sequential_10/lstm_10/while/lstm_cell_10/mul_5Mul<sequential_10/lstm_10/while/lstm_cell_10/clip_by_value_2:z:0=sequential_10/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
@sequential_10/lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_10_while_placeholder_1'sequential_10_lstm_10_while_placeholder2sequential_10/lstm_10/while/lstm_cell_10/mul_5:z:0*
_output_shapes
: *
element_dtype0:éčŇc
!sequential_10/lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_10/lstm_10/while/addAddV2'sequential_10_lstm_10_while_placeholder*sequential_10/lstm_10/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_10/lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ż
!sequential_10/lstm_10/while/add_1AddV2Dsequential_10_lstm_10_while_sequential_10_lstm_10_while_loop_counter,sequential_10/lstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 
$sequential_10/lstm_10/while/IdentityIdentity%sequential_10/lstm_10/while/add_1:z:0!^sequential_10/lstm_10/while/NoOp*
T0*
_output_shapes
: Â
&sequential_10/lstm_10/while/Identity_1IdentityJsequential_10_lstm_10_while_sequential_10_lstm_10_while_maximum_iterations!^sequential_10/lstm_10/while/NoOp*
T0*
_output_shapes
: 
&sequential_10/lstm_10/while/Identity_2Identity#sequential_10/lstm_10/while/add:z:0!^sequential_10/lstm_10/while/NoOp*
T0*
_output_shapes
: Č
&sequential_10/lstm_10/while/Identity_3IdentityPsequential_10/lstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_10/while/NoOp*
T0*
_output_shapes
: ź
&sequential_10/lstm_10/while/Identity_4Identity2sequential_10/lstm_10/while/lstm_cell_10/mul_5:z:0!^sequential_10/lstm_10/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ź
&sequential_10/lstm_10/while/Identity_5Identity2sequential_10/lstm_10/while/lstm_cell_10/add_5:z:0!^sequential_10/lstm_10/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ň
 sequential_10/lstm_10/while/NoOpNoOp8^sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp:^sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_1:^sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_2:^sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_3>^sequential_10/lstm_10/while/lstm_cell_10/split/ReadVariableOp@^sequential_10/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Y
&sequential_10_lstm_10_while_identity_1/sequential_10/lstm_10/while/Identity_1:output:0"Y
&sequential_10_lstm_10_while_identity_2/sequential_10/lstm_10/while/Identity_2:output:0"Y
&sequential_10_lstm_10_while_identity_3/sequential_10/lstm_10/while/Identity_3:output:0"Y
&sequential_10_lstm_10_while_identity_4/sequential_10/lstm_10/while/Identity_4:output:0"Y
&sequential_10_lstm_10_while_identity_5/sequential_10/lstm_10/while/Identity_5:output:0"U
$sequential_10_lstm_10_while_identity-sequential_10/lstm_10/while/Identity:output:0"
@sequential_10_lstm_10_while_lstm_cell_10_readvariableop_resourceBsequential_10_lstm_10_while_lstm_cell_10_readvariableop_resource_0"
Hsequential_10_lstm_10_while_lstm_cell_10_split_1_readvariableop_resourceJsequential_10_lstm_10_while_lstm_cell_10_split_1_readvariableop_resource_0"
Fsequential_10_lstm_10_while_lstm_cell_10_split_readvariableop_resourceHsequential_10_lstm_10_while_lstm_cell_10_split_readvariableop_resource_0"
Asequential_10_lstm_10_while_sequential_10_lstm_10_strided_slice_1Csequential_10_lstm_10_while_sequential_10_lstm_10_strided_slice_1_0"
}sequential_10_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_10_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2v
9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_19sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_12v
9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_29sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_22v
9sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_39sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp_32r
7sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp7sequential_10/lstm_10/while/lstm_cell_10/ReadVariableOp2~
=sequential_10/lstm_10/while/lstm_cell_10/split/ReadVariableOp=sequential_10/lstm_10/while/lstm_cell_10/split/ReadVariableOp2
?sequential_10/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp?sequential_10/lstm_10/while/lstm_cell_10/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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
_user_specified_name0.sequential_10/lstm_10/while/maximum_iterations:` \

_output_shapes
: 
B
_user_specified_name*(sequential_10/lstm_10/while/loop_counter
×
ň
-__inference_sequential_10_layer_call_fn_91490

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91377o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:˙˙˙˙˙˙˙˙˙: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă

(__inference_dense_10_layer_call_fn_93091

inputs
unknown:	
	unknown_0:
identity˘StatefulPartitionedCallŘ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_91036o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"ó
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ť
serving_default§
K
lstm_10_input:
serving_default_lstm_10_input:0˙˙˙˙˙˙˙˙˙<
dense_100
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:ť
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
Ă
trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__
cell

state_spec"
_tf_keras_rnn_layer
ť
trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__

kernel
bias"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
Ę
layer_regularization_losses
trainable_variables
 layer_metrics
!metrics

"layers
regularization_losses
#non_trainable_variables
	variables
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ë
$trace_0
%trace_1
&trace_2
'trace_32ŕ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91752
H__inference_sequential_10_layer_call_and_return_conditional_losses_92014
H__inference_sequential_10_layer_call_and_return_conditional_losses_91421
H__inference_sequential_10_layer_call_and_return_conditional_losses_91437ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z$trace_0z%trace_1z&trace_2z'trace_3
ß
(trace_0
)trace_1
*trace_2
+trace_32ô
-__inference_sequential_10_layer_call_fn_91056
-__inference_sequential_10_layer_call_fn_91475
-__inference_sequential_10_layer_call_fn_91490
-__inference_sequential_10_layer_call_fn_91405ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z(trace_0z)trace_1z*trace_2z+trace_3

,trace_02ë
 __inference__wrapped_model_90294Ć
˛
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *0˘-
+(
lstm_10_input˙˙˙˙˙˙˙˙˙z,trace_0
ť

-beta_1

.beta_2
	/decay
0learning_rate
1iterm]m^m_m`mavbvcvdvevf"
tf_deprecated_optimizer
,
2serving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
š
3layer_regularization_losses
trainable_variables
4non_trainable_variables
5layer_metrics
6metrics

7layers
regularization_losses

8states
	variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Č
9trace_0
:trace_1
;trace_2
<trace_32Ý
B__inference_lstm_10_layer_call_and_return_conditional_losses_92314
B__inference_lstm_10_layer_call_and_return_conditional_losses_92570
B__inference_lstm_10_layer_call_and_return_conditional_losses_92826
B__inference_lstm_10_layer_call_and_return_conditional_losses_93082Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z9trace_0z:trace_1z;trace_2z<trace_3
Ü
=trace_0
>trace_1
?trace_2
@trace_32ń
'__inference_lstm_10_layer_call_fn_92025
'__inference_lstm_10_layer_call_fn_92036
'__inference_lstm_10_layer_call_fn_92047
'__inference_lstm_10_layer_call_fn_92058Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z=trace_0z>trace_1z?trace_2z@trace_3
á
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
*E&call_and_return_all_conditional_losses
F__call__
G
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Hlayer_regularization_losses
trainable_variables
Ilayer_metrics
Jmetrics

Klayers
regularization_losses
Lnon_trainable_variables
	variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ý
Mtrace_02ŕ
C__inference_dense_10_layer_call_and_return_conditional_losses_93101
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zMtrace_0
â
Ntrace_02Ĺ
(__inference_dense_10_layer_call_fn_93091
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zNtrace_0
": 	2dense_10/kernel
:2dense_10/bias
.:,	2lstm_10/lstm_cell_10/kernel
9:7
2%lstm_10/lstm_cell_10/recurrent_kernel
(:&2lstm_10/lstm_cell_10/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
O0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
B
H__inference_sequential_10_layer_call_and_return_conditional_losses_91752inputs"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
H__inference_sequential_10_layer_call_and_return_conditional_losses_92014inputs"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
H__inference_sequential_10_layer_call_and_return_conditional_losses_91421lstm_10_input"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
H__inference_sequential_10_layer_call_and_return_conditional_losses_91437lstm_10_input"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
-__inference_sequential_10_layer_call_fn_91056lstm_10_input"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ôBń
-__inference_sequential_10_layer_call_fn_91475inputs"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ôBń
-__inference_sequential_10_layer_call_fn_91490inputs"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
-__inference_sequential_10_layer_call_fn_91405lstm_10_input"ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙Bü
 __inference__wrapped_model_90294lstm_10_input"Ć
˛
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *0˘-
+(
lstm_10_input˙˙˙˙˙˙˙˙˙
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
ĐBÍ
#__inference_signature_wrapper_91460lstm_10_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 B
B__inference_lstm_10_layer_call_and_return_conditional_losses_92314inputs_0"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
B__inference_lstm_10_layer_call_and_return_conditional_losses_92570inputs_0"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
B__inference_lstm_10_layer_call_and_return_conditional_losses_92826inputs"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
B__inference_lstm_10_layer_call_and_return_conditional_losses_93082inputs"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
'__inference_lstm_10_layer_call_fn_92025inputs_0"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
'__inference_lstm_10_layer_call_fn_92036inputs_0"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
'__inference_lstm_10_layer_call_fn_92047inputs"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
'__inference_lstm_10_layer_call_fn_92058inputs"Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
­
Player_regularization_losses
Atrainable_variables
Qlayer_metrics
Rmetrics

Slayers
Bregularization_losses
Tnon_trainable_variables
C	variables
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
˙
Utrace_0
Vtrace_12Č
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93224
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93313ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zUtrace_0zVtrace_1
É
Wtrace_0
Xtrace_12
,__inference_lstm_cell_10_layer_call_fn_93118
,__inference_lstm_cell_10_layer_call_fn_93135ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zWtrace_0zXtrace_1
 "
trackable_list_wrapper
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
íBę
C__inference_dense_10_layer_call_and_return_conditional_losses_93101inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŇBĎ
(__inference_dense_10_layer_call_fn_93091inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
N
Y	variables
Z	keras_api
	[total
	\count"
_tf_keras_metric
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
 B
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93224inputsstates_0states_1"ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93313inputsstates_0states_1"ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
,__inference_lstm_cell_10_layer_call_fn_93118inputsstates_0states_1"ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
,__inference_lstm_cell_10_layer_call_fn_93135inputsstates_0states_1"ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
[0
\1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
':%	2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
3:1	2"Adam/lstm_10/lstm_cell_10/kernel/m
>:<
2,Adam/lstm_10/lstm_cell_10/recurrent_kernel/m
-:+2 Adam/lstm_10/lstm_cell_10/bias/m
':%	2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
3:1	2"Adam/lstm_10/lstm_cell_10/kernel/v
>:<
2,Adam/lstm_10/lstm_cell_10/recurrent_kernel/v
-:+2 Adam/lstm_10/lstm_cell_10/bias/v
 __inference__wrapped_model_90294x:˘7
0˘-
+(
lstm_10_input˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
dense_10"
dense_10˙˙˙˙˙˙˙˙˙Ť
C__inference_dense_10_layer_call_and_return_conditional_losses_93101d0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
(__inference_dense_10_layer_call_fn_93091Y0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "!
unknown˙˙˙˙˙˙˙˙˙Ě
B__inference_lstm_10_layer_call_and_return_conditional_losses_92314O˘L
E˘B
41
/,
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "-˘*
# 
tensor_0˙˙˙˙˙˙˙˙˙
 Ě
B__inference_lstm_10_layer_call_and_return_conditional_losses_92570O˘L
E˘B
41
/,
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "-˘*
# 
tensor_0˙˙˙˙˙˙˙˙˙
 ť
B__inference_lstm_10_layer_call_and_return_conditional_losses_92826u?˘<
5˘2
$!
inputs˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "-˘*
# 
tensor_0˙˙˙˙˙˙˙˙˙
 ť
B__inference_lstm_10_layer_call_and_return_conditional_losses_93082u?˘<
5˘2
$!
inputs˙˙˙˙˙˙˙˙˙

 
p

 
Ş "-˘*
# 
tensor_0˙˙˙˙˙˙˙˙˙
 Ľ
'__inference_lstm_10_layer_call_fn_92025zO˘L
E˘B
41
/,
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş ""
unknown˙˙˙˙˙˙˙˙˙Ľ
'__inference_lstm_10_layer_call_fn_92036zO˘L
E˘B
41
/,
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş ""
unknown˙˙˙˙˙˙˙˙˙
'__inference_lstm_10_layer_call_fn_92047j?˘<
5˘2
$!
inputs˙˙˙˙˙˙˙˙˙

 
p 

 
Ş ""
unknown˙˙˙˙˙˙˙˙˙
'__inference_lstm_10_layer_call_fn_92058j?˘<
5˘2
$!
inputs˙˙˙˙˙˙˙˙˙

 
p

 
Ş ""
unknown˙˙˙˙˙˙˙˙˙ć
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93224˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p 
Ş "˘
˘~
%"

tensor_0_0˙˙˙˙˙˙˙˙˙
UR
'$
tensor_0_1_0˙˙˙˙˙˙˙˙˙
'$
tensor_0_1_1˙˙˙˙˙˙˙˙˙
 ć
G__inference_lstm_cell_10_layer_call_and_return_conditional_losses_93313˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p
Ş "˘
˘~
%"

tensor_0_0˙˙˙˙˙˙˙˙˙
UR
'$
tensor_0_1_0˙˙˙˙˙˙˙˙˙
'$
tensor_0_1_1˙˙˙˙˙˙˙˙˙
 ¸
,__inference_lstm_cell_10_layer_call_fn_93118˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p 
Ş "{˘x
# 
tensor_0˙˙˙˙˙˙˙˙˙
QN
%"

tensor_1_0˙˙˙˙˙˙˙˙˙
%"

tensor_1_1˙˙˙˙˙˙˙˙˙¸
,__inference_lstm_cell_10_layer_call_fn_93135˘
x˘u
 
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p
Ş "{˘x
# 
tensor_0˙˙˙˙˙˙˙˙˙
QN
%"

tensor_1_0˙˙˙˙˙˙˙˙˙
%"

tensor_1_1˙˙˙˙˙˙˙˙˙Ĺ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91421yB˘?
8˘5
+(
lstm_10_input˙˙˙˙˙˙˙˙˙
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ĺ
H__inference_sequential_10_layer_call_and_return_conditional_losses_91437yB˘?
8˘5
+(
lstm_10_input˙˙˙˙˙˙˙˙˙
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 ž
H__inference_sequential_10_layer_call_and_return_conditional_losses_91752r;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 ž
H__inference_sequential_10_layer_call_and_return_conditional_losses_92014r;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 
-__inference_sequential_10_layer_call_fn_91056nB˘?
8˘5
+(
lstm_10_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
-__inference_sequential_10_layer_call_fn_91405nB˘?
8˘5
+(
lstm_10_input˙˙˙˙˙˙˙˙˙
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
-__inference_sequential_10_layer_call_fn_91475g;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
-__inference_sequential_10_layer_call_fn_91490g;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "!
unknown˙˙˙˙˙˙˙˙˙ą
#__inference_signature_wrapper_91460K˘H
˘ 
AŞ>
<
lstm_10_input+(
lstm_10_input˙˙˙˙˙˙˙˙˙"3Ş0
.
dense_10"
dense_10˙˙˙˙˙˙˙˙˙