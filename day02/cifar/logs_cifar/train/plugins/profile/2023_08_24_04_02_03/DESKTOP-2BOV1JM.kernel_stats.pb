
g
&ampere_scudnn_128x64_relu_medium_nn_v1���*�2+8��@��H��bsequential/conv2d_1/ReluhuMUB
}
ampere_sgemm_128x128_ntv��*�2$8��@��H��Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhuMUB
�
g_Z17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEviiiPKT_iPS0_S2_18kernel_grad_paramsyifiiiiP�*2@8��@��H��Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  HB
f
&ampere_scudnn_128x64_relu_medium_nn_v1���*�2�8�@�H�bsequential/conv2d/ReluhuMUB
�
3ampere_scudnn_128x64_stridedB_splitK_interior_nn_v1���*�28��@��H��Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  �A
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*�2-8��@��H��b(gradient_tape/sequential/conv2d/ReluGradhuZU�B
�
�_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( �*�2 8��@��H��b sequential/max_pooling2d/MaxPoolhu���B
�
�_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_smallIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_ �8*�2  8��@��H��b:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradhu  �B
W
ampere_sgemm_128x128_nnv��*�2$8��@��H��bsequential/conv2d_2/ReluhuMUB
�
Z_ZN5cudnn17winograd_nonfused22winogradForwardData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@��*�2@8��@��H��Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhuZU�B
}
ampere_sgemm_128x128_ntv��*�2$8��@��H��Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhuMUB
�
�_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_smallIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_ �
*�2 @8��@��H��b<gradient_tape/sequential/max_pooling2d_1/MaxPool/MaxPoolGradhu  �B
|
ampere_cgemm_64x32_tn^�h*�2�8��@��H��Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu��&B
o
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*�28��@��H��bsequential/dense_1/BiasAddhuZU�B
L
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bAdam/PowhuZU�B
�
+_Z15fft2d_r2c_16x16IfEvP6float2PKT_iiiiiiii( �L*�2@8�@�H�Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  �B
�
9_Z15fft2d_c2r_16x16IfLb0EEvPT_P6float2iiiiiiiiiiffiS1_S1_( �D*�2�8�o@�oH�oXb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS_9IndexListINS_10type2indexILx1EEEJEEEKNS_17TensorGeneratorOpIN10tensorflow9generator23SparseXentLossGeneratorIfxEEKNS4_INS5_IfLi2ELi1EiEELi16ES7_EEEES7_EEEENS_9GpuDeviceEEEiEEvT_T0_2*�28�l@�lH�lbgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshuZU�B
�
+_Z15fft2d_r2c_16x16IfEvP6float2PKT_iiiiiiii( �L*�2 8�f@�fH�fXb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  �B
�
Z_ZN5cudnn17winograd_nonfused22winogradForwardData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@��*�2@8�f@�fH�fbsequential/conv2d_2/ReluhuZU�B

 _Z11flip_filterIffEvPT0_PKT_iiii*2 @8�`@�`H�`Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu�
�B
�
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*�2�8�X@�XH�Xb3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  �B
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �!*�2@8�V@�VH�Vb@sequential/conv2d_2/Relu-0-2-TransposeNCHWToNHWC-LayoutOptimizerhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*�2-8�R@�RH�Rb*gradient_tape/sequential/conv2d_1/ReluGradhuZU�B
�
^_ZN5cudnn17winograd_nonfused24winogradForwardOutput4x4IffEEvNS0_20WinogradOutputParamsIT_T0_EE0��*�2 8�P@�PH�PXb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu���B
�
v_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi1024ELi1024ELi2ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �`*�2@8�K@�KH�Kbagradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhuZU�B
�
^_ZN5cudnn17winograd_nonfused24winogradForwardOutput4x4IffEEvNS0_20WinogradOutputParamsIT_T0_EE0��*�2@8�H@�HH�HXb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu���B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�C@�CH�Cb$Adam/Adam/update_4/ResourceApplyAdamhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�C@�CH�Cb$Adam/Adam/update_6/ResourceApplyAdamhuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1EEvNT_6ParamsEp ��*�28�A@�AH�AXbsequential/dense_1/MatMulhugU�A
�
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*�2�8�@@�@H�@b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhu  �B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4EEvNT_6ParamsE^ ��*�28�;@�;H�;Xbsequential/dense/MatMulhugU�A
�
Z_ZN5cudnn17winograd_nonfused22winogradForwardData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@��*�2@8�9@�9H�9Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align1EEvNT_6ParamsEp ��*�28�7@�7H�7b)gradient_tape/sequential/dense_1/MatMul_1hugU�A
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS_9IndexListINS_10type2indexILx1EEEJEEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKNS4_INS5_IfLi2ELi1EiEELi16ES7_EEEES7_EEEENS_9GpuDeviceEEEiEEvT_T0_-*�28�5@�5H�5bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align1EEvNT_6ParamsEv ��*�28�3@�3H�3Xb'gradient_tape/sequential/dense_1/MatMulhugU�A
�
<_ZN10tensorflow26BiasGradNCHW_SharedAtomicsIfEEvPKT_PS1_iiii�*�2@8�2@�2H�2b5gradient_tape/sequential/conv2d_2/BiasAdd/BiasAddGradhuZU�B
�
^_ZN5cudnn17winograd_nonfused24winogradForwardOutput4x4IffEEvNS0_20WinogradOutputParamsIT_T0_EE0��*�2@8�2@�2H�2bsequential/conv2d_2/Reluhu���B
�
�_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( �*H2 8�2@�2H�2b"sequential/max_pooling2d_1/MaxPoolhu ��B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�/@�/H�/b$Adam/Adam/update_2/ResourceApplyAdamhuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4EEvNT_6ParamsE^ ��*�28�.@�.H�.Xb%gradient_tape/sequential/dense/MatMulhugU�A
�
V_ZN7cutlass6KernelI51cutlass_80_tensorop_s1688gemm_64x64_16x10_nt_align4EEvNT_6ParamsE` ��*�28�-@�-H�-b'gradient_tape/sequential/dense/MatMul_1h
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�+@�+H�+Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�(@�(H�(Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�(@�(H�(bsequential/conv2d_2/ReluhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�$@�$H�$b"Adam/Adam/update/ResourceApplyAdamhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�$@�$H�$b$Adam/Adam/update_9/ResourceApplyAdamhuZU�B
�
^_ZN5cudnn17winograd_nonfused24winogradForwardFilter4x4IffEEvNS0_20WinogradFilterParamsIT_T0_EE(�H* 28�#@�#H�#bsequential/conv2d_2/Reluhu  �B
�
^_ZN5cudnn17winograd_nonfused24winogradForwardFilter4x4IffEEvNS0_20WinogradFilterParamsIT_T0_EE(�H* 28�#@�#H�#Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  �B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�"@�"H�"b$Adam/Adam/update_5/ResourceApplyAdamhuZU�B
�
^_ZN5cudnn17winograd_nonfused24winogradForwardFilter4x4IffEEvNS0_20WinogradFilterParamsIT_T0_EE(�H* 28�"@�"H�"Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  �B
�
:_ZN10tensorflow26BiasGradNHWC_SharedAtomicsIfEEviPKT_PS1_i (*�28�!@�!H�!b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�!@�!H�!Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�!@�!H�!b$Adam/Adam/update_3/ResourceApplyAdamhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�!@�!H�!b$Adam/Adam/update_8/ResourceApplyAdamhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�!@�!H�!bsequential/conv2d_1/ReluhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28� @� H� Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�@�H�b$Adam/Adam/update_7/ResourceApplyAdamhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_20TensorTupleReducerOpINS0_18ArgMaxTupleReducerINS_5TupleIxfEEEEKNS_5arrayIxLy1EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_ *�28�@�H�bArgMaxhuZU�B
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b *�28�@�H�b$Adam/Adam/update_1/ResourceApplyAdamhuZU�B
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �!*�2@8�@�H�bPgradient_tape/sequential/conv2d_2/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  �B
�
b_Z19splitKreduce_kernelIffffLb1ELb0EEv18cublasSplitKParamsIT1_EPKT_PKT0_PS6_PKS1_SB_PKT2_PvxPS1_Pi * 28�@�H�Xbsequential/dense/MatMulhu  �B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�@�H�Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*�28�@�H�bAdam/addhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�@�H�bsequential/conv2d/ReluhuZU�B
�
:_ZN10tensorflow26BiasGradNHWC_SharedAtomicsIfEEviPKT_PS1_i �*�28�@�H�b2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bLgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*�2 8�@�H�b*gradient_tape/sequential/conv2d_2/ReluGradhuZU�B
�
K_ZN10cask_cudnn20computeOffsetsKernelILb0ELb0EEEvNS_20ComputeOffsetsParamsE*�28�@�H�bsequential/conv2d/Reluhu  �B
�
Q_ZN10cask_cudnn31computeWgradSplitKOffsetsKernelENS_26ComputeSplitKOffsetsParamsE*�28�@�H�Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bsequential/dense/ReluhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKhLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�b`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_CasthuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIKfSB_EEKNS4_INS5_ISB_Li2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_9IndexListINS_10type2indexILx1EEEJiEEEKNS_17TensorReshapingOpIKNSH_IiJSJ_EEENS4_INS5_IfLi1ELi1EiEELi16ES7_EEEEEEEEEENS_9GpuDeviceEEEiEEvT_T0_*�28�@�H�bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshuZU�B
�
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE�!*  28�@�H�b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bAdam/Adam/AssignAddVariableOphuZU�B
�
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE�!*  28�@�H�b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhuZU�B
G
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�28�@�H�bCast_3hu  �B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b
Adam/Pow_1huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_17TensorGeneratorOpIN10tensorflow9generator23SparseXentGradGeneratorIfxEEKS8_EEEENS_9GpuDeviceEEEiEEvT_T0_*�28�@�H�bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bAssignAddVariableOp_4huZU�B
�
K_ZN10cask_cudnn20computeOffsetsKernelILb0ELb0EEEvNS_20ComputeOffsetsParamsE*�28�@�H�bsequential/conv2d_1/Reluhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy1EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*�28�@�H�bBgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1huZU�B
j
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*�28�@�H�bsequential/dense/BiasAddhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�b
div_no_nanhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bdiv_no_nan_1huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�b3sparse_categorical_crossentropy/weighted_loss/valuehuZU�B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*�28�@�H�bEqualhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bAssignAddVariableOphuZU�B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28�@�H�b
LogicalAndhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bAssignAddVariableOp_2huZU�B
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*�28�@�H�bSum_2hu  �B
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�@�H�bAdam/Cast_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bUgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulhuZU�B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bMulhuZU�B
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*�28�@�H�b1sparse_categorical_crossentropy/weighted_loss/Sumhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bAssignAddVariableOp_1huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bAssignAddVariableOp_3huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�b'gradient_tape/sequential/dense/ReluGradhuZU�B
�
L_ZN10cask_cudnn26computeWgradBOffsetsKernelENS_26ComputeWgradBOffsetsParamsE*�28�@�H�Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  �B
�
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3MaxEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE *�28�@�H�bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKhLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�b$sparse_categorical_crossentropy/CasthuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKS8_EEEENS_9GpuDeviceEEEiEEvT_T0_*�28�@�H�bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshuZU�B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�bCasthu  �B
�
!Cast_GPU_DT_FLOAT_DT_INT64_kernel*�28�@�H�bbsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1hu  �B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�bCast_4hu  �B
H
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�@�H�bCast_2hu  �B
�
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�@�H�b?sparse_categorical_crossentropy/weighted_loss/num_elements/Casthu  �B