	�a��4�@�a��4�@!�a��4�@	�S�r
�%@�S�r
�%@!�S�r
�%@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�a��4�@�J�4�?1H�z���?Ib���L�@Yc� ��^�?r0*	�����lY@2T
Iterator::Root::ParallelMapV2�Q���?!%���5A@)�Q���?1%���5A@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat2�%䃞?!�)�<GM=@)�?Ɯ?1P�$^Z�;@:Preprocessing2E
Iterator::Root��ͪ�զ?!L+�f_�E@)a2U0*��?1�.�	�"@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�St$���?!]~�|�R @)�St$���?1]~�|�R @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�46<�?!����L@)-C��6z?1'�,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate%u��?!q�N��,@)-C��6z?1'�,@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap��A�f�?!z����4@)a��+ey?1v�0��b@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!��V�;�?)_�Q�[?1��V�;�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 10.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�64.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�S�r
�%@I>��t0P@Q<�\`*g8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�J�4�?�J�4�?!�J�4�?      ��!       "	H�z���?H�z���?!H�z���?*      ��!       2      ��!       :	b���L�@b���L�@!b���L�@B      ��!       J	c� ��^�?c� ��^�?!c� ��^�?R      ��!       Z	c� ��^�?c� ��^�?!c� ��^�?b      ��!       JGPUY�S�r
�%@b q>��t0P@y<�\`*g8@