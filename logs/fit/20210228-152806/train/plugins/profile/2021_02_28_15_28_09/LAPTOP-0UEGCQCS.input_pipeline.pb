	�HP�H7@�HP�H7@!�HP�H7@	͇�C��?͇�C��?!͇�C��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�HP�H7@�'��@A��n��4@Y�5�;N��?*	����̄r@2F
Iterator::Model�C�����?!9���gG@)J{�/L��?1�IST�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�,C��?!�5>,>@)vq�-�?1�VhiT5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��MbX�?!<J��0@)��d�`T�?1p�)*(@:Preprocessing2U
Iterator::Model::ParallelMapV2e�X��?!���%�Z'@)e�X��?1���%�Z'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��<,Ԛ?!�k1�M�!@)��<,Ԛ?1�k1�M�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���<,�?!��6"<�J@)���&�?1ع��>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�?�߾�?!.I+�~@)�?�߾�?1.I+�~@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 10.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9̇�C��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�'��@�'��@!�'��@      ��!       "      ��!       *      ��!       2	��n��4@��n��4@!��n��4@:      ��!       B      ��!       J	�5�;N��?�5�;N��?!�5�;N��?R      ��!       Z	�5�;N��?�5�;N��?!�5�;N��?JCPU_ONLYẎ�C��?b 