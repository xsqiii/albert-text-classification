	?$????@?$????@!?$????@	????[???????[???!????[???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?$????@j?t???A9??v?ʚ@Y/?$??@*     ?B@     z?@2F
Iterator::Model?Zd;_@!??և?T@)-????F@1???ΒT@:Preprocessing2l
5Iterator::Model::Prefetch::BatchV2::Shuffle::Map::Map	%??C???!I???-@)?A`??"??1??W<?'@:Preprocessing2u
>Iterator::Model::Prefetch::BatchV2::Shuffle::Map::Map::FlatMap ?"??~j??!c??b?@)?5^?I??1~^r???@:Preprocessing2?
KIterator::Model::Prefetch::BatchV2::Shuffle::Map::Map::FlatMap[0]::TFRecord	?l??????!#qU8????)?l??????1#qU8????:Advanced file read2b
+Iterator::Model::Prefetch::BatchV2::Shuffle	??Q????!?q??.@)L7?A`???1?)?T???:Preprocessing2g
0Iterator::Model::Prefetch::BatchV2::Shuffle::Map	ˡE?????!-?oo?	.@)y?&1???1?<???W??:Preprocessing2P
Iterator::Model::Prefetch?~j?t???!P?????)?~j?t???1P?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????[???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	j?t???j?t???!j?t???      ??!       "      ??!       *      ??!       2	9??v?ʚ@9??v?ʚ@!9??v?ʚ@:      ??!       B      ??!       J	/?$??@/?$??@!/?$??@R      ??!       Z	/?$??@/?$??@!/?$??@JCPU_ONLYY????[???b 