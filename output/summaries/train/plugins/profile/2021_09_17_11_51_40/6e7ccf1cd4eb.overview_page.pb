?  *F??????@bXy?(A2m
5Iterator::Model::Prefetch::BatchV2::Shuffle::Map::Map?,??d?&?@!vJ?wV@)ퟧ?e?@1;?G:?BU@:Preprocessing2Z
"Iterator::Model::Prefetch::BatchV2?3nj??@!\??F??X@)u?Rz?HQ@1K??OT,!@:Preprocessing2?
KIterator::Model::Prefetch::BatchV2::Shuffle::Map::Map::FlatMap[0]::TFRecord?,?ᱟ?21@!???N?@)?ᱟ?21@1???N?@:Advanced file read2c
+Iterator::Model::Prefetch::BatchV2::Shuffle?,V?1???@!??g?!?V@)?e??SI0@1d?Q?. @:Preprocessing2h
0Iterator::Model::Prefetch::BatchV2::Shuffle::Map?,)????}?@!5?`?XV@)#e??ݰ%@1?o??R???:Preprocessing2v
>Iterator::Model::Prefetch::BatchV2::Shuffle::Map::Map::FlatMap?,\??J? 8@!f?B?@)6????@1%?a!????:Preprocessing2F
Iterator::Model??.ޏ??!a??]?os?);S???.??1,A???q?:Preprocessing2P
Iterator::Model::Prefetch?fd??s?!??E-??B?)?fd??s?1??E-??B?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@qea?_#{r?"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.G6e7ccf1cd4eb: Failed to load libcupti (is it installed and accessible?)