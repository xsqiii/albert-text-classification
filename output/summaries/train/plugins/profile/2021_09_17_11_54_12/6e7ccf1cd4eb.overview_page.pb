?  *㥛???@?l?{9?(A)      ?=2m
5Iterator::Model::Prefetch::BatchV2::Shuffle::Map::Map?0g???5?@!\?}??JU@)fl?f_e?@1??-???T@:Preprocessing2Z
"Iterator::Model::Prefetch::BatchV2?J?i???@!N?s6?X@)G?P?}W@1??+?4?&@:Preprocessing2?
KIterator::Model::Prefetch::BatchV2::Shuffle::Map::Map::FlatMap[0]::TFRecord?0W????{1@!??0o? @)W????{1@1??0o? @:Advanced file read2c
+Iterator::Model::Prefetch::BatchV2::Shuffle?0@?ϝ $?@!?j}??.V@)?5???0@1???d&, @:Preprocessing2h
0Iterator::Model::Prefetch::BatchV2::Shuffle::Map?0????)??@!?|W???U@)=
ףp?)@1Q]y6????:Preprocessing2v
>Iterator::Model::Prefetch::BatchV2::Shuffle::Map::Map::FlatMap?0)???^:@!??cbI?@)???o'!@1???b?q??:Preprocessing2F
Iterator::ModelsI?v??!???hN?~?)?N??C??1g=?:|?:Preprocessing2P
Iterator::Model::Prefetchs?,&6w?!??-rA*F?)s?,&6w?1??-rA*F?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@qfD?v?r?"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.G6e7ccf1cd4eb: Failed to load libcupti (is it installed and accessible?)