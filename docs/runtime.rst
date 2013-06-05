.. _runtime:

======================
Scout Runtime Settings
======================


*  .. envvar:: SC_RUNTIME_NTHREADS 

  Controls the number of threads executed when the -mt (Cpu multithreading) 
  option is used. By default it will use all availaible processing units.

* .. envvar:: SC_RUNTIME_HT

  Controls if hyperthreading is used. Default is enabled (SC_RUNTIME_HT=1)
  If enable will use one thread per processing unit, if disabled will use
  one thread per cpu core.

* .. envvar:: SC_RUNTIME_NUMA

  EXPERIMENTAL: Control if numa support is used. Default is disabled
  (SC_RUNTIME_NUMA=0)

* .. envvar:: SC_RUNTIME_BPT

  ADVANCED: Set how many Blocks will be send to each thread, Default 4.

* .. envvar:: SC_RUNTIME_DEBUG

  ADVANCED: Enable printing of runtime debug messages. 

* .. envvar:: SC_RUNTIME_NDOMAINS

  EXPERIMENTAL: Set number of NUMA doamins to use, not fully fuctional.

* .. envvar:: SC_RUNTIME_THREADBIND

  EXPERIMENTAL/ADVANCED: method to bind threads to NUMA domains. Default is no
  thread binding (SC_RUNTIME_THREADBIND=0). If set to 1 bindThreadInside()
  is used, if set to 2 bindThreadOutside() is used.

* .. envvar:: SC_RUNTIME_WORKSTEALING

  EXPERIMENTAL/ADVANCED: Queue workstealing method to use. Default is no 
  workstealing (SC_RUNTIME_WORKSTEALING=0) If enabled threads will steal
  work from other queues when they have no work of there own. If set to 1
  will only steal from neighboring queues. If set to 2 will steal from all 
  queues.

