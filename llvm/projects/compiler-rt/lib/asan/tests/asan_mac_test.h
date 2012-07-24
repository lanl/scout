extern "C" {
  void *CFAllocatorDefaultDoubleFree(void *unused);
  void CFAllocatorSystemDefaultDoubleFree();
  void CFAllocatorMallocDoubleFree();
  void CFAllocatorMallocZoneDoubleFree();
  void CallFreeOnWorkqueue(void *mem);
  void TestGCDDispatchAsync();
  void TestGCDDispatchSync();
  void TestGCDReuseWqthreadsAsync();
  void TestGCDReuseWqthreadsSync();
  void TestGCDDispatchAfter();
  void TestGCDInTSDDestructor();
  void TestGCDSourceEvent();
  void TestGCDSourceCancel();
  void TestGCDGroupAsync();
  void TestOOBNSObjects();
  void TestNSURLDeallocation();
<<<<<<< HEAD
=======
  void TestPassCFMemoryToAnotherThread();
>>>>>>> 853733e772b2885d93fdf994dedc4a1b5dc1369e
}
