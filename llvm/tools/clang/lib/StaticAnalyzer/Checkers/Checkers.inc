
#ifdef GET_GROUPS
GROUP("all-experimental")
#endif // GET_GROUPS


#ifdef GET_PACKAGES
PACKAGE("core", -1, false)
PACKAGE("core.builtin", -1, false)
PACKAGE("core.experimental", 0, true)
PACKAGE("core.uninitialized", -1, false)
PACKAGE("cplusplus", -1, false)
PACKAGE("cplusplus.experimental", 0, true)
PACKAGE("deadcode", -1, false)
PACKAGE("deadcode.experimental", 0, true)
PACKAGE("debug", -1, false)
PACKAGE("llvm", -1, false)
PACKAGE("osx", -1, false)
PACKAGE("osx.cocoa", -1, false)
PACKAGE("osx.cocoa.experimental", 0, true)
PACKAGE("osx.coreFoundation", -1, false)
PACKAGE("security", -1, false)
PACKAGE("security.experimental", 0, true)
PACKAGE("unix", -1, false)
PACKAGE("unix.experimental", 0, true)
#endif // GET_PACKAGES


#ifdef GET_CHECKERS
CHECKER("core.AdjustedReturnValue", AdjustedReturnValueChecker, AdjustedReturnValueChecker.cpp, "Check to see if the return value of a function call is different than the caller expects (e.g., from calls through function pointers)", -1, false)
CHECKER("debug.Stats", AnalyzerStatsChecker, AnalyzerStatsChecker.cpp, "Emit warnings with analyzer statistics", -1, false)
CHECKER("security.experimental.ArrayBound", ArrayBoundChecker, ArrayBoundChecker.cpp, "Warn about buffer overflows (older checker)", -1, true)
CHECKER("security.experimental.ArrayBoundV2", ArrayBoundCheckerV2, ArrayBoundCheckerV2.cpp, "Warn about buffer overflows (newer checker)", -1, true)
CHECKER("core.AttributeNonNull", AttrNonNullChecker, AttrNonNullChecker.cpp, "Check for null pointers passed as arguments to a function whose arguments are marked with the 'nonnull' attribute", -1, false)
CHECKER("core.builtin.BuiltinFunctions", BuiltinFunctionChecker, BuiltinFunctionChecker.cpp, "Evaluate compiler builtin functions (e.g., alloca())", -1, false)
CHECKER("osx.coreFoundation.CFError", CFErrorChecker, NSErrorChecker.cpp, "Check usage of CFErrorRef* parameters", -1, false)
CHECKER("debug.DumpCFG", CFGDumper, DebugCheckers.cpp, "Display Control-Flow Graphs", -1, false)
CHECKER("debug.ViewCFG", CFGViewer, DebugCheckers.cpp, "View Control-Flow Graphs using GraphViz", -1, false)
CHECKER("osx.coreFoundation.CFNumber", CFNumberCreateChecker, BasicObjCFoundationChecks.cpp, "Check for proper uses of CFNumberCreate", -1, false)
CHECKER("osx.coreFoundation.CFRetainRelease", CFRetainReleaseChecker, BasicObjCFoundationChecks.cpp, "Check for null arguments to CFRetain/CFRelease", -1, false)
CHECKER("unix.experimental.CString", CStringChecker, CStringChecker.cpp, "Check calls to functions in <string.h>", -1, true)
CHECKER("core.CallAndMessage", CallAndMessageChecker, CallAndMessageChecker.cpp, "Check for logical errors for function calls and Objective-C message expressions (e.g., uninitialized arguments, null function pointers)", -1, false)
CHECKER("core.experimental.CastSize", CastSizeChecker, CastSizeChecker.cpp, "Check when casting a malloc'ed type T, whether the size is a multiple of the size of T", -1, true)
CHECKER("core.experimental.CastToStruct", CastToStructChecker, CastToStructChecker.cpp, "Check for cast from non-struct pointer to struct pointer", -1, true)
CHECKER("unix.experimental.Chroot", ChrootChecker, ChrootChecker.cpp, "Check improper use of chroot", -1, true)
CHECKER("osx.cocoa.ClassRelease", ClassReleaseChecker, BasicObjCFoundationChecks.cpp, "Check for sending 'retain', 'release', or 'autorelease' directly to a Class", -1, false)
CHECKER("deadcode.DeadStores", DeadStoresChecker, DeadStoresChecker.cpp, "Check for values stored to variables that are never read afterwards", -1, false)
CHECKER("core.NullDereference", DereferenceChecker, DereferenceChecker.cpp, "Check for dereferences of null pointers", -1, false)
CHECKER("core.DivideZero", DivZeroChecker, DivZeroChecker.cpp, "Check for division by zero", -1, false)
CHECKER("core.experimental.FixedAddr", FixedAddressChecker, FixedAddressChecker.cpp, "Check for assignment of a fixed address to a pointer", -1, true)
CHECKER("deadcode.IdempotentOperations", IdempotentOperationChecker, IdempotentOperationChecker.cpp, "Warn about idempotent operations", -1, false)
CHECKER("cplusplus.experimental.Iterators", IteratorsChecker, IteratorsChecker.cpp, "Check improper uses of STL vector iterators", -1, true)
CHECKER("llvm.Conventions", LLVMConventionsChecker, LLVMConventionsChecker.cpp, "Check code for LLVM codebase conventions", -1, false)
CHECKER("debug.DumpLiveVars", LiveVariablesDumper, DebugCheckers.cpp, "Print results of live variable analysis", -1, false)
CHECKER("osx.API", MacOSXAPIChecker, MacOSXAPIChecker.cpp, "Check for proper uses of various Mac OS X APIs", -1, false)
CHECKER("unix.experimental.Malloc", MallocChecker, MallocChecker.cpp, "Check for potential memory leaks, double free, and use-after-free problems", -1, true)
CHECKER("osx.cocoa.NSAutoreleasePool", NSAutoreleasePoolChecker, NSAutoreleasePoolChecker.cpp, "Warn for suboptimal uses of NSAutoreleasePool in Objective-C GC mode", -1, false)
CHECKER("osx.cocoa.NSError", NSErrorChecker, NSErrorChecker.cpp, "Check usage of NSError** parameters", -1, false)
CHECKER("osx.cocoa.NilArg", NilArgChecker, BasicObjCFoundationChecks.cpp, "Check for prohibited nil arguments to ObjC method calls", -1, false)
CHECKER("core.builtin.NoReturnFunctions", NoReturnFunctionChecker, NoReturnFunctionChecker.cpp, "Evaluate \"panic\" functions that are known to not return to the caller", -1, false)
CHECKER("osx.AtomicCAS", OSAtomicChecker, OSAtomicChecker.cpp, "Evaluate calls to OSAtomic functions", -1, false)
CHECKER("osx.cocoa.AtSync", ObjCAtSyncChecker, ObjCAtSyncChecker.cpp, "Check for null pointers used as mutexes for @synchronized", -1, false)
CHECKER("osx.cocoa.experimental.Dealloc", ObjCDeallocChecker, CheckObjCDealloc.cpp, "Warn about Objective-C classes that lack a correct implementation of -dealloc", -1, true)
CHECKER("osx.cocoa.IncompatibleMethodTypes", ObjCMethSigsChecker, CheckObjCInstMethSignature.cpp, "Warn about Objective-C method signatures with type incompatibilities", -1, false)
CHECKER("osx.cocoa.experimental.SelfInit", ObjCSelfInitChecker, ObjCSelfInitChecker.cpp, "Check that 'self' is properly initialized inside an initializer method", -1, true)
CHECKER("osx.cocoa.UnusedIvars", ObjCUnusedIvarsChecker, ObjCUnusedIVarsChecker.cpp, "Warn about private ivars that are never used", -1, false)
CHECKER("core.experimental.PointerArithm", PointerArithChecker, PointerArithChecker, "Check for pointer arithmetic on locations other than array elements", -1, true)
CHECKER("core.experimental.PointerSub", PointerSubChecker, PointerSubChecker, "Check for pointer subtractions on two pointers pointing to different memory chunks", -1, true)
CHECKER("unix.experimental.PthreadLock", PthreadLockChecker, PthreadLockChecker.cpp, "Simple lock -> unlock checker", -1, true)
CHECKER("security.experimental.ReturnPtrRange", ReturnPointerRangeChecker, ReturnPointerRangeChecker.cpp, "Check for an out-of-bound pointer being returned to callers", -1, true)
CHECKER("core.uninitialized.UndefReturn", ReturnUndefChecker, ReturnUndefChecker.cpp, "Check for uninitialized values being returned to the caller", -1, false)
CHECKER("security.experimental.SecuritySyntactic", SecuritySyntaxChecker, CheckSecuritySyntaxOnly.cpp, "Perform quick security API checks that require no data flow", -1, true)
CHECKER("core.experimental.SizeofPtr", SizeofPointerChecker, CheckSizeofPointer.cpp, "Warn about unintended use of sizeof() on pointer expressions", -1, true)
CHECKER("core.StackAddressEscape", StackAddrEscapeChecker, StackAddrEscapeChecker.cpp, "Check that addresses to stack memory do not escape the function", -1, false)
CHECKER("unix.experimental.Stream", StreamChecker, StreamChecker.cpp, "Check stream handling functions", -1, true)
CHECKER("core.uninitialized.Branch", UndefBranchChecker, UndefBranchChecker.cpp, "Check for uninitialized values used as branch conditions", -1, false)
CHECKER("core.uninitialized.CapturedBlockVariable", UndefCapturedBlockVarChecker, UndefCapturedBlockVarChecker.cpp, "Check for blocks that capture uninitialized values", -1, false)
CHECKER("core.UndefinedBinaryOperatorResult", UndefResultChecker, UndefResultChecker.cpp, "Check for undefined results of binary operators", -1, false)
CHECKER("core.uninitialized.ArraySubscript", UndefinedArraySubscriptChecker, UndefinedArraySubscriptChecker.cpp, "Check for uninitialized values used as array subscripts", -1, false)
CHECKER("core.uninitialized.Assign", UndefinedAssignmentChecker, UndefinedAssignmentChecker.cpp, "Check for assigning uninitialized values", -1, false)
CHECKER("unix.API", UnixAPIChecker, UnixAPIChecker.cpp, "Check calls to various UNIX/Posix functions", -1, false)
CHECKER("deadcode.experimental.UnreachableCode", UnreachableCodeChecker, UnreachableCodeChecker.cpp, "Check unreachable code", -1, true)
CHECKER("core.VLASize", VLASizeChecker, VLASizeChecker.cpp, "Check for declarations of VLA of undefined or zero size", -1, false)
CHECKER("osx.cocoa.VariadicMethodTypes", VariadicMethodTypeChecker, BasicObjCFoundationChecks.cpp, "Check for passing non-Objective-C types to variadic methods that expectonly Objective-C types", -1, false)
#endif // GET_CHECKERS


#ifdef GET_MEMBER_ARRAYS
static const short CheckerArray0[] = { 13, 14, 20, 37, 38, 43, 22, 52, 42, 2, 3, 40, 15, 11, 26, 39, 45, 35, 33, -1 };
static const short SubPackageArray1[] = { 10, 20, 13, 6, 4, 2, 3, 9, 5, 8, 7, -1 };
static const short CheckerArray2[] = { 0, -1 };
static const short CheckerArray3[] = { 4, -1 };
static const short CheckerArray4[] = { 12, -1 };
static const short CheckerArray5[] = { 19, -1 };
static const short CheckerArray6[] = { 18, -1 };
static const short CheckerArray7[] = { 44, -1 };
static const short CheckerArray8[] = { 48, -1 };
static const short CheckerArray9[] = { 53, -1 };
static const short SubPackageArray10[] = { 12, 11, -1 };
static const short CheckerArray11[] = { 5, -1 };
static const short CheckerArray12[] = { 30, -1 };
static const short SubPackageArray13[] = { 14, 15, 16, 17, 18, 19, -1 };
static const short CheckerArray14[] = { 13, -1 };
static const short CheckerArray15[] = { 14, -1 };
static const short CheckerArray16[] = { 20, -1 };
static const short CheckerArray17[] = { 37, -1 };
static const short CheckerArray18[] = { 38, -1 };
static const short CheckerArray19[] = { 43, -1 };
static const short SubPackageArray20[] = { 21, 22, 23, 24, 25, -1 };
static const short CheckerArray21[] = { 49, -1 };
static const short CheckerArray22[] = { 50, -1 };
static const short CheckerArray23[] = { 46, -1 };
static const short CheckerArray24[] = { 47, -1 };
static const short CheckerArray25[] = { 41, -1 };
static const short SubPackageArray26[] = { 27, -1 };
static const short SubPackageArray27[] = { 28, -1 };
static const short CheckerArray28[] = { 22, -1 };
static const short SubPackageArray29[] = { 32, 30, 31, -1 };
static const short CheckerArray30[] = { 17, -1 };
static const short CheckerArray31[] = { 21, -1 };
static const short SubPackageArray32[] = { 33, -1 };
static const short CheckerArray33[] = { 52, -1 };
static const short SubPackageArray34[] = { 36, 38, 35, 37, -1 };
static const short CheckerArray35[] = { 7, -1 };
static const short CheckerArray36[] = { 24, -1 };
static const short CheckerArray37[] = { 1, -1 };
static const short CheckerArray38[] = { 8, -1 };
static const short SubPackageArray39[] = { 40, -1 };
static const short CheckerArray40[] = { 23, -1 };
static const short SubPackageArray41[] = { 44, 56, 42, 43, -1 };
static const short CheckerArray42[] = { 25, -1 };
static const short CheckerArray43[] = { 31, -1 };
static const short SubPackageArray44[] = { 53, 45, 50, 46, 52, 48, 47, 51, 49, -1 };
static const short CheckerArray45[] = { 32, -1 };
static const short CheckerArray46[] = { 16, -1 };
static const short CheckerArray47[] = { 34, -1 };
static const short CheckerArray48[] = { 27, -1 };
static const short CheckerArray49[] = { 28, -1 };
static const short CheckerArray50[] = { 29, -1 };
static const short CheckerArray51[] = { 36, -1 };
static const short CheckerArray52[] = { 54, -1 };
static const short SubPackageArray53[] = { 55, 54, -1 };
static const short CheckerArray54[] = { 33, -1 };
static const short CheckerArray55[] = { 35, -1 };
static const short SubPackageArray56[] = { 58, 59, 57, -1 };
static const short CheckerArray57[] = { 6, -1 };
static const short CheckerArray58[] = { 9, -1 };
static const short CheckerArray59[] = { 10, -1 };
static const short SubPackageArray60[] = { 61, -1 };
static const short SubPackageArray61[] = { 65, 62, 63, 64, -1 };
static const short CheckerArray62[] = { 2, -1 };
static const short CheckerArray63[] = { 3, -1 };
static const short CheckerArray64[] = { 40, -1 };
static const short CheckerArray65[] = { 42, -1 };
static const short SubPackageArray66[] = { 68, 67, -1 };
static const short CheckerArray67[] = { 51, -1 };
static const short SubPackageArray68[] = { 70, 69, 71, 72, 73, -1 };
static const short CheckerArray69[] = { 11, -1 };
static const short CheckerArray70[] = { 15, -1 };
static const short CheckerArray71[] = { 26, -1 };
static const short CheckerArray72[] = { 39, -1 };
static const short CheckerArray73[] = { 45, -1 };
#endif // GET_MEMBER_ARRAYS


#ifdef GET_CHECKNAME_TABLE
  { "all-experimental",                         CheckerArray0, 0, false },
  { "core",                                     0, SubPackageArray1, false },
  { "core.AdjustedReturnValue",                 CheckerArray2, 0, false },
  { "core.AttributeNonNull",                    CheckerArray3, 0, false },
  { "core.CallAndMessage",                      CheckerArray4, 0, false },
  { "core.DivideZero",                          CheckerArray5, 0, false },
  { "core.NullDereference",                     CheckerArray6, 0, false },
  { "core.StackAddressEscape",                  CheckerArray7, 0, false },
  { "core.UndefinedBinaryOperatorResult",       CheckerArray8, 0, false },
  { "core.VLASize",                             CheckerArray9, 0, false },
  { "core.builtin",                             0, SubPackageArray10, false },
  { "core.builtin.BuiltinFunctions",            CheckerArray11, 0, false },
  { "core.builtin.NoReturnFunctions",           CheckerArray12, 0, false },
  { "core.experimental",                        0, SubPackageArray13, true },
  { "core.experimental.CastSize",               CheckerArray14, 0, false },
  { "core.experimental.CastToStruct",           CheckerArray15, 0, false },
  { "core.experimental.FixedAddr",              CheckerArray16, 0, false },
  { "core.experimental.PointerArithm",          CheckerArray17, 0, false },
  { "core.experimental.PointerSub",             CheckerArray18, 0, false },
  { "core.experimental.SizeofPtr",              CheckerArray19, 0, false },
  { "core.uninitialized",                       0, SubPackageArray20, false },
  { "core.uninitialized.ArraySubscript",        CheckerArray21, 0, false },
  { "core.uninitialized.Assign",                CheckerArray22, 0, false },
  { "core.uninitialized.Branch",                CheckerArray23, 0, false },
  { "core.uninitialized.CapturedBlockVariable", CheckerArray24, 0, false },
  { "core.uninitialized.UndefReturn",           CheckerArray25, 0, false },
  { "cplusplus",                                0, SubPackageArray26, false },
  { "cplusplus.experimental",                   0, SubPackageArray27, true },
  { "cplusplus.experimental.Iterators",         CheckerArray28, 0, false },
  { "deadcode",                                 0, SubPackageArray29, false },
  { "deadcode.DeadStores",                      CheckerArray30, 0, false },
  { "deadcode.IdempotentOperations",            CheckerArray31, 0, false },
  { "deadcode.experimental",                    0, SubPackageArray32, true },
  { "deadcode.experimental.UnreachableCode",    CheckerArray33, 0, false },
  { "debug",                                    0, SubPackageArray34, false },
  { "debug.DumpCFG",                            CheckerArray35, 0, false },
  { "debug.DumpLiveVars",                       CheckerArray36, 0, false },
  { "debug.Stats",                              CheckerArray37, 0, false },
  { "debug.ViewCFG",                            CheckerArray38, 0, false },
  { "llvm",                                     0, SubPackageArray39, false },
  { "llvm.Conventions",                         CheckerArray40, 0, false },
  { "osx",                                      0, SubPackageArray41, false },
  { "osx.API",                                  CheckerArray42, 0, false },
  { "osx.AtomicCAS",                            CheckerArray43, 0, false },
  { "osx.cocoa",                                0, SubPackageArray44, false },
  { "osx.cocoa.AtSync",                         CheckerArray45, 0, false },
  { "osx.cocoa.ClassRelease",                   CheckerArray46, 0, false },
  { "osx.cocoa.IncompatibleMethodTypes",        CheckerArray47, 0, false },
  { "osx.cocoa.NSAutoreleasePool",              CheckerArray48, 0, false },
  { "osx.cocoa.NSError",                        CheckerArray49, 0, false },
  { "osx.cocoa.NilArg",                         CheckerArray50, 0, false },
  { "osx.cocoa.UnusedIvars",                    CheckerArray51, 0, false },
  { "osx.cocoa.VariadicMethodTypes",            CheckerArray52, 0, false },
  { "osx.cocoa.experimental",                   0, SubPackageArray53, true },
  { "osx.cocoa.experimental.Dealloc",           CheckerArray54, 0, false },
  { "osx.cocoa.experimental.SelfInit",          CheckerArray55, 0, false },
  { "osx.coreFoundation",                       0, SubPackageArray56, false },
  { "osx.coreFoundation.CFError",               CheckerArray57, 0, false },
  { "osx.coreFoundation.CFNumber",              CheckerArray58, 0, false },
  { "osx.coreFoundation.CFRetainRelease",       CheckerArray59, 0, false },
  { "security",                                 0, SubPackageArray60, false },
  { "security.experimental",                    0, SubPackageArray61, true },
  { "security.experimental.ArrayBound",         CheckerArray62, 0, false },
  { "security.experimental.ArrayBoundV2",       CheckerArray63, 0, false },
  { "security.experimental.ReturnPtrRange",     CheckerArray64, 0, false },
  { "security.experimental.SecuritySyntactic",  CheckerArray65, 0, false },
  { "unix",                                     0, SubPackageArray66, false },
  { "unix.API",                                 CheckerArray67, 0, false },
  { "unix.experimental",                        0, SubPackageArray68, true },
  { "unix.experimental.CString",                CheckerArray69, 0, false },
  { "unix.experimental.Chroot",                 CheckerArray70, 0, false },
  { "unix.experimental.Malloc",                 CheckerArray71, 0, false },
  { "unix.experimental.PthreadLock",            CheckerArray72, 0, false },
  { "unix.experimental.Stream",                 CheckerArray73, 0, false },
#endif // GET_CHECKNAME_TABLE

