//===-- driver.cpp - Clang GCC-Compatible Driver --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang driver; it is a thin wrapper
// for functionality in the Driver clang library.
//
//===----------------------------------------------------------------------===//

#include <unistd.h>
#include <iostream>
#include <sstream>

#include "clang/Basic/CharInfo.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>
using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>

#include <unistd.h>

#include "scout/Config/Configuration.h"


std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes)
    return Argv0;

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

static const char *GetStableCStr(std::set<std::string> &SavedStrings,
                                 StringRef S) {
  return SavedStrings.insert(S).first->c_str();
}

/// ApplyQAOverride - Apply a list of edits to the input argument lists.
///
/// The input string is a space separate list of edits to perform,
/// they are applied in order to the input argument lists. Edits
/// should be one of the following forms:
///
///  '#': Silence information about the changes to the command line arguments.
///
///  '^': Add FOO as a new argument at the beginning of the command line.
///
///  '+': Add FOO as a new argument at the end of the command line.
///
///  's/XXX/YYY/': Substitute the regular expression XXX with YYY in the command
///  line.
///
///  'xOPTION': Removes all instances of the literal argument OPTION.
///
///  'XOPTION': Removes all instances of the literal argument OPTION,
///  and the following argument.
///
///  'Ox': Removes all flags matching 'O' or 'O[sz0-9]' and adds 'Ox'
///  at the end of the command line.
///
/// \param OS - The stream to write edit information to.
/// \param Args - The vector of command line arguments.
/// \param Edit - The override command to perform.
/// \param SavedStrings - Set to use for storing string representations.
static void ApplyOneQAOverride(raw_ostream &OS,
                               SmallVectorImpl<const char*> &Args,
                               StringRef Edit,
                               std::set<std::string> &SavedStrings) {
  // This does not need to be efficient.

  if (Edit[0] == '^') {
    const char *Str =
      GetStableCStr(SavedStrings, Edit.substr(1));
    OS << "### Adding argument " << Str << " at beginning\n";
    Args.insert(Args.begin() + 1, Str);
  } else if (Edit[0] == '+') {
    const char *Str =
      GetStableCStr(SavedStrings, Edit.substr(1));
    OS << "### Adding argument " << Str << " at end\n";
    Args.push_back(Str);
  } else if (Edit[0] == 's' && Edit[1] == '/' && Edit.endswith("/") &&
             Edit.slice(2, Edit.size()-1).find('/') != StringRef::npos) {
    StringRef MatchPattern = Edit.substr(2).split('/').first;
    StringRef ReplPattern = Edit.substr(2).split('/').second;
    ReplPattern = ReplPattern.slice(0, ReplPattern.size()-1);

    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      // Ignore end-of-line response file markers
      if (Args[i] == nullptr)
        continue;
      std::string Repl = llvm::Regex(MatchPattern).sub(ReplPattern, Args[i]);

      if (Repl != Args[i]) {
        OS << "### Replacing '" << Args[i] << "' with '" << Repl << "'\n";
        Args[i] = GetStableCStr(SavedStrings, Repl);
      }
    }
  } else if (Edit[0] == 'x' || Edit[0] == 'X') {
    std::string Option = Edit.substr(1, std::string::npos);
    for (unsigned i = 1; i < Args.size();) {
      if (Option == Args[i]) {
        OS << "### Deleting argument " << Args[i] << '\n';
        Args.erase(Args.begin() + i);
        if (Edit[0] == 'X') {
          if (i < Args.size()) {
            OS << "### Deleting argument " << Args[i] << '\n';
            Args.erase(Args.begin() + i);
          } else
            OS << "### Invalid X edit, end of command line!\n";
        }
      } else
        ++i;
    }
  } else if (Edit[0] == 'O') {
    for (unsigned i = 1; i < Args.size();) {
      const char *A = Args[i];
      // Ignore end-of-line response file markers
      if (A == nullptr)
        continue;
      if (A[0] == '-' && A[1] == 'O' &&
          (A[2] == '\0' ||
           (A[3] == '\0' && (A[2] == 's' || A[2] == 'z' ||
                             ('0' <= A[2] && A[2] <= '9'))))) {
        OS << "### Deleting argument " << Args[i] << '\n';
        Args.erase(Args.begin() + i);
      } else
        ++i;
    }
    OS << "### Adding argument " << Edit << " at end\n";
    Args.push_back(GetStableCStr(SavedStrings, '-' + Edit.str()));
  } else {
    OS << "### Unrecognized edit: " << Edit << "\n";
  }
}

/// ApplyQAOverride - Apply a comma separate list of edits to the
/// input argument lists. See ApplyOneQAOverride.
static void ApplyQAOverride(SmallVectorImpl<const char*> &Args,
                            const char *OverrideStr,
                            std::set<std::string> &SavedStrings) {
  raw_ostream *OS = &llvm::errs();

  if (OverrideStr[0] == '#') {
    ++OverrideStr;
    OS = &llvm::nulls();
  }

  *OS << "### CCC_OVERRIDE_OPTIONS: " << OverrideStr << "\n";

  // This does not need to be efficient.

  const char *S = OverrideStr;
  while (*S) {
    const char *End = ::strchr(S, ' ');
    if (!End)
      End = S + strlen(S);
    if (End != S)
      ApplyOneQAOverride(*OS, Args, std::string(S, End), SavedStrings);
    S = End;
    if (*S != '\0')
      ++S;
  }
}

extern int cc1_main(ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr, bool Rewrite, bool DumpRewrite);
extern int cc1as_main(ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);

struct DriverSuffix {
  const char *Suffix;
  const char *ModeFlag;
};

static const DriverSuffix *FindDriverSuffix(StringRef ProgName) {
  // A list of known driver suffixes. Suffixes are compared against the
  // program name in order. If there is a match, the frontend type if updated as
  // necessary by applying the ModeFlag.
  static const DriverSuffix DriverSuffixes[] = {
    { "scc",       "--driver-mode=scout" },
    { "clang",     0 },
    // Scout has to go high-up or we'll miss it...
    { "scc",       "--driver-mode=scout" },
    { "clang++",   "--driver-mode=g++" },
    { "clang-c++", "--driver-mode=g++" },
    { "clang-cc",  0 },
    { "clang-cpp", "--driver-mode=cpp" },
    { "clang-g++", "--driver-mode=g++" },
    { "clang-gcc", 0 },
    { "clang-cl",  "--driver-mode=cl"  },
    { "cc",        0 },
    { "cpp",       "--driver-mode=cpp" },
    { "cl" ,       "--driver-mode=cl"  },
    { "++",        "--driver-mode=g++" }
  };

  for (size_t i = 0; i < llvm::array_lengthof(DriverSuffixes); ++i)
    if (ProgName.endswith(DriverSuffixes[i].Suffix))
      return &DriverSuffixes[i];
  return nullptr;
}

static void ParseProgName(SmallVectorImpl<const char *> &ArgVector,
                          std::set<std::string> &SavedStrings) {
  // Try to infer frontend type and default target from the program name by
  // comparing it against DriverSuffixes in order.

  // If there is a match, the function tries to identify a target as prefix.
  // E.g. "x86_64-linux-clang" as interpreted as suffix "clang" with target
  // prefix "x86_64-linux". If such a target prefix is found, is gets added via
  // -target as implicit first argument.

  std::string ProgName =llvm::sys::path::stem(ArgVector[0]);
#ifdef LLVM_ON_WIN32
  // Transform to lowercase for case insensitive file systems.
  ProgName = StringRef(ProgName).lower();
#endif

  StringRef ProgNameRef = ProgName;
  const DriverSuffix *DS = FindDriverSuffix(ProgNameRef);

  if (!DS) {
    // Try again after stripping any trailing version number:
    // clang++3.5 -> clang++
    ProgNameRef = ProgNameRef.rtrim("0123456789.");
    DS = FindDriverSuffix(ProgNameRef);
  }

  if (!DS) {
    // Try again after stripping trailing -component.
    // clang++-tot -> clang++
    ProgNameRef = ProgNameRef.slice(0, ProgNameRef.rfind('-'));
    DS = FindDriverSuffix(ProgNameRef);
  }

  if (DS) {
    if (const char *Flag = DS->ModeFlag) {
      // Add Flag to the arguments.
      auto it = ArgVector.begin();
      if (it != ArgVector.end())
        ++it;
      ArgVector.insert(it, Flag);
    }

    StringRef::size_type LastComponent = ProgNameRef.rfind(
        '-', ProgNameRef.size() - strlen(DS->Suffix));
    if (LastComponent == StringRef::npos)
      return;

    // Infer target from the prefix.
    StringRef Prefix = ProgNameRef.slice(0, LastComponent);
    std::string IgnoredError;
    if (llvm::TargetRegistry::lookupTarget(Prefix, IgnoredError)) {
      auto it = ArgVector.begin();
      if (it != ArgVector.end())
        ++it;
      const char *arr[] = { "-target", GetStableCStr(SavedStrings, Prefix) };
      ArgVector.insert(it, std::begin(arr), std::end(arr));
    }
  }
}

namespace {
  class StringSetSaver : public llvm::cl::StringSaver {
  public:
    StringSetSaver(std::set<std::string> &Storage) : Storage(Storage) {}
    const char *SaveString(const char *Str) override {
      return GetStableCStr(Storage, Str);
    }
  private:
    std::set<std::string> &Storage;
  };
}

static void SetBackdoorDriverOutputsFromEnvVars(Driver &TheDriver) {
  // Handle CC_PRINT_OPTIONS and CC_PRINT_OPTIONS_FILE.
  TheDriver.CCPrintOptions = !!::getenv("CC_PRINT_OPTIONS");
  if (TheDriver.CCPrintOptions)
    TheDriver.CCPrintOptionsFilename = ::getenv("CC_PRINT_OPTIONS_FILE");

  // Handle CC_PRINT_HEADERS and CC_PRINT_HEADERS_FILE.
  TheDriver.CCPrintHeaders = !!::getenv("CC_PRINT_HEADERS");
  if (TheDriver.CCPrintHeaders)
    TheDriver.CCPrintHeadersFilename = ::getenv("CC_PRINT_HEADERS_FILE");

  // Handle CC_LOG_DIAGNOSTICS and CC_LOG_DIAGNOSTICS_FILE.
  TheDriver.CCLogDiagnostics = !!::getenv("CC_LOG_DIAGNOSTICS");
  if (TheDriver.CCLogDiagnostics)
    TheDriver.CCLogDiagnosticsFilename = ::getenv("CC_LOG_DIAGNOSTICS_FILE");
}

static void FixupDiagPrefixExeName(TextDiagnosticPrinter *DiagClient,
                                   const std::string &Path) {
  // If the clang binary happens to be named cl.exe for compatibility reasons,
  // use clang-cl.exe as the prefix to avoid confusion between clang and MSVC.
  StringRef ExeBasename(llvm::sys::path::filename(Path));
  if (ExeBasename.equals_lower("cl.exe"))
    ExeBasename = "clang-cl.exe";
  DiagClient->setPrefix(ExeBasename);
}

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance.
static DiagnosticOptions *
CreateAndPopulateDiagOpts(SmallVectorImpl<const char *> &argv) {
  auto *DiagOpts = new DiagnosticOptions;
  std::unique_ptr<OptTable> Opts(createDriverOptTable());
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<InputArgList> Args(Opts->ParseArgs(
      argv.begin() + 1, argv.end(), MissingArgIndex, MissingArgCount));
  // We ignore MissingArgCount and the return value of ParseDiagnosticArgs.
  // Any errors that would be diagnosed here will also be diagnosed later,
  // when the DiagnosticsEngine actually exists.
  (void) ParseDiagnosticArgs(*DiagOpts, *Args);
  return DiagOpts;
}

static void SetInstallDir(SmallVectorImpl<const char *> &argv,
                          Driver &TheDriver) {
  // Attempt to find the original path used to invoke the driver, to determine
  // the installed path. We do this manually, because we want to support that
  // path being a symlink.
  SmallString<128> InstalledPath(argv[0]);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(InstalledPath) == InstalledPath)
    if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(InstalledPath.str())))
      InstalledPath = *Tmp;
  llvm::sys::fs::make_absolute(InstalledPath);
  InstalledPath = llvm::sys::path::parent_path(InstalledPath);
  if (llvm::sys::fs::exists(InstalledPath.c_str()))
    TheDriver.setInstalledDir(InstalledPath);
}

static int ExecuteCC1Tool(ArrayRef<const char *> argv,
                          StringRef Tool,
                          bool Rewrite,
                          bool DumpRewrite) {
  void *GetExecutablePathVP = (void *)(intptr_t) GetExecutablePath;
  if (Tool == "")
    return cc1_main(argv.slice(2), argv[0], GetExecutablePathVP,
                    Rewrite, DumpRewrite);
  if (Tool == "as")
    return cc1as_main(argv.slice(2), argv[0], GetExecutablePathVP);

  // Reject unknown tools.
  llvm::errs() << "error: unknown integrated tool '" << Tool << "'\n";
  return 1;
}

// ----- scAddStringIfUnique
//
static void scAddStringIfUnique(std::string &arg_str,
                                const std::string &s) {
  if (s == "+-framework") {
    arg_str += s + std::string(" ");
  } else {
    if (arg_str.find(s) == std::string::npos) {
      arg_str += s + std::string(" ");
    }
  }
}


// ----- scCheckForFlag
//
static bool scCheckForFlag(const char *flag,
                           SmallVectorImpl<const char*> &argv) {
  for (int i = 1, size = argv.size(); i < size; ++i) {
    if (StringRef(argv[i]) == flag) {
      return true;
    }
  }
  return false;
}



// ---- scAddFlagSet
//
static void scAddFlagSet(std::string& args,
                         const char *args_to_add[]) {
  int i = 0;
  while(args_to_add[i] != 0) {
    // split space-delimited string into words.
    std::string input_string(args_to_add[i]);
    std::istringstream iss(input_string);
    std::vector<std::string> tokens;

    copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter<std::vector<std::string> >(tokens));

    std::vector<std::string>::iterator it = tokens.begin();

    while(it != tokens.end()) {
      scAddStringIfUnique(args, std::string("+") + *it);
      ++it;
    }
    ++i;
  }
}


// ----- scAddFlags
//
static void scAddFlags(Driver &driver,
                       SmallVectorImpl<const char*> &argv,
                       std::set<std::string> &SavedStrings) {

  // Check the command line arguments in a bit more detail before
  // we get to work...
  std::unique_ptr<OptTable> CC1Opts(createDriverOptTable());
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<InputArgList> Args(CC1Opts->ParseArgs(argv.begin()+1, argv.end(),
                                                  MissingArgIndex, MissingArgCount));

  // Build up a string of edits to make to the command line options
  // and use the Arg override mechanism above to weasel scout-centric
  // options into place...

  // Not sure this works in all cases (e.g. odd sym links?).
  std::string sc_install_prefix=llvm::sys::path::parent_path(driver.Dir);

  // Put scout's include directory up front (assuming we search for
  // headers in order from first to last on the command line).  Place
  // a '#' at the head of the sc_args string to silence info about the
  // changes when we run 'scc'.
  std::string sc_args = "#^-I" + sc_install_prefix + "/include ";

  scAddFlagSet(sc_args, scout::config::Configuration::CompileOptions);
  scAddFlagSet(sc_args, scout::config::Configuration::IncludePaths);

  // Check to see if we are linking -- avoid adding link flags if we
  // don't need them (otherwise, we get a lot of potentially confusing
  // warnings at the command line).
  if (!(Args->hasArg(options::OPT_c) ||
        Args->hasArg(options::OPT_S) ||
        Args->hasArg(options::OPT_fsyntax_only))) {
    scAddFlagSet(sc_args,
                 scout::config::Configuration::LinkOptions);
    scAddFlagSet(sc_args,
                 scout::config::Configuration::LibraryPaths);
    scAddFlagSet(sc_args,
                 scout::config::Configuration::Libraries);
    // SC_TODO: might be nice to do something with 'rpath' here to
    // simplify the compiler and all the hidden dyanmic library
    // paths... This however, probably won't play well with modules...
  }

  ApplyQAOverride(argv, sc_args.c_str(), SavedStrings);
}


int main(int argc_, const char **argv_) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc_, argv_);

  if (llvm::sys::Process::FixupStandardFileDescriptors())
    return 1;

  SmallVector<const char *, 256> argv;
  llvm::SpecificBumpPtrAllocator<char> ArgAllocator;
  std::error_code EC = llvm::sys::Process::GetArgumentVector(
      argv, llvm::makeArrayRef(argv_, argc_), ArgAllocator);
  if (EC) {
    llvm::errs() << "error: couldn't get arguments: " << EC.message() << '\n';
    return 1;
  }

  std::set<std::string> SavedStrings;
  StringSetSaver Saver(SavedStrings);

  // Determines whether we want nullptr markers in argv to indicate response
  // files end-of-lines. We only use this for the /LINK driver argument.
  bool MarkEOLs = true;
  if (argv.size() > 1 && StringRef(argv[1]).startswith("-cc1"))
    MarkEOLs = false;
  llvm::cl::ExpandResponseFiles(Saver, llvm::cl::TokenizeGNUCommandLine, argv,
                                MarkEOLs);

  //-debug flag
  for (int i = 2, size = argv.size(); i < size; ++i) {
    if (StringRef(argv[i]) == "-debug") {
      size_t pid = getpid();
      
      std::cerr << "PID: " << pid << std::endl;
      std::cerr << "<press any key after attaching debugger, then 'continue' in debugger>" << std::endl;
      std::string str;
      std::getline(std::cin, str);
    }
  }

  bool Rewrite = false;
  Rewrite = scCheckForFlag("-rewrite", argv);

  bool DumpRewrite = false;
  DumpRewrite = scCheckForFlag("-dump-rewrite", argv);

  // Handle -cc1 integrated tools, even if -cc1 was expanded from a response
  // file.
  auto FirstArg = std::find_if(argv.begin() + 1, argv.end(),
                               [](const char *A) { return A != nullptr; });
  if (FirstArg != argv.end() && StringRef(*FirstArg).startswith("-cc1")) {
    // If -cc1 came from a response file, remove the EOL sentinels.
    if (MarkEOLs) {
      auto newEnd = std::remove(argv.begin(), argv.end(), nullptr);
      argv.resize(newEnd - argv.begin());
    }
    return ExecuteCC1Tool(argv, argv[1] + 4, Rewrite, DumpRewrite);
  }

  bool CanonicalPrefixes = true;
  for (int i = 1, size = argv.size(); i < size; ++i) {
    // Skip end-of-line response file markers
    if (argv[i] == nullptr)
      continue;
    if (StringRef(argv[i]) == "-no-canonical-prefixes") {
      CanonicalPrefixes = false;
      break;
    }
  }

  // Handle CCC_OVERRIDE_OPTIONS, used for editing a command line behind the
  // scenes.
  if (const char *OverrideStr = ::getenv("CCC_OVERRIDE_OPTIONS")) {
    // FIXME: Driver shouldn't take extra initial argument.
    ApplyQAOverride(argv, OverrideStr, SavedStrings);
  }

  std::string Path = GetExecutablePath(argv[0], CanonicalPrefixes);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(argv);

  TextDiagnosticPrinter *DiagClient
    = new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
  FixupDiagPrefixExeName(DiagClient, Path);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  if (!DiagOpts->DiagnosticSerializationFile.empty()) {
    auto SerializedConsumer =
        clang::serialized_diags::create(DiagOpts->DiagnosticSerializationFile,
                                        &*DiagOpts, /*MergeChildRecords=*/true);
    Diags.setClient(new ChainedDiagnosticConsumer(
        Diags.takeClient(), std::move(SerializedConsumer)));
  }

  ProcessWarningOptions(Diags, *DiagOpts, /*ReportDiags=*/false);

  Driver TheDriver(Path, llvm::sys::getDefaultTargetTriple(), Diags);
  SetInstallDir(argv, TheDriver);

  llvm::InitializeAllTargets();
  ParseProgName(argv, SavedStrings);

  SetBackdoorDriverOutputsFromEnvVars(TheDriver);

  scAddFlags(TheDriver, argv, SavedStrings);

  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(argv));
  int Res = 0;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  if (C.get())
    Res = TheDriver.ExecuteCompilation(*C, FailingCommands);

  // Force a crash to test the diagnostics.
  if (::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH")) {
    Diags.Report(diag::err_drv_force_crash) << "FORCE_CLANG_DIAGNOSTICS_CRASH";

    // Pretend that every command failed.
    FailingCommands.clear();
    for (const auto &J : C->getJobs())
      if (const Command *C = dyn_cast<Command>(&J))
        FailingCommands.push_back(std::make_pair(-1, C));
  }

  for (const auto &P : FailingCommands) {
    int CommandRes = P.first;
    const Command *FailingCommand = P.second;
    if (!Res)
      Res = CommandRes;

    // If result status is < 0, then the driver command signalled an error.
    // If result status is 70, then the driver command reported a fatal error.
    // On Windows, abort will return an exit code of 3.  In these cases,
    // generate additional diagnostic information if possible.
    bool DiagnoseCrash = CommandRes < 0 || CommandRes == 70;
#ifdef LLVM_ON_WIN32
    DiagnoseCrash |= CommandRes == 3;
#endif
    if (DiagnoseCrash) {
      TheDriver.generateCompilationDiagnostics(*C, *FailingCommand);
      break;
    }
  }

  Diags.getClient()->finish();

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());

  llvm::llvm_shutdown();

#ifdef LLVM_ON_WIN32
  // Exit status should not be negative on Win32, unless abnormal termination.
  // Once abnormal termiation was caught, negative status should not be
  // propagated.
  if (Res < 0)
    Res = 1;
#endif

  // If we have multiple failing commands, we return the result of the first
  // failing command.
  return Res;
}
