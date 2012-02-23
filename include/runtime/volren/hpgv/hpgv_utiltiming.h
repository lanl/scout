/**
 * hpgv_utiltiming.h
 *
 * Copyright (c) Hongfeng Yu
 *
 * Contact:
 * Hongfeng Yu
 * hfstudio@gmail.com
 * 
 * 
 * All rights reserved.  May not be used, modified, or copied 
 * without permission.
 *
 */

#ifndef HPGV_UTILTIMING_H
#define HPGV_UTILTIMING_H

#include <mpi.h>


namespace scout {

  extern "C" {

#define HPGV_MAX_TIMING_UNIT     170

#define HPGV_TIMING_UNIT_0       0                 
#define HPGV_TIMING_UNIT_1       1
#define HPGV_TIMING_UNIT_2       2
#define HPGV_TIMING_UNIT_3       3                 
#define HPGV_TIMING_UNIT_4       4
#define HPGV_TIMING_UNIT_5       5
#define HPGV_TIMING_UNIT_6       6
#define HPGV_TIMING_UNIT_7       7
#define HPGV_TIMING_UNIT_8       8                 
#define HPGV_TIMING_UNIT_9       9

#define HPGV_TIMING_UNIT_10      10
#define HPGV_TIMING_UNIT_11      11                 
#define HPGV_TIMING_UNIT_12      12
#define HPGV_TIMING_UNIT_13      13
#define HPGV_TIMING_UNIT_14      14
#define HPGV_TIMING_UNIT_15      15
#define HPGV_TIMING_UNIT_16      16                 
#define HPGV_TIMING_UNIT_17      17
#define HPGV_TIMING_UNIT_18      18
#define HPGV_TIMING_UNIT_19      19

#define HPGV_TIMING_UNIT_20      20
#define HPGV_TIMING_UNIT_21      21
#define HPGV_TIMING_UNIT_22      22
#define HPGV_TIMING_UNIT_23      23
#define HPGV_TIMING_UNIT_24      24
#define HPGV_TIMING_UNIT_25      25
#define HPGV_TIMING_UNIT_26      26                 
#define HPGV_TIMING_UNIT_27      27
#define HPGV_TIMING_UNIT_28      28
#define HPGV_TIMING_UNIT_29      29                 

#define HPGV_TIMING_UNIT_30      30
#define HPGV_TIMING_UNIT_31      31
#define HPGV_TIMING_UNIT_32      32
#define HPGV_TIMING_UNIT_33      33
#define HPGV_TIMING_UNIT_34      34
#define HPGV_TIMING_UNIT_35      35
#define HPGV_TIMING_UNIT_36      36                 
#define HPGV_TIMING_UNIT_37      37
#define HPGV_TIMING_UNIT_38      38
#define HPGV_TIMING_UNIT_39      39                 

#define HPGV_TIMING_UNIT_40      40
#define HPGV_TIMING_UNIT_41      41
#define HPGV_TIMING_UNIT_42      42
#define HPGV_TIMING_UNIT_43      43
#define HPGV_TIMING_UNIT_44      44
#define HPGV_TIMING_UNIT_45      45
#define HPGV_TIMING_UNIT_46      46                 
#define HPGV_TIMING_UNIT_47      47
#define HPGV_TIMING_UNIT_48      48
#define HPGV_TIMING_UNIT_49      49 

#define HPGV_TIMING_UNIT_50      50
#define HPGV_TIMING_UNIT_51      51
#define HPGV_TIMING_UNIT_52      52
#define HPGV_TIMING_UNIT_53      53
#define HPGV_TIMING_UNIT_54      54
#define HPGV_TIMING_UNIT_55      55
#define HPGV_TIMING_UNIT_56      56                 
#define HPGV_TIMING_UNIT_57      57
#define HPGV_TIMING_UNIT_58      58
#define HPGV_TIMING_UNIT_59      59 

#define HPGV_TIMING_UNIT_60      60
#define HPGV_TIMING_UNIT_61      61
#define HPGV_TIMING_UNIT_62      62
#define HPGV_TIMING_UNIT_63      63
#define HPGV_TIMING_UNIT_64      64
#define HPGV_TIMING_UNIT_65      65
#define HPGV_TIMING_UNIT_66      66
#define HPGV_TIMING_UNIT_67      67
#define HPGV_TIMING_UNIT_68      68
#define HPGV_TIMING_UNIT_69      69

#define HPGV_TIMING_UNIT_70      70
#define HPGV_TIMING_UNIT_71      71
#define HPGV_TIMING_UNIT_72      72
#define HPGV_TIMING_UNIT_73      73
#define HPGV_TIMING_UNIT_74      74
#define HPGV_TIMING_UNIT_75      75
#define HPGV_TIMING_UNIT_76      76
#define HPGV_TIMING_UNIT_77      77
#define HPGV_TIMING_UNIT_78      78
#define HPGV_TIMING_UNIT_79      79

#define HPGV_TIMING_UNIT_80      80
#define HPGV_TIMING_UNIT_81      81
#define HPGV_TIMING_UNIT_82      82
#define HPGV_TIMING_UNIT_83      83
#define HPGV_TIMING_UNIT_84      84
#define HPGV_TIMING_UNIT_85      85
#define HPGV_TIMING_UNIT_86      86
#define HPGV_TIMING_UNIT_87      87
#define HPGV_TIMING_UNIT_88      88
#define HPGV_TIMING_UNIT_89      89

#define HPGV_TIMING_UNIT_90      90
#define HPGV_TIMING_UNIT_91      91
#define HPGV_TIMING_UNIT_92      92
#define HPGV_TIMING_UNIT_93      93
#define HPGV_TIMING_UNIT_94      94
#define HPGV_TIMING_UNIT_95      95
#define HPGV_TIMING_UNIT_96      96
#define HPGV_TIMING_UNIT_97      97
#define HPGV_TIMING_UNIT_98      98
#define HPGV_TIMING_UNIT_99      99

#define HPGV_TIMING_UNIT_100      100
#define HPGV_TIMING_UNIT_101      101
#define HPGV_TIMING_UNIT_102      102
#define HPGV_TIMING_UNIT_103      103
#define HPGV_TIMING_UNIT_104      104
#define HPGV_TIMING_UNIT_105      105
#define HPGV_TIMING_UNIT_106      106
#define HPGV_TIMING_UNIT_107      107
#define HPGV_TIMING_UNIT_108      108
#define HPGV_TIMING_UNIT_109      109

#define HPGV_TIMING_UNIT_110      110
#define HPGV_TIMING_UNIT_111      111
#define HPGV_TIMING_UNIT_112      112
#define HPGV_TIMING_UNIT_113      113
#define HPGV_TIMING_UNIT_114      114
#define HPGV_TIMING_UNIT_115      115
#define HPGV_TIMING_UNIT_116      116
#define HPGV_TIMING_UNIT_117      117
#define HPGV_TIMING_UNIT_118      118
#define HPGV_TIMING_UNIT_119      119

#define HPGV_TIMING_UNIT_120      120
#define HPGV_TIMING_UNIT_121      121
#define HPGV_TIMING_UNIT_122      122
#define HPGV_TIMING_UNIT_123      123
#define HPGV_TIMING_UNIT_124      124
#define HPGV_TIMING_UNIT_125      125
#define HPGV_TIMING_UNIT_126      126
#define HPGV_TIMING_UNIT_127      127
#define HPGV_TIMING_UNIT_128      128
#define HPGV_TIMING_UNIT_129      129

#define HPGV_TIMING_UNIT_130      130
#define HPGV_TIMING_UNIT_131      131
#define HPGV_TIMING_UNIT_132      132
#define HPGV_TIMING_UNIT_133      133
#define HPGV_TIMING_UNIT_134      134
#define HPGV_TIMING_UNIT_135      135
#define HPGV_TIMING_UNIT_136      136
#define HPGV_TIMING_UNIT_137      137
#define HPGV_TIMING_UNIT_138      138
#define HPGV_TIMING_UNIT_139      139

#define HPGV_TIMING_UNIT_140      140
#define HPGV_TIMING_UNIT_141      141
#define HPGV_TIMING_UNIT_142      142
#define HPGV_TIMING_UNIT_143      143
#define HPGV_TIMING_UNIT_144      144
#define HPGV_TIMING_UNIT_145      145
#define HPGV_TIMING_UNIT_146      146
#define HPGV_TIMING_UNIT_147      147
#define HPGV_TIMING_UNIT_148      148
#define HPGV_TIMING_UNIT_149      149

#define HPGV_TIMING_UNIT_150      150
#define HPGV_TIMING_UNIT_151      151
#define HPGV_TIMING_UNIT_152      152
#define HPGV_TIMING_UNIT_153      153
#define HPGV_TIMING_UNIT_154      154
#define HPGV_TIMING_UNIT_155      155
#define HPGV_TIMING_UNIT_156      156
#define HPGV_TIMING_UNIT_157      157
#define HPGV_TIMING_UNIT_158      158
#define HPGV_TIMING_UNIT_159      159

#define HPGV_TIMING_UNIT_160      160
#define HPGV_TIMING_UNIT_161      161
#define HPGV_TIMING_UNIT_162      162
#define HPGV_TIMING_UNIT_163      163
#define HPGV_TIMING_UNIT_164      164
#define HPGV_TIMING_UNIT_165      165
#define HPGV_TIMING_UNIT_166      166
#define HPGV_TIMING_UNIT_167      167
#define HPGV_TIMING_UNIT_168      168
#define HPGV_TIMING_UNIT_169      169

    void
      hpgv_timing_contextglobal(char *context);

    void
      hpgv_timing_contextlocal(char *context);

    void
      hpgv_timing_showlocal(int local);

    void
      hpgv_timing_showglobal(int global);

    void
      hpgv_timing_savelocal(int local); 

    void
      hpgv_timing_saveglobal(int global);

    void
      hpgv_timing_showbreakdown();

    void
      hpgv_timing_countroot(int countroot);

    int
      hpgv_timing_valid();

    void
      hpgv_timing_finalize();

    void
      hpgv_timing_init(int root, MPI_Comm mpicomm);         

    void
      hpgv_timing_name(int unit, char *name);

    void
      hpgv_timing_decrease(int unit, double value);

    void
      hpgv_timing_increase(int unit, double value);

    double
      hpgv_timing_get(int unit);

    double
      hpgv_timing_getgather(int proc, int unit);

    void
      hpgv_timing_set(int unit, double value);    

    void
      hpgv_timing_begin(int unit);

    void
      hpgv_timing_end(int unit);       

    void 
      hpgv_timing_count(int unit);

    void
      hpgv_timing_statistics();

#define HPGV_TIMING_SAVE_LOCAL  0
#define HPGV_TIMING_SHOW_LOCAL  1
#define HPGV_TIMING_SAVE_GLOBAL 2
#define HPGV_TIMING_SHOW_GLOBAL 3 


#ifdef HPGV_TIMING 
#define HPGV_TIMING_CONTEXTLOCAL(context) \
    hpgv_timing_contextlocal(context)

#define HPGV_TIMING_CONTEXTGLOBAL(context) \
    hpgv_timing_contextglobal(context)

#define HPGV_TIMING_SHOWGLOBAL(global) \
    hpgv_timing_showglobal(global)        

#define HPGV_TIMING_SHOWLOCAL(local) \
    hpgv_timing_showlocal(local)    

#define HPGV_TIMING_SAVEGLOBAL(global) \
    hpgv_timing_saveglobal(global)

#define HPGV_TIMING_SAVELOCAL(local) \
    hpgv_timing_savelocal(local)

#define HPGV_TIMING_SHOWBREAKDOWN() \
    hpgv_timing_showbreakdown()

#define HPGV_TIMING_COUNTROOT(countroot) \
    hpgv_timing_countroot(countroot)

#define HPGV_TIMING_VALID() \
    hpgv_timing_valid()

#define HPGV_TIMING_FINALIZE() {\
  hpgv_timing_finalize();\
}

#define HPGV_TIMING_INIT(root, mpicomm) {\
  hpgv_timing_init(root, mpicomm);\
}

#define HPGV_TIMING_NAME(unit, name) {\
  hpgv_timing_name(unit, name);\
}

#define HPGV_TIMING_DECREASE(unit, value) {\
  hpgv_timing_decrease(unit, value);\
}

#define HPGV_TIMING_INCREASE(unit, value) {\
  hpgv_timing_increase(unit, value);\
}

#define HPGV_TIMING_SET(unit, value) {\
  hpgv_timing_set(unit, value);\
}

#define HPGV_TIMING_GET(unit) hpgv_timing_get(unit)

#define HPGV_TIMING_GETGATHER(proc, unit) hpgv_timing_getgather(proc, unit)

#define HPGV_TIMING_BEGIN(unit) {\
  hpgv_timing_begin(unit);\
}

#define HPGV_TIMING_END(unit) {\
  hpgv_timing_end(unit);\
}

#define HPGV_TIMING_COUNT(unit) {\
  hpgv_timing_count(unit);\
}

#define HPGV_TIMING_STATISTICS() {\
  hpgv_timing_statistics();\
}

#define HPGV_TIMING_BARRIER(mpicomm) {\
  MPI_Barrier(mpicomm);\
}

#else
#define HPGV_TIMING_CONTEXTLOCAL(context)

#define HPGV_TIMING_CONTEXTGLOBAL(context)

#define HPGV_TIMING_SHOWGLOBAL(global)

#define HPGV_TIMING_SHOWLOCAL(local)

#define HPGV_TIMING_SAVEGLOBAL(global)

#define HPGV_TIMING_SAVELOCAL(local)

#define HPGV_TIMING_SHOWBREAKDOWN()

#define HPGV_TIMING_COUNTROOT(countroot) 

#define HPGV_TIMING_VALID() HPGV_FALSE

#define HPGV_TIMING_FINALIZE()

#define HPGV_TIMING_INIT(root, mpicomm)

#define HPGV_TIMING_NAME(unit, name)

#define HPGV_TIMING_DECREASE(unit, value)

#define HPGV_TIMING_INCREASE(unit, value)

#define HPGV_TIMING_GET(unit)

#define HPGV_TIMING_GETGATHER(proc, unit)

#define HPGV_TIMING_SET(unit, value)

#define HPGV_TIMING_BEGIN(unit)

#define HPGV_TIMING_END(unit)

#define HPGV_TIMING_COUNT(unit)

#define HPGV_TIMING_STATISTICS()

#define HPGV_TIMING_BARRIER(mpicomm)
#endif

      }

} // end namespace scout
    
#endif
