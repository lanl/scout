// Note - this file is included by the StmtProfile source file 
// one directory up (StmtProfile is all contained in a single
// file there...). 
//  

void StmtProfiler::VisitForAllStmt(const ForAllStmt *S) 
{ VisitStmt(S); }

void StmtProfiler::VisitForAllArrayStmt(const ForAllArrayStmt *S) 
{ VisitStmt(S); }

void StmtProfiler::VisitRenderAllStmt(const RenderAllStmt *S) 
{ VisitStmt(S); }

void StmtProfiler::VisitVolumeRenderAllStmt(const VolumeRenderAllStmt *S) 
{ VisitStmt(S); }

void StmtProfiler::VisitScoutVectorMemberExpr(const ScoutVectorMemberExpr *S)
{  
	// SC_TODO - we need to replace scout vectors with clang's "builtin"
	// version.  This has been done in the "refactor" branch but needs to
	// be merged w/ devel. 
}

