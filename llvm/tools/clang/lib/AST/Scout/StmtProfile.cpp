// Note - this file is included by the StmtProfile source file
// one directory up (StmtProfile is all contained in a single
// file there...).
//

void StmtProfiler::VisitForallStmt(const ForallStmt *S)
{ VisitStmt(S); }

void StmtProfiler::VisitForallMeshStmt(const ForallMeshStmt *S)
{ VisitStmt(S); }



void StmtProfiler::VisitRenderallStmt(const RenderallStmt *S)
{ VisitStmt(S); }

void StmtProfiler::VisitRenderallMeshStmt(const RenderallMeshStmt *S)
{ VisitStmt(S); }
